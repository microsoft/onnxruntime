#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/session/onnxruntime_float16.h"

#if defined(__GNUC__) && defined(__x86_64__)
#include <immintrin.h>
#define QMOE_SHARED_HAS_X86 1
#else
#define QMOE_SHARED_HAS_X86 0
#endif

namespace onnxruntime {
namespace contrib {
namespace qmoe_shared {

namespace {

struct BoundaryFloat16 : onnxruntime_float16::Float16Impl<BoundaryFloat16> {
  static float Round(float value) noexcept {
    BoundaryFloat16 result;
    result.val = ToUint16Impl(value);
    return result.ToFloatImpl();
  }
  static uint16_t ToBits(float value) noexcept { return ToUint16Impl(value); }
};

struct BoundaryBFloat16 : onnxruntime_float16::BFloat16Impl<BoundaryBFloat16> {
  static float Round(float value) noexcept {
    BoundaryBFloat16 result;
    result.val = ToUint16Impl(value);
    return result.ToFloatImpl();
  }
  static uint16_t ToBits(float value) noexcept { return ToUint16Impl(value); }
  static float FromBits(uint16_t bits) noexcept {
    BoundaryBFloat16 result;
    result.val = bits;
    return result.ToFloatImpl();
  }
};

// kPerm_W4_A16 from contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors_impl.h: stored row r of a
// 32-row K group holds original row kPerm[r].
constexpr int kPerm[32] = {0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
                           4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31};

// Nibble pair-interleave inside a 32-bit word: source nibble i lands at this position.
constexpr int NibbleDest(int i) { return (i % 2 == 0) ? i / 2 : (i - 1) / 2 + 4; }

alignas(64) constexpr float kFp4E2M1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

const std::array<float, 256>& GetE4M3LookupTable() {
  static const std::array<float, 256> table = []() {
    std::array<float, 256> t{};
    for (int code = 0; code < 256; ++code) {
      const uint32_t exp = (static_cast<uint32_t>(code) >> 3) & 0x0Fu;
      const uint32_t man = static_cast<uint32_t>(code) & 0x07u;
      float val = 0.0f;
      if (exp == 0u) {
        val = static_cast<float>(man) * 0.001953125f;  // 2^-9
      } else if (exp == 0x0Fu && man == 0x07u) {
        uint32_t nan_bits = 0x7fffffffu;
        std::memcpy(&val, &nan_bits, sizeof(val));
      } else {
        uint32_t bits = ((exp + 120u) << 23) | (man << 20);
        std::memcpy(&val, &bits, sizeof(val));
      }
      t[code] = (code & 0x80) ? -val : val;
    }
    return t;
  }();
  return table;
}

void ScalarGemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total,
                     int64_t k, const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin,
                     int64_t n_end) {
  const int64_t blocks = k / kBlockSize;
  for (int64_t n = n_begin; n < n_end; ++n) {
    const uint8_t* group = weights + (n / kColumnsInterleaved) * (kColumnsInterleaved * k / 2);
    const int64_t c = n % kColumnsInterleaved;
    for (int64_t r = 0; r < m; ++r) {
      float acc = 0.0f;
      for (int64_t kb = 0; kb < blocks; ++kb) {
        const uint8_t* bytes = group + (kb >> 1) * 128 + c * 32 + (kb & 1) * 16;
        const float* ar = a + r * k + kb * kBlockSize;
        float sum = 0.0f;
        for (int b = 0; b < 16; ++b) {
          sum += static_cast<float>((bytes[b] & 0x0F) - 8) * ar[b];
          sum += static_cast<float>((bytes[b] >> 4) - 8) * ar[16 + b];
        }
        acc += sum * HalfToFloat(scales[kb * n_total + n]);
      }
      if (bias != nullptr) acc += HalfToFloat(bias[n]);
      out[r * out_stride + (n - n_begin)] = acc;
    }
  }
}

inline float RoundBf16Fast(float v) noexcept {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & 0x7F800000u) != 0x7F800000u) {
    bits += 0x7FFFu + ((bits >> 16) & 1u);
  }
  bits &= 0xFFFF0000u;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

void ScalarNvfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                          const uint16_t* bias, bool bias_is_bf16, int64_t /*n_total*/, int64_t k,
                          const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                          int64_t n_begin, int64_t n_end, bool round_scale_bf16) {
  const auto& e4m3_lut = GetE4M3LookupTable();
  const int64_t k_pairs = k / 32;
  const int64_t weight_row_stride = k / 2;
  const int64_t scale_row_stride = k / 16;
  for (int64_t n = n_begin; n < n_end; ++n) {
    const uint8_t* w_row = weights + n * weight_row_stride;
    const uint8_t* s_row = block_scales + n * scale_row_stride;
    const float b_val = (bias != nullptr)
                            ? (bias_is_bf16 ? BoundaryBFloat16::FromBits(bias[n]) : HalfToFloat(bias[n]))
                            : 0.0f;
    for (int64_t r = 0; r < m; ++r) {
      const float* ar = a_reordered + r * k;
      float acc = 0.0f;
      for (int64_t kp = 0; kp < k_pairs; ++kp) {
        const uint8_t* bytes = w_row + kp * 16;
        const float raw_s0 = e4m3_lut[s_row[kp * 2]] * global_scale;
        const float raw_s1 = e4m3_lut[s_row[kp * 2 + 1]] * global_scale;
        const float s0 = round_scale_bf16 ? BoundaryBFloat16::Round(raw_s0) : BoundaryFloat16::Round(raw_s0);
        const float s1 = round_scale_bf16 ? BoundaryBFloat16::Round(raw_s1) : BoundaryFloat16::Round(raw_s1);
        const float* al = ar + kp * 32;
        const float* ah = al + 16;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int b = 0; b < 8; ++b) {
          const float wl0 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b] & 0x0F] * s0)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b] & 0x0F] * s0);
          const float wh0 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b] >> 4] * s0)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b] >> 4] * s0);
          const float wl1 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b + 8] & 0x0F] * s1)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b + 8] & 0x0F] * s1);
          const float wh1 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b + 8] >> 4] * s1)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b + 8] >> 4] * s1);
          sum0 += wl0 * al[b] + wh0 * ah[b];
          sum1 += wl1 * al[b + 8] + wh1 * ah[b + 8];
        }
        acc += sum0 + sum1;
      }
      out[r * out_stride + (n - n_begin)] = acc + b_val;
    }
  }
}

#if QMOE_SHARED_HAS_X86

template <int RT>
__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void Group4Avx512(
    const uint8_t* group, const uint8_t* next_group, const uint16_t* scales, int64_t n_total, int64_t n0, int64_t k,
    const float* a, float (&result)[4][RT]) {
  const int64_t blocks = k / kBlockSize;
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m512i eight = _mm512_set1_epi32(8);
  __m512 acc[4][RT];
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < RT; ++r) acc[c][r] = _mm512_setzero_ps();

  for (int64_t kb = 0; kb < blocks; ++kb) {
    const uint8_t* tile = group + (kb >> 1) * 128 + (kb & 1) * 16;
    const uint16_t* sc = scales + kb * n_total + n0;
    if (next_group != nullptr) _mm_prefetch(reinterpret_cast<const char*>(next_group + kb * 64), _MM_HINT_T0);
    __m512 al[RT];
    __m512 ah[RT];
    for (int r = 0; r < RT; ++r) {
      al[r] = _mm512_loadu_ps(a + r * k + kb * kBlockSize);
      ah[r] = _mm512_loadu_ps(a + r * k + kb * kBlockSize + 16);
    }
    for (int c = 0; c < 4; ++c) {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + c * 32));
      const __m128i lo = _mm_and_si128(x, mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
      const __m512 wl = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), eight));
      const __m512 wh = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), eight));
      const __m512 s = _mm512_set1_ps(_cvtsh_ss(sc[c]));
      for (int r = 0; r < RT; ++r) {
        const __m512 prod = _mm512_fmadd_ps(wl, al[r], _mm512_mul_ps(wh, ah[r]));
        acc[c][r] = _mm512_fmadd_ps(prod, s, acc[c][r]);
      }
    }
  }
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < RT; ++r) result[c][r] = _mm512_reduce_add_ps(acc[c][r]);
}

template <int RT>
__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void RowTileAvx512(
    const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total, int64_t k,
    const float* a, float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  for (int64_t n0 = n_begin; n0 < n_end; n0 += kColumnsInterleaved) {
    const uint8_t* group = weights + (n0 / kColumnsInterleaved) * (kColumnsInterleaved * k / 2);
    float result[4][RT];
    const uint8_t* next_group = (n0 + kColumnsInterleaved < n_end) ? group + kColumnsInterleaved * k / 2 : nullptr;
    Group4Avx512<RT>(group, next_group, scales, n_total, n0, k, a, result);
    for (int c = 0; c < 4; ++c) {
      const float b = bias != nullptr ? _cvtsh_ss(bias[n0 + c]) : 0.0f;
      for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void Avx512GemmRange(
    const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total, int64_t k,
    const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin, int64_t n_end) {
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a + r * k;
    float* orow = out + r * out_stride;
    if (left >= 4) {
      RowTileAvx512<4>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 4;
    } else if (left == 3) {
      RowTileAvx512<3>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 3;
    } else if (left == 2) {
      RowTileAvx512<2>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 2;
    } else {
      RowTileAvx512<1>(weights, scales, bias, n_total, k, ar, orow, out_stride, n_begin, n_end);
      r += 1;
    }
  }
}

template <int RT>
__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void Nvfp4Group4Avx512(
    const uint8_t* w_base, const uint8_t* s_base, const float (&scaled_lut)[256], int64_t k,
    const float* a_reordered, float (&result)[4][RT]) {
  const int64_t k_pairs = k / 32;
  const int64_t w_stride = k / 2;
  const int64_t s_stride = k / 16;
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m512 fp4_lut = _mm512_load_ps(kFp4E2M1Table);

  __m512 acc[4][RT];
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < RT; ++r) acc[c][r] = _mm512_setzero_ps();

  const uint8_t* w_rows[4] = {w_base, w_base + w_stride, w_base + 2 * w_stride, w_base + 3 * w_stride};
  const uint8_t* s_rows[4] = {s_base, s_base + s_stride, s_base + 2 * s_stride, s_base + 3 * s_stride};

  for (int64_t kp = 0; kp < k_pairs; ++kp) {
    __m512 al[RT];
    __m512 ah[RT];
    for (int r = 0; r < RT; ++r) {
      al[r] = _mm512_loadu_ps(a_reordered + r * k + kp * 32);
      ah[r] = _mm512_loadu_ps(a_reordered + r * k + kp * 32 + 16);
    }
    for (int c = 0; c < 4; ++c) {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_rows[c] + kp * 16));
      const __m128i lo = _mm_and_si128(x, mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
      const __m512 wl = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(lo), fp4_lut);
      const __m512 wh = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(hi), fp4_lut);

      const uint8_t sc0 = s_rows[c][kp * 2];
      const uint8_t sc1 = s_rows[c][kp * 2 + 1];
      const __m512 s = _mm512_mask_broadcastss_ps(_mm512_set1_ps(scaled_lut[sc0]), static_cast<__mmask16>(0xFF00u),
                                                  _mm_set_ss(scaled_lut[sc1]));

      for (int r = 0; r < RT; ++r) {
        const __m512 prod = _mm512_fmadd_ps(wl, al[r], _mm512_mul_ps(wh, ah[r]));
        acc[c][r] = _mm512_fmadd_ps(prod, s, acc[c][r]);
      }
    }
  }
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < RT; ++r) result[c][r] = _mm512_reduce_add_ps(acc[c][r]);
}

template <int RT>
__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void Nvfp4RowTileAvx512(
    const uint8_t* weights, const uint8_t* block_scales, float global_scale, const float (&scaled_lut)[256],
    const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
    const float* a_reordered, float* out, int64_t out_stride, int64_t n_begin, int64_t n_end,
    bool round_scale_bf16) {
  const int64_t w_stride = k / 2;
  const int64_t s_stride = k / 16;
  int64_t n0 = n_begin;
  for (; n0 + 4 <= n_end; n0 += 4) {
    float result[4][RT];
    Nvfp4Group4Avx512<RT>(weights + n0 * w_stride, block_scales + n0 * s_stride, scaled_lut, k,
                          a_reordered, result);
    for (int c = 0; c < 4; ++c) {
      const float b = (bias != nullptr)
                          ? (bias_is_bf16 ? BoundaryBFloat16::FromBits(bias[n0 + c]) : _cvtsh_ss(bias[n0 + c]))
                          : 0.0f;
      for (int r = 0; r < RT; ++r) out[r * out_stride + (n0 + c - n_begin)] = result[c][r] + b;
    }
  }
  if (n0 < n_end) {
    ScalarNvfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                         a_reordered, RT, out + (n0 - n_begin), out_stride, n0, n_end, round_scale_bf16);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) void Avx512Nvfp4GemmRange(
    const uint8_t* weights, const uint8_t* block_scales, float global_scale,
    const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
    const float* a_reordered, int64_t m, float* out, int64_t out_stride,
    int64_t n_begin, int64_t n_end, bool round_scale_bf16) {
  const auto* e4m3_lut = GetE4M3LookupTable().data();
  float scaled_lut[256];
  if (round_scale_bf16) {
    for (int i = 0; i < 256; ++i) scaled_lut[i] = RoundBf16Fast(e4m3_lut[i] * global_scale);
  } else {
    for (int i = 0; i < 256; ++i) scaled_lut[i] = _cvtsh_ss(_cvtss_sh(e4m3_lut[i] * global_scale, _MM_FROUND_TO_NEAREST_INT));
  }
  int64_t r = 0;
  while (r < m) {
    const int64_t left = m - r;
    const float* ar = a_reordered + r * k;
    float* orow = out + r * out_stride;
    if (left >= 4) {
      Nvfp4RowTileAvx512<4>(weights, block_scales, global_scale, scaled_lut, bias, bias_is_bf16, n_total, k,
                            ar, orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 4;
    } else if (left == 3) {
      Nvfp4RowTileAvx512<3>(weights, block_scales, global_scale, scaled_lut, bias, bias_is_bf16, n_total, k,
                            ar, orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 3;
    } else if (left == 2) {
      Nvfp4RowTileAvx512<2>(weights, block_scales, global_scale, scaled_lut, bias, bias_is_bf16, n_total, k,
                            ar, orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 2;
    } else {
      Nvfp4RowTileAvx512<1>(weights, block_scales, global_scale, scaled_lut, bias, bias_is_bf16, n_total, k,
                            ar, orow, out_stride, n_begin, n_end, round_scale_bf16);
      r += 1;
    }
  }
}

#endif  // QMOE_SHARED_HAS_X86

}  // namespace

bool CpuHasAvx512() {
#if QMOE_SHARED_HAS_X86
  static const bool has = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("f16c");
  return has;
#else
  return false;
#endif
}

bool IsAcceptedPrepackContract(int64_t weights_prepacked, int64_t block_size) {
  return weights_prepacked == kRequiredWeightsPrepacked && block_size == kBlockSize;
}

bool IsAcceptedNvfp4PrepackContract(int64_t weights_prepacked, int64_t block_size) {
  return weights_prepacked == kRequiredWeightsPrepacked && block_size == kNvfp4BlockSize;
}

size_t ComputePointerAlignment(const void* ptr) noexcept {
  if (ptr == nullptr) return 0;
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  if (addr == 0) return 0;
  const uintptr_t low_bit = addr & (~addr + 1);
  return static_cast<size_t>( std::min<uintptr_t>(low_bit, 4096) );
}

bool MakeExpertBlockLease(const void* ptr, size_t length, size_t min_required_alignment,
                          ExpertBlockLease* lease) noexcept {
  if (lease == nullptr || ptr == nullptr || length == 0) return false;
  if (min_required_alignment == 0 || (min_required_alignment & (min_required_alignment - 1)) != 0) return false;
  const size_t align = ComputePointerAlignment(ptr);
  if (align < min_required_alignment) return false;
  lease->data = static_cast<const uint8_t*>(ptr);
  lease->length = length;
  lease->alignment = align;
  return true;
}

bool ValidateExpertStorageRange(int64_t num_experts, size_t stride, uint64_t external_offset,
                                uint64_t external_length, size_t* total_bytes) noexcept {
  if (num_experts <= 0 || num_experts > static_cast<int64_t>(std::numeric_limits<int>::max()) || stride == 0) {
    return false;
  }
  size_t total = 0;
  if (!CheckedMulSize(static_cast<size_t>(num_experts), stride, &total)) return false;
  if (external_length != 0) {
    if (external_length != static_cast<uint64_t>(total)) return false;
    if (external_offset > std::numeric_limits<uint64_t>::max() - external_length) return false;
  }
  if (total_bytes != nullptr) *total_bytes = total;
  return true;
}

bool MakeExpertBlockLayout(int64_t hidden, int64_t inter, ExpertBlockLayout* layout) {
  if (layout == nullptr || hidden <= 0 || inter <= 0) return false;
  // Enforce backend int32 index limits and prevent 2 * inter overflow before multiplication (F12).
  if (hidden > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
      inter > static_cast<int64_t>(std::numeric_limits<int>::max() / 2)) {
    return false;
  }
  if (hidden % kRowsPerTile != 0 || inter % kRowsPerTile != 0) return false;
  const size_t h = static_cast<size_t>(hidden);
  const size_t i = static_cast<size_t>(inter);
  size_t n1 = 0;
  if (!CheckedMulSize(2, i, &n1)) return false;
  if (n1 % static_cast<size_t>(kColumnsInterleaved) != 0 ||
      h % static_cast<size_t>(kColumnsInterleaved) != 0) {
    return false;
  }

  auto add_bank = [](size_t current_off, size_t raw_bytes, size_t* bank_off, size_t* next_off) -> bool {
    *bank_off = current_off;
    size_t aligned = 0;
    if (!CheckedAlignUp(raw_bytes, kBankAlignment, &aligned)) return false;
    return CheckedAddSize(current_off, aligned, next_off);
  };

  ExpertBlockLayout l;
  l.hidden = hidden;
  l.inter = inter;
  size_t off = 0;
  size_t bytes = 0;

  if (!CheckedMulSize(n1, h / 2, &bytes) || !add_bank(off, bytes, &l.fc1_weights, &off)) return false;
  size_t fc1_s_elems = 0;
  if (!CheckedMulSize(h / static_cast<size_t>(kBlockSize), n1, &fc1_s_elems) ||
      !CheckedMulSize(fc1_s_elems, 2, &bytes) || !add_bank(off, bytes, &l.fc1_scales, &off)) {
    return false;
  }
  if (!CheckedMulSize(n1, 2, &bytes) || !add_bank(off, bytes, &l.fc1_bias, &off)) return false;
  if (!CheckedMulSize(h, i / 2, &bytes) || !add_bank(off, bytes, &l.fc2_weights, &off)) return false;
  size_t fc2_s_elems = 0;
  if (!CheckedMulSize(i / static_cast<size_t>(kBlockSize), h, &fc2_s_elems) ||
      !CheckedMulSize(fc2_s_elems, 2, &bytes) || !add_bank(off, bytes, &l.fc2_scales, &off)) {
    return false;
  }
  if (!CheckedMulSize(h, 2, &bytes) || !add_bank(off, bytes, &l.fc2_bias, &off)) return false;

  l.stride = off;
  *layout = l;
  return true;
}

bool MakeNvfp4ExpertBlockLayout(int64_t hidden, int64_t inter, Nvfp4ExpertBlockLayout* layout) {
  if (layout == nullptr || hidden <= 0 || inter <= 0) return false;
  if (hidden > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
      inter > static_cast<int64_t>(std::numeric_limits<int>::max() / 2)) {
    return false;
  }
  if (hidden % 32 != 0 || inter % 32 != 0) return false;
  const size_t h = static_cast<size_t>(hidden);
  const size_t i = static_cast<size_t>(inter);
  size_t n1 = 0;
  if (!CheckedMulSize(2, i, &n1)) return false;

  auto add_bank = [](size_t current_off, size_t raw_bytes, size_t* bank_off, size_t* next_off) -> bool {
    *bank_off = current_off;
    size_t aligned = 0;
    if (!CheckedAlignUp(raw_bytes, kBankAlignment, &aligned)) return false;
    return CheckedAddSize(current_off, aligned, next_off);
  };

  Nvfp4ExpertBlockLayout l;
  l.hidden = hidden;
  l.inter = inter;
  size_t off = 0;
  size_t bytes = 0;

  if (!CheckedMulSize(n1, h / 2, &bytes) || !add_bank(off, bytes, &l.fc1_weights, &off)) return false;
  if (!CheckedMulSize(n1, h / 16, &bytes) || !add_bank(off, bytes, &l.fc1_scales, &off)) return false;
  if (!add_bank(off, sizeof(float), &l.fc1_global_scale, &off)) return false;
  if (!CheckedMulSize(h, i / 2, &bytes) || !add_bank(off, bytes, &l.fc2_weights, &off)) return false;
  if (!CheckedMulSize(h, i / 16, &bytes) || !add_bank(off, bytes, &l.fc2_scales, &off)) return false;
  if (!add_bank(off, sizeof(float), &l.fc2_global_scale, &off)) return false;

  l.stride = off;
  *layout = l;
  return true;
}

int ReorderedPosition(int k_offset_in_block) {
  for (int b = 0; b < 16; ++b) {
    const int lo = LowNibbleK(b);
    if (k_offset_in_block == lo) return b;
    if (k_offset_in_block == lo + 8) return 16 + b;
  }
  return -1;
}

void ReorderActivationRow(const float* src, float* dst, int64_t k) {
  for (int64_t kb = 0; kb < k / kBlockSize; ++kb) {
    const float* s = src + kb * kBlockSize;
    float* d = dst + kb * kBlockSize;
    for (int b = 0; b < 16; ++b) {
      d[b] = s[LowNibbleK(b)];
      d[16 + b] = s[LowNibbleK(b) + 8];
    }
  }
}

void ReorderNvfp4ActivationRow(const float* src, float* dst, int64_t k) {
  const int64_t k_pairs = k / 32;
  for (int64_t kp = 0; kp < k_pairs; ++kp) {
    const float* s = src + kp * 32;
    float* d = dst + kp * 32;
    for (int b = 0; b < 8; ++b) {
      d[b] = s[2 * b];
      d[8 + b] = s[16 + 2 * b];
      d[16 + b] = s[2 * b + 1];
      d[24 + b] = s[16 + 2 * b + 1];
    }
  }
}

uint8_t WeightCodeAt(const uint8_t* weights, int64_t n_total, int64_t k_total, int64_t n, int64_t k) {
  (void)n_total;
  int inv[32];
  for (int r = 0; r < 32; ++r) inv[kPerm[r]] = r;
  const int64_t tile64 = k / kRowsPerTile;
  const int64_t stored_k = (k / 32) * 32 + inv[k % 32];
  const int64_t word_in_col = stored_k / 8;
  const int pos = NibbleDest(static_cast<int>(stored_k % 8));
  const int64_t group = n / kColumnsInterleaved;
  const int64_t col = n % kColumnsInterleaved;
  const int64_t word = group * kColumnsInterleaved * (k_total / 8) + 32 * tile64 + 8 * col + (word_in_col % 8);
  const uint8_t byte = weights[word * 4 + pos / 2];
  return (pos % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t bits = 0;
  if (exp == 0) {
    if (man != 0) {
      uint32_t m = man;
      uint32_t e = 0;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++e;
      }
      bits = (sign << 31) | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 0x1F) {
    bits = (sign << 31) | 0x7F800000u | (man << 13);
  } else {
    bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t FloatToHalf(float f) { return BoundaryFloat16::ToBits(f); }
float BFloat16ToFloat(uint16_t b) { return BoundaryBFloat16::FromBits(b); }
uint16_t FloatToBFloat16(float f) { return BoundaryBFloat16::ToBits(f); }
float DecodeFp4E2M1(uint8_t code) { return kFp4E2M1Table[code & 0x0F]; }
float DecodeFloat8E4M3FN(uint8_t code) { return GetE4M3LookupTable()[code]; }

void GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total,
               int64_t k, const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin,
               int64_t n_end, bool force_scalar) {
#if QMOE_SHARED_HAS_X86
  if (!force_scalar && CpuHasAvx512()) {
    Avx512GemmRange(weights, scales, bias, n_total, k, a, m, out, out_stride, n_begin, n_end);
    return;
  }
#endif
  (void)force_scalar;
  ScalarGemmRange(weights, scales, bias, n_total, k, a, m, out, out_stride, n_begin, n_end);
}

void Fc1SwiGluRange(const ExpertBlockLayout& layout, const uint8_t* block, const float* x_reordered,
                    int64_t m, const SwiGluParams& swiglu, float* inter_reordered, int64_t n_begin,
                    int64_t n_end, bool force_scalar, bool fp16_boundaries) {
  constexpr int64_t kChunk = 64;
  float tmp[kMaxRowsPerCall * kChunk];
  const int64_t n_total = 2 * layout.inter;
  const uint8_t* w = block + layout.fc1_weights;
  const uint16_t* s = reinterpret_cast<const uint16_t*>(block + layout.fc1_scales);
  const uint16_t* bias = reinterpret_cast<const uint16_t*>(block + layout.fc1_bias);
  for (int64_t nb = n_begin; nb < n_end; nb += kChunk) {
    const int64_t ne = std::min(nb + kChunk, n_end);
    GemmRange(w, s, bias, n_total, layout.hidden, x_reordered, m, tmp, kChunk, nb, ne, force_scalar);
    for (int64_t n = nb; n < ne; n += 2) {
      const int64_t j = n / 2;
      const int pos = ReorderedPosition(static_cast<int>(j % 32));
      for (int64_t r = 0; r < m; ++r) {
        const float gate_projection = tmp[r * kChunk + (n - nb)];
        const float up_projection = tmp[r * kChunk + (n - nb) + 1];
        const float gate = std::min(fp16_boundaries ? BoundaryFloat16::Round(gate_projection) : gate_projection,
                                    swiglu.limit);
        const float value = std::clamp(fp16_boundaries ? BoundaryFloat16::Round(up_projection) : up_projection,
                                       -swiglu.limit, swiglu.limit);
        const float act = gate / (1.0f + std::exp(-swiglu.alpha * gate)) * (value + swiglu.beta);
        inter_reordered[r * layout.inter + (j / 32) * 32 + pos] = fp16_boundaries ? BoundaryFloat16::Round(act) : act;
      }
    }
  }
}

void Fc2Range(const ExpertBlockLayout& layout, const uint8_t* block, const float* inter_reordered,
              int64_t m, float* out, int64_t n_begin, int64_t n_end, bool force_scalar, bool apply_bias) {
  GemmRange(block + layout.fc2_weights, reinterpret_cast<const uint16_t*>(block + layout.fc2_scales),
            apply_bias ? reinterpret_cast<const uint16_t*>(block + layout.fc2_bias) : nullptr, layout.hidden, layout.inter,
            inter_reordered, m, out + n_begin, layout.hidden, n_begin, n_end, force_scalar);
}

bool ExpertForward(const ExpertBlockLayout& layout, const uint8_t* block, const float* x, int64_t m,
                   const SwiGluParams& swiglu, float* out, bool force_scalar, bool apply_fc2_bias, bool fp16_boundaries) {
  if (m < 1 || m > kMaxRowsPerCall || layout.hidden <= 0) return false;
  std::vector<float> xr(static_cast<size_t>(m * layout.hidden));
  std::vector<float> inter(static_cast<size_t>(m * layout.inter));
  for (int64_t r = 0; r < m; ++r) ReorderActivationRow(x + r * layout.hidden, xr.data() + r * layout.hidden, layout.hidden);
  Fc1SwiGluRange(layout, block, xr.data(), m, swiglu, inter.data(), 0, 2 * layout.inter, force_scalar, fp16_boundaries);
  Fc2Range(layout, block, inter.data(), m, out, 0, layout.hidden, force_scalar, apply_fc2_bias);
  return true;
}

void Nvfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                    const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                    const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                    int64_t n_begin, int64_t n_end, bool force_scalar, bool round_scale_bf16) {
#if QMOE_SHARED_HAS_X86
  if (!force_scalar && CpuHasAvx512()) {
    Avx512Nvfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                         a_reordered, m, out, out_stride, n_begin, n_end, round_scale_bf16);
    return;
  }
#endif
  (void)force_scalar;
  ScalarNvfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                       a_reordered, m, out, out_stride, n_begin, n_end, round_scale_bf16);
}

void Nvfp4Fc1SwiGluRange(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                         const float* x_reordered, int64_t m, const SwiGluParams& swiglu,
                         float* inter_reordered, int64_t n_begin, int64_t n_end,
                         bool force_scalar, bool bf16_boundaries, bool fp16_boundaries) {
  constexpr int64_t kChunk = 64;
  float tmp[kMaxRowsPerCall * kChunk];
  const int64_t n_total = 2 * inter;
  const bool round_scale_bf16 = !fp16_boundaries;
  for (int64_t nb = n_begin; nb < n_end; nb += kChunk) {
    const int64_t ne = std::min(nb + kChunk, n_end);
    Nvfp4GemmRange(view.fc1_weights, view.fc1_scales, view.fc1_global_scale,
                   view.fc1_bias, view.bias_is_bf16, n_total, hidden, x_reordered, m,
                   tmp, kChunk, nb, ne, force_scalar, round_scale_bf16);
    for (int64_t n = nb; n < ne; n += 2) {
      const int64_t j = n / 2;
      const int pos = Nvfp4ReorderedPosition(static_cast<int>(j % 32));
      for (int64_t r = 0; r < m; ++r) {
        const float gate_projection = tmp[r * kChunk + (n - nb)];
        const float up_projection = tmp[r * kChunk + (n - nb) + 1];
        float gate = gate_projection;
        float value = up_projection;
        if (std::isfinite(swiglu.limit)) {
          gate = std::min(gate, swiglu.limit);
          value = std::clamp(value, -swiglu.limit, swiglu.limit);
        }
        const float act = gate / (1.0f + std::exp(-swiglu.alpha * gate)) * (value + swiglu.beta);
        const float rounded_act = bf16_boundaries   ? BoundaryBFloat16::Round(act)
                                  : fp16_boundaries ? BoundaryFloat16::Round(act)
                                                    : act;
        inter_reordered[r * inter + (j / 32) * 32 + pos] = rounded_act;
      }
    }
  }
}

void Nvfp4Fc2Range(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                   const float* inter_reordered, int64_t m, float* out,
                   int64_t n_begin, int64_t n_end, bool force_scalar,
                   bool apply_bias, bool round_scale_bf16) {
  Nvfp4GemmRange(view.fc2_weights, view.fc2_scales, view.fc2_global_scale,
                 apply_bias ? view.fc2_bias : nullptr, view.bias_is_bf16,
                 hidden, inter, inter_reordered, m, out + n_begin, hidden,
                 n_begin, n_end, force_scalar, round_scale_bf16);
}

bool Nvfp4ExpertForward(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                        const float* x, int64_t m, const SwiGluParams& swiglu, float* out,
                        bool force_scalar, bool apply_fc2_bias,
                        bool bf16_boundaries, bool fp16_boundaries) {
  if (m < 1 || m > kMaxRowsPerCall || hidden <= 0 || inter <= 0 || hidden % 32 != 0 || inter % 32 != 0) {
    return false;
  }
  if (!view.fc1_weights || !view.fc1_scales || !view.fc2_weights || !view.fc2_scales) return false;
  std::vector<float> xr(static_cast<size_t>(m * hidden));
  std::vector<float> inter_buf(static_cast<size_t>(m * inter));
  for (int64_t r = 0; r < m; ++r) {
    ReorderNvfp4ActivationRow(x + r * hidden, xr.data() + r * hidden, hidden);
  }
  Nvfp4Fc1SwiGluRange(hidden, inter, view, xr.data(), m, swiglu, inter_buf.data(),
                      0, 2 * inter, force_scalar, bf16_boundaries, fp16_boundaries);
  Nvfp4Fc2Range(hidden, inter, view, inter_buf.data(), m, out,
                0, hidden, force_scalar, apply_fc2_bias, !fp16_boundaries);
  return true;
}

}  // namespace qmoe_shared
}  // namespace contrib
}  // namespace onnxruntime

using namespace onnxruntime::contrib::qmoe_shared;

extern "C" {
int qmoe_shared_expert_forward(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                               int64_t m, float* out, float alpha, float beta, float limit,
                               int force_scalar) {
  ExpertBlockLayout layout;
  if (!MakeExpertBlockLayout(hidden, inter, &layout)) return 0;
  SwiGluParams p{alpha, beta, limit};
  return ExpertForward(layout, block, x, m, p, out, force_scalar != 0) ? 1 : 0;
}

int qmoe_shared_expert_contribution(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                                    int64_t m, float* out, float alpha, float beta, float limit,
                                    int force_scalar) {
  ExpertBlockLayout layout;
  if (!MakeExpertBlockLayout(hidden, inter, &layout)) return 0;
  SwiGluParams params{alpha, beta, limit};
  return ExpertForward(layout, block, x, m, params, out, force_scalar != 0, false) ? 1 : 0;
}

int qmoe_shared_expert_contribution_fp16(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                                         int64_t m, float* out, float alpha, float beta, float limit,
                                         int force_scalar) {
  ExpertBlockLayout layout;
  if (!MakeExpertBlockLayout(hidden, inter, &layout)) return 0;
  SwiGluParams params{alpha, beta, limit};
  return ExpertForward(layout, block, x, m, params, out, force_scalar != 0, false, true) ? 1 : 0;
}

int64_t qmoe_shared_block_bytes(int64_t hidden, int64_t inter) {
  ExpertBlockLayout layout;
  return MakeExpertBlockLayout(hidden, inter, &layout) ? static_cast<int64_t>(layout.stride) : -1;
}

int qmoe_shared_cpu_has_avx512(void) { return CpuHasAvx512() ? 1 : 0; }

int qmoe_shared_accepts_prepack_contract(int64_t weights_prepacked, int64_t block_size) {
  return IsAcceptedPrepackContract(weights_prepacked, block_size) ? 1 : 0;
}

int qmoe_shared_weight_code_at(const uint8_t* weights, int64_t n_total, int64_t k_total, int64_t n,
                               int64_t k) {
  return WeightCodeAt(weights, n_total, k_total, n, k);
}

int qmoe_nvfp4_expert_contribution_bf16(const uint8_t* fc1_w, const uint8_t* fc1_s, float fc1_g,
                                        const uint8_t* fc2_w, const uint8_t* fc2_s, float fc2_g,
                                        int64_t hidden, int64_t inter, const float* x, int64_t m,
                                        float* out, float alpha, float beta, float limit,
                                        int force_scalar) {
  Nvfp4ExpertView view{};
  view.fc1_weights = fc1_w;
  view.fc1_scales = fc1_s;
  view.fc1_global_scale = fc1_g;
  view.fc2_weights = fc2_w;
  view.fc2_scales = fc2_s;
  view.fc2_global_scale = fc2_g;
  view.bias_is_bf16 = true;
  SwiGluParams params{alpha, beta, limit};
  return Nvfp4ExpertForward(hidden, inter, view, x, m, params, out, force_scalar != 0,
                            /*apply_fc2_bias=*/false, /*bf16_boundaries=*/true, /*fp16_boundaries=*/false)
             ? 1
             : 0;
}

int64_t qmoe_nvfp4_block_bytes(int64_t hidden, int64_t inter) {
  Nvfp4ExpertBlockLayout layout;
  return MakeNvfp4ExpertBlockLayout(hidden, inter, &layout) ? static_cast<int64_t>(layout.stride) : -1;
}
}
