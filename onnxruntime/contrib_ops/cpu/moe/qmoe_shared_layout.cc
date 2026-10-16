#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <memory>
#include <sstream>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#endif
#endif

#include "core/session/onnxruntime_float16.h"

#if defined(QMOE_SHARED_USE_MLAS)
#include "core/mlas/inc/mlas.h"
#endif

#if !defined(QMOE_COMPILER_SUPPORTS_AVX512)
#if defined(__AVX512F__) || (defined(__GNUC__) && defined(__x86_64__))
#define QMOE_COMPILER_SUPPORTS_AVX512 1
#else
#define QMOE_COMPILER_SUPPORTS_AVX512 0
#endif
#endif

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(__x86_64__)
#define QMOE_SHARED_ARCH_X86_64 1
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif
#else
#define QMOE_SHARED_ARCH_X86_64 0
#endif

namespace onnxruntime {
namespace contrib {
namespace qmoe_shared {

#if QMOE_COMPILER_SUPPORTS_AVX512
namespace detail {
void Avx512GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total, int64_t k,
                     const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin, int64_t n_end);
void Avx512Nvfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                          const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                          const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                          int64_t n_begin, int64_t n_end, bool round_scale_bf16);
void Avx512Mxfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                          const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                          const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                          int64_t n_begin, int64_t n_end, bool round_scale_bf16);
void Avx512HalfBitsToFloatRow(const uint16_t* src, float* dst, int64_t count);
void Avx512FloatToHalfBitsRow(const float* src, uint16_t* dst, int64_t count);
void Avx512BFloat16BitsToFloatRow(const uint16_t* src, float* dst, int64_t count);
void Avx512FloatToBFloat16BitsRow(const float* src, uint16_t* dst, int64_t count);
}  // namespace detail
#endif

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

const std::array<float, 256>& GetE8M0LookupTable() {
  static const std::array<float, 256> table = []() {
    std::array<float, 256> t{};
    for (int code = 0; code < 256; ++code) {
      if (code == 255) {
        uint32_t nan_bits = 0x7fffffffu;
        std::memcpy(&t[code], &nan_bits, sizeof(float));
      } else {
        t[code] = std::ldexp(1.0f, code - 127);
      }
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

void ScalarMxfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                          const uint16_t* bias, bool bias_is_bf16, int64_t /*n_total*/, int64_t k,
                          const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                          int64_t n_begin, int64_t n_end, bool round_scale_bf16) {
  const auto& e8m0_lut = GetE8M0LookupTable();
  const int64_t k_pairs = k / 32;
  const int64_t weight_row_stride = k / 2;
  const int64_t scale_row_stride = k / 32;
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
        const float raw_s = e8m0_lut[s_row[kp]] * global_scale;
        const float s = round_scale_bf16 ? BoundaryBFloat16::Round(raw_s) : BoundaryFloat16::Round(raw_s);
        const float* al = ar + kp * 32;
        const float* ah = al + 16;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int b = 0; b < 8; ++b) {
          const float wl0 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b] & 0x0F] * s)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b] & 0x0F] * s);
          const float wh0 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b] >> 4] * s)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b] >> 4] * s);
          const float wl1 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b + 8] & 0x0F] * s)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b + 8] & 0x0F] * s);
          const float wh1 = round_scale_bf16 ? RoundBf16Fast(kFp4E2M1Table[bytes[b + 8] >> 4] * s)
                                             : BoundaryFloat16::Round(kFp4E2M1Table[bytes[b + 8] >> 4] * s);
          sum0 += wl0 * al[b] + wh0 * ah[b];
          sum1 += wl1 * al[b + 8] + wh1 * ah[b + 8];
        }
        acc += sum0 + sum1;
      }
      out[r * out_stride + (n - n_begin)] = acc + b_val;
    }
  }
}

#if QMOE_SHARED_ARCH_X86_64 && !defined(QMOE_SHARED_USE_MLAS)
bool DetectRuntimeAvx512CpuSupport() noexcept {
  unsigned cpuid1[4]{0, 0, 0, 0};
  unsigned cpuid0[4]{0, 0, 0, 0};
#if defined(_MSC_VER)
  __cpuid(reinterpret_cast<int*>(cpuid0), 0);
  if (cpuid0[0] < 7u) return false;
  __cpuid(reinterpret_cast<int*>(cpuid1), 1);
#else
  __cpuid(0, cpuid0[0], cpuid0[1], cpuid0[2], cpuid0[3]);
  if (cpuid0[0] < 7u) return false;
  __cpuid(1, cpuid1[0], cpuid1[1], cpuid1[2], cpuid1[3]);
#endif
  // Require OSXSAVE (bit 27), AVX (bit 28), FMA (bit 12), and F16C (bit 29).
  constexpr unsigned kLeaf1EcxMask = (1u << 27) | (1u << 28) | (1u << 12) | (1u << 29);
  if ((cpuid1[2] & kLeaf1EcxMask) != kLeaf1EcxMask) return false;

  uint64_t xcr0 = 0;
#if defined(_MSC_VER)
  xcr0 = _xgetbv(0);
#else
  uint32_t eax = 0, edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  xcr0 = (static_cast<uint64_t>(edx) << 32) | eax;
#endif
  // Require OS XSAVE state for XMM/YMM (0x6) and AVX-512 opmask + ZMM (0xE0).
  if ((xcr0 & 0xE6u) != 0xE6u) return false;

  unsigned cpuid7[4]{0, 0, 0, 0};
#if defined(_MSC_VER)
  __cpuidex(reinterpret_cast<int*>(cpuid7), 7, 0);
#else
  __cpuid_count(7, 0, cpuid7[0], cpuid7[1], cpuid7[2], cpuid7[3]);
#endif
  // Require AVX2 (bit 5), AVX512F (bit 16), AVX512DQ (bit 17), AVX512BW (bit 30), AVX512VL (bit 31).
  constexpr unsigned kLeaf7EbxMask = (1u << 5) | (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
  return (cpuid7[1] & kLeaf7EbxMask) == kLeaf7EbxMask;
}
#endif

}  // namespace

bool CompilerHasAvx512() noexcept {
#if QMOE_COMPILER_SUPPORTS_AVX512
  return true;
#else
  return false;
#endif
}

bool CpuHasAvx512() {
#if QMOE_COMPILER_SUPPORTS_AVX512 && QMOE_SHARED_ARCH_X86_64
  static const bool has = []() {
#if defined(QMOE_SHARED_USE_MLAS)
    return MlasPlatformAvx512Supported();
#else
    return DetectRuntimeAvx512CpuSupport();
#endif
  }();
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

bool IsAcceptedMxfp4PrepackContract(int64_t weights_prepacked, int64_t block_size) {
  return weights_prepacked == kRequiredWeightsPrepacked && block_size == kMxfp4BlockSize;
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

bool TensorExpertStore::Initialize(const void* base_ptr, size_t total_bytes, int64_t num_experts, size_t expert_stride,
                                   size_t min_required_alignment, uint64_t model_content_id) noexcept {
  size_t expected_total = 0;
  if (!ValidateExpertStorageRange(num_experts, expert_stride, 0, total_bytes, &expected_total)) {
    return false;
  }
  ExpertBlockLease lease{};
  if (!MakeExpertBlockLease(base_ptr, expected_total, min_required_alignment, &lease)) {
    return false;
  }
  // Compute a fast deterministic content fingerprint over representative slices if model_content_id == 0.
  if (model_content_id == 0 && expected_total > 0) {
    uint64_t hash = 14695981039346656037ULL;
    const size_t step = std::max<size_t>(1, expected_total / 64);
    for (size_t off = 0; off < expected_total; off += step) {
      hash ^= static_cast<uint64_t>(lease.data[off]);
      hash *= 1099511628211ULL;
    }
    model_content_id = hash ^ static_cast<uint64_t>(expected_total);
  }
  lease.content_id = model_content_id;
  store_lease_ = lease;
  num_experts_ = num_experts;
  expert_stride_ = expert_stride;
  min_alignment_ = min_required_alignment;
  model_content_id_ = model_content_id;
  return true;
}

bool TensorExpertStore::AcquireReady(int32_t expert_id, ExpertBlockLease* out_lease) const noexcept {
  if (out_lease == nullptr || store_lease_.data == nullptr || expert_id < 0 ||
      static_cast<int64_t>(expert_id) >= num_experts_) {
    return false;
  }
  size_t offset = 0;
  if (!CheckedMulSize(static_cast<size_t>(expert_id), expert_stride_, &offset)) {
    return false;
  }
  if (!MakeExpertBlockLease(store_lease_.data + offset, expert_stride_, min_alignment_, out_lease)) {
    return false;
  }
  out_lease->content_id = model_content_id_;
  return true;
}

bool TensorExpertStore::Prefetch(const int32_t* expert_ids, size_t count, uint64_t /*generation*/) const noexcept {
  if (store_lease_.data == nullptr || (count > 0 && expert_ids == nullptr)) return false;
  for (size_t i = 0; i < count; ++i) {
    if (expert_ids[i] < 0 || static_cast<int64_t>(expert_ids[i]) >= num_experts_) return false;
  }
  return true;
}

struct FileExpertStore::Impl {
  enum SlotState : uint8_t {
    kEmpty = 0,
    kLoading = 1,
    kReady = 2,
  };
  struct PrefetchReq {
    int32_t expert_id = -1;
    uint64_t generation = 0;
  };

#if !defined(_WIN32)
  int fd = -1;
#endif
  struct ReadBank {
    int fd = -1;
    uint64_t offset = 0;
    size_t bytes = 0;
    size_t destination = 0;
    ~ReadBank() {
#if !defined(_WIN32)
      if (fd >= 0) close(fd);
#endif
    }
  };
  std::vector<std::unique_ptr<ReadBank>> banks;
  std::string file_path;
  uint64_t file_offset = 0;
  int64_t num_experts = 0;
  size_t expert_stride = 0;
  size_t cache_slots = 0;
  size_t min_alignment = 256;
  uint64_t content_id = 0;

  uint8_t* raw_alloc = nullptr;
  uint8_t* aligned_base = nullptr;

  mutable std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable worker_cv;
  std::vector<int32_t> expert_to_slot;
  std::vector<int32_t> slot_expert;
  std::vector<SlotState> slot_state;
  std::vector<uint32_t> slot_pins;
  std::vector<uint64_t> slot_lru;
  std::vector<uint8_t> slot_prefetched;
  uint64_t lru_clock = 0;

  std::deque<PrefetchReq> prefetch_queue;
  uint64_t latest_generation = 0;
  bool worker_busy = false;
  bool stop_worker = false;
  std::thread worker_thread;

  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> prefetch_hits{0};
  std::atomic<uint64_t> ssd_misses{0};
  std::atomic<uint64_t> bytes_read{0};

  bool drop_page_cache = false;

  bool ReadExpertFromDisk(int32_t expert_id, uint8_t* dst) noexcept {
    const uint64_t offset = file_offset + static_cast<uint64_t>(expert_id) * expert_stride;
#if !defined(_WIN32)
    if (!banks.empty()) {
      for (const auto& bank : banks) {
        const uint64_t bank_offset = bank->offset + static_cast<uint64_t>(expert_id) * bank->bytes;
        size_t done = 0;
        while (done < bank->bytes) {
          const ssize_t r = pread(bank->fd, dst + bank->destination + done, bank->bytes - done,
                                  static_cast<off_t>(bank_offset + done));
          if (r < 0 && errno == EINTR) continue;
          if (r <= 0) return false;
          done += static_cast<size_t>(r);
        }
        if (drop_page_cache) {
          posix_fadvise(bank->fd, static_cast<off_t>(bank_offset), static_cast<off_t>(bank->bytes), POSIX_FADV_DONTNEED);
        }
        bytes_read.fetch_add(bank->bytes, std::memory_order_relaxed);
      }
      return true;
    }
    if (fd >= 0) {
      size_t done = 0;
      while (done < expert_stride) {
        const ssize_t r = pread(fd, dst + done, expert_stride - done, static_cast<off_t>(offset + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        done += static_cast<size_t>(r);
      }
      bytes_read.fetch_add(expert_stride, std::memory_order_relaxed);
      return true;
    }
#endif
    std::ifstream in(file_path, std::ios::binary);
    if (!in) return false;
    in.seekg(static_cast<std::streamoff>(offset));
    if (!in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(expert_stride))) return false;
    bytes_read.fetch_add(expert_stride, std::memory_order_relaxed);
    return true;
  }

  int SelectVictimSlotLocked() noexcept {
    // Prefer an empty slot first.
    for (size_t s = 0; s < cache_slots; ++s) {
      if (slot_state[s] == kEmpty && slot_pins[s] == 0) {
        return static_cast<int>(s);
      }
    }
    // Otherwise pick the unpinned Ready slot with minimum LRU timestamp.
    int best_slot = -1;
    uint64_t best_lru = std::numeric_limits<uint64_t>::max();
    for (size_t s = 0; s < cache_slots; ++s) {
      if (slot_state[s] == kReady && slot_pins[s] == 0 && slot_lru[s] < best_lru) {
        best_lru = slot_lru[s];
        best_slot = static_cast<int>(s);
      }
    }
    if (best_slot >= 0) {
      const int32_t old_exp = slot_expert[best_slot];
      if (old_exp >= 0 && static_cast<int64_t>(old_exp) < num_experts) {
        expert_to_slot[old_exp] = -1;
      }
      slot_expert[best_slot] = -1;
      slot_state[best_slot] = kEmpty;
      slot_prefetched[best_slot] = 0;
    }
    return best_slot;
  }

  void WorkerLoop() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop_worker) {
      if (prefetch_queue.empty()) {
        worker_busy = false;
        cv.notify_all();
        worker_cv.wait(lock, [this] { return stop_worker || !prefetch_queue.empty(); });
        continue;
      }
      worker_busy = true;
      const PrefetchReq req = prefetch_queue.front();
      prefetch_queue.pop_front();
      if (req.expert_id < 0 || static_cast<int64_t>(req.expert_id) >= num_experts) {
        continue;
      }
      if (expert_to_slot[req.expert_id] >= 0) {
        continue;
      }
      bool has_empty_slot = false;
      for (size_t s = 0; s < cache_slots; ++s) {
        if (slot_state[s] == kEmpty) {
          has_empty_slot = true;
          break;
        }
      }
      if (!has_empty_slot && req.generation < latest_generation) {
        continue;
      }
      const int victim = SelectVictimSlotLocked();
      if (victim < 0) {
        continue;
      }
      expert_to_slot[req.expert_id] = victim;
      slot_expert[victim] = req.expert_id;
      slot_state[victim] = kLoading;
      slot_prefetched[victim] = 1;
      uint8_t* dst = aligned_base + static_cast<size_t>(victim) * expert_stride;
      lock.unlock();
      const bool ok = ReadExpertFromDisk(req.expert_id, dst);
      lock.lock();
      if (ok && slot_expert[victim] == req.expert_id) {
        slot_state[victim] = kReady;
        slot_lru[victim] = ++lru_clock;
      } else {
        if (slot_expert[victim] == req.expert_id) {
          expert_to_slot[req.expert_id] = -1;
          slot_expert[victim] = -1;
          slot_state[victim] = kEmpty;
          slot_prefetched[victim] = 0;
        }
      }
      cv.notify_all();
    }
    worker_busy = false;
    cv.notify_all();
  }
};

FileExpertStore::~FileExpertStore() noexcept {
  Shutdown();
}

void FileExpertStore::Shutdown() noexcept {
  if (!impl_) return;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stop_worker = true;
    impl_->prefetch_queue.clear();
  }
  impl_->worker_cv.notify_all();
  if (impl_->worker_thread.joinable()) {
    impl_->worker_thread.join();
  }
#if !defined(_WIN32)
  if (impl_->fd >= 0) {
    close(impl_->fd);
    impl_->fd = -1;
  }
#endif
  if (impl_->raw_alloc) {
    std::free(impl_->raw_alloc);
    impl_->raw_alloc = nullptr;
    impl_->aligned_base = nullptr;
  }
  delete impl_;
  impl_ = nullptr;
  initialized_ = false;
}

bool FileExpertStore::Initialize(const char* file_path, uint64_t file_offset, int64_t num_experts,
                                 size_t expert_stride, size_t cache_slots, size_t min_required_alignment,
                                 uint64_t model_content_id) noexcept {
  return InitializeStorage(file_path, file_offset, num_experts, expert_stride, cache_slots,
                           min_required_alignment, model_content_id, nullptr, 0);
}

bool FileExpertStore::InitializeMapped(const MappedBank* banks, size_t bank_count, int64_t num_experts,
                                      size_t expert_stride, size_t cache_slots) noexcept {
  if (!banks || bank_count == 0) return false;
  return InitializeStorage(nullptr, 0, num_experts, expert_stride, cache_slots, kBankAlignment, 0, banks, bank_count);
}

bool FileExpertStore::InitializeStorage(const char* file_path, uint64_t file_offset, int64_t num_experts,
                                       size_t expert_stride, size_t cache_slots, size_t min_required_alignment,
                                       uint64_t model_content_id, const MappedBank* banks, size_t bank_count) noexcept {
  Shutdown();
  if ((!file_path && !banks) || num_experts <= 0 || expert_stride == 0 || cache_slots == 0 ||
      min_required_alignment == 0 || (min_required_alignment & (min_required_alignment - 1)) != 0) {
    return false;
  }
  size_t total_expert_bytes = 0;
  if (!ValidateExpertStorageRange(num_experts, expert_stride, file_offset, 0, &total_expert_bytes)) {
    return false;
  }
  if (file_offset > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(total_expert_bytes)) {
    return false;
  }
  const size_t eff_slots = std::min<size_t>(cache_slots, static_cast<size_t>(num_experts));
  size_t cache_bytes = 0;
  if (!CheckedMulSize(eff_slots, expert_stride, &cache_bytes)) {
    return false;
  }
  const size_t align = std::max<size_t>({min_required_alignment, kBankAlignment, size_t{4096}});
  if ((expert_stride % min_required_alignment) != 0) {
    return false;
  }
  size_t padded_cache_bytes = 0;
  if (!CheckedAlignUp(cache_bytes, size_t{4096}, &padded_cache_bytes)) {
    return false;
  }
  size_t alloc_bytes = 0;
  if (!CheckedAddSize(padded_cache_bytes, align, &alloc_bytes)) {
    return false;
  }

  std::vector<std::unique_ptr<Impl::ReadBank>> read_banks;
  try {
    for (size_t i = 0; i < bank_count; ++i) {
#if defined(__linux__)
      const auto& source = banks[i];
      size_t length = 0;
      const uintptr_t address = reinterpret_cast<uintptr_t>(source.data);
      if (!source.data || source.expert_bytes == 0 || source.block_offset > expert_stride ||
          source.expert_bytes > expert_stride - source.block_offset ||
          !CheckedMulSize(static_cast<size_t>(num_experts), source.expert_bytes, &length) ||
          address > std::numeric_limits<uintptr_t>::max() - length) return false;
      for (size_t j = 0; j < i; ++j) {
        if (source.block_offset < banks[j].block_offset + banks[j].expert_bytes &&
            banks[j].block_offset < source.block_offset + source.expert_bytes) return false;
      }
      std::ifstream maps("/proc/self/maps");
      std::string line;
      auto bank = std::make_unique<Impl::ReadBank>();
      while (std::getline(maps, line)) {
        unsigned long long lo = 0, hi = 0, offset = 0, inode = 0;
        unsigned int major_id = 0, minor_id = 0;
        char perms[5]{};
        int path_pos = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &lo, &hi, perms,
                        &offset, &major_id, &minor_id, &inode, &path_pos) != 7 ||
            address < lo || address + length > hi || perms[0] != 'r' || perms[1] == 'w' || inode == 0) continue;
        const std::string path = line.substr(static_cast<size_t>(path_pos));
        bank->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (bank->fd < 0) return false;
        struct stat st{};
        if (fstat(bank->fd, &st) != 0 || !S_ISREG(st.st_mode) ||
            static_cast<uint64_t>(st.st_ino) != inode || major(st.st_dev) != major_id || minor(st.st_dev) != minor_id ||
            offset > static_cast<uint64_t>(st.st_size) || address - lo > static_cast<uint64_t>(st.st_size) - offset ||
            length > static_cast<uint64_t>(st.st_size) - offset - (address - lo)) return false;
        bank->offset = offset + address - lo;
        bank->bytes = source.expert_bytes;
        bank->destination = source.block_offset;
        break;
      }
      if (bank->fd < 0) return false;
      read_banks.push_back(std::move(bank));
#else
      return false;
#endif
    }
  } catch (...) {
    return false;
  }

#if !defined(_WIN32)
  const int fd = file_path ? open(file_path, O_RDONLY | O_CLOEXEC) : -1;
  if (file_path) {
    if (fd < 0) return false;
    const off_t file_len = lseek(fd, 0, SEEK_END);
    if (file_len < 0 || static_cast<uint64_t>(file_len) < file_offset + static_cast<uint64_t>(total_expert_bytes)) {
      close(fd);
      return false;
    }
  }
#else
  if (!file_path) return false;
  std::ifstream check_in(file_path, std::ios::binary | std::ios::ate);
  if (!check_in) return false;
  const auto file_len = check_in.tellg();
  if (file_len < 0 || static_cast<uint64_t>(file_len) < file_offset + static_cast<uint64_t>(total_expert_bytes)) return false;
#endif

  uint8_t* raw = static_cast<uint8_t*>(std::malloc(alloc_bytes));
  if (!raw) {
#if !defined(_WIN32)
    close(fd);
#endif
    return false;
  }
  const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw);
  const uintptr_t aligned_addr = (raw_addr + align - 1) & ~(static_cast<uintptr_t>(align) - 1);

  Impl* impl = new (std::nothrow) Impl();
  if (!impl) {
    std::free(raw);
#if !defined(_WIN32)
    close(fd);
#endif
    return false;
  }
#if !defined(_WIN32)
  impl->fd = fd;
#endif
  try {
  impl->banks = std::move(read_banks);
  impl->file_path = file_path ? file_path : "mapped-banks";
  impl->file_offset = file_offset;
  impl->num_experts = num_experts;
  impl->expert_stride = expert_stride;
  impl->cache_slots = eff_slots;
  impl->min_alignment = min_required_alignment;
  impl->raw_alloc = raw;
  impl->aligned_base = reinterpret_cast<uint8_t*>(aligned_addr);
  if (const char* drop_env = std::getenv("ORT_QMOE_DROP_PAGE_CACHE");
      drop_env != nullptr && std::strcmp(drop_env, "1") == 0) {
    impl->drop_page_cache = true;
  }
#if defined(__linux__)
  if (!impl->banks.empty()) {
    const char* numa = std::getenv("ORT_QMOE_NUMA_INTERLEAVE");
    if (!numa || std::strcmp(numa, "0") != 0) {
      constexpr size_t kMaxNodes = 1024;
      std::array<unsigned long, kMaxNodes / (8 * sizeof(unsigned long))> nodes{};
      const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
      const uintptr_t first = (aligned_addr + page - 1) / page * page;
      const uintptr_t end = (aligned_addr + cache_bytes) / page * page;
      if (end > first && syscall(SYS_get_mempolicy, nullptr, nodes.data(), kMaxNodes, nullptr,
                                 MPOL_F_MEMS_ALLOWED) == 0) {
        // Anonymous cache pages obey VMA policy, unlike the MAP_SHARED source files.
        static_cast<void>(syscall(SYS_mbind, reinterpret_cast<void*>(first), end - first,
                                  MPOL_INTERLEAVE, nodes.data(), kMaxNodes, 0));
      }
    }
  }
#endif
    impl->expert_to_slot.assign(static_cast<size_t>(num_experts), -1);
    impl->slot_expert.assign(eff_slots, -1);
    impl->slot_state.assign(eff_slots, Impl::kEmpty);
    impl->slot_pins.assign(eff_slots, 0);
    impl->slot_lru.assign(eff_slots, 0);
    impl->slot_prefetched.assign(eff_slots, 0);
    if (model_content_id == 0) {
      uint64_t h = 14695981039346656037ULL;
      for (char c : impl->file_path) {
        h ^= static_cast<uint64_t>(static_cast<uint8_t>(c));
        h *= 1099511628211ULL;
      }
      model_content_id = h ^ static_cast<uint64_t>(total_expert_bytes) ^ file_offset;
    }
    impl->content_id = model_content_id;
    impl->worker_thread = std::thread([impl] { impl->WorkerLoop(); });
  } catch (...) {
#if !defined(_WIN32)
    if (impl->fd >= 0) close(impl->fd);
#endif
    std::free(raw);
    delete impl;
    return false;
  }

  impl_ = impl;
  initialized_ = true;
  num_experts_ = num_experts;
  expert_stride_ = expert_stride;
  cache_slots_ = eff_slots;
  model_content_id_ = model_content_id;
  return true;
}

bool FileExpertStore::AcquireReady(int32_t expert_id, ExpertBlockLease* out_lease) noexcept {
  if (!initialized_ || !impl_ || out_lease == nullptr || expert_id < 0 ||
      static_cast<int64_t>(expert_id) >= num_experts_) {
    return false;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  while (true) {
    const int slot = impl_->expert_to_slot[expert_id];
    if (slot >= 0) {
      if (impl_->slot_state[slot] == Impl::kReady) {
        if (impl_->slot_prefetched[slot]) {
          impl_->prefetch_hits.fetch_add(1, std::memory_order_relaxed);
          impl_->slot_prefetched[slot] = 0;
        } else {
          impl_->cache_hits.fetch_add(1, std::memory_order_relaxed);
        }
        ++impl_->slot_pins[slot];
        impl_->slot_lru[slot] = ++impl_->lru_clock;
        const uint8_t* ptr = impl_->aligned_base + static_cast<size_t>(slot) * expert_stride_;
        if (!MakeExpertBlockLease(ptr, expert_stride_, impl_->min_alignment, out_lease)) {
          --impl_->slot_pins[slot];
          return false;
        }
        out_lease->content_id = model_content_id_;
        return true;
      }
      // Currently loading by prefetch or another thread: wait for completion.
      impl_->cv.wait(lock);
      continue;
    }

    // Cold SSD miss: allocate a victim slot in the RAM cache.
    const int victim = impl_->SelectVictimSlotLocked();
    if (victim < 0) {
      // A prefetch may own the last available slot. Wait for I/O, but never wait on
      // caller-held leases: a group larger than the cache must be split by the caller.
      bool loading = false;
      for (size_t i = 0; i < impl_->cache_slots; ++i)
        loading |= impl_->slot_state[i] == Impl::kLoading && impl_->slot_pins[i] == 0;
      if (!loading) return false;
      impl_->cv.wait(lock);
      continue;
    }
    impl_->ssd_misses.fetch_add(1, std::memory_order_relaxed);
    impl_->expert_to_slot[expert_id] = victim;
    impl_->slot_expert[victim] = expert_id;
    impl_->slot_state[victim] = Impl::kLoading;
    impl_->slot_pins[victim] = 1;
    impl_->slot_prefetched[victim] = 0;
    uint8_t* dst = impl_->aligned_base + static_cast<size_t>(victim) * expert_stride_;
    lock.unlock();
    const bool ok = impl_->ReadExpertFromDisk(expert_id, dst);
    lock.lock();
    if (!ok) {
      impl_->expert_to_slot[expert_id] = -1;
      impl_->slot_expert[victim] = -1;
      impl_->slot_state[victim] = Impl::kEmpty;
      impl_->slot_pins[victim] = 0;
      impl_->cv.notify_all();
      return false;
    }
    impl_->slot_state[victim] = Impl::kReady;
    impl_->slot_lru[victim] = ++impl_->lru_clock;
    impl_->cv.notify_all();
    if (!MakeExpertBlockLease(dst, expert_stride_, impl_->min_alignment, out_lease)) {
      --impl_->slot_pins[victim];
      return false;
    }
    out_lease->content_id = model_content_id_;
    return true;
  }
}

void FileExpertStore::ReleaseLease(int32_t expert_id) noexcept {
  if (!initialized_ || !impl_ || expert_id < 0 || static_cast<int64_t>(expert_id) >= num_experts_) return;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const int slot = impl_->expert_to_slot[expert_id];
  if (slot >= 0 && impl_->slot_pins[slot] > 0) {
    --impl_->slot_pins[slot];
  }
}

bool FileExpertStore::Evict(int32_t expert_id) noexcept {
  if (!initialized_ || !impl_ || expert_id < 0 || expert_id >= num_experts_) return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const int slot = impl_->expert_to_slot[expert_id];
  if (slot < 0) return true;
  if (impl_->slot_pins[slot] != 0) return false;
  impl_->expert_to_slot[expert_id] = -1;
  impl_->slot_expert[slot] = -1;
  impl_->slot_state[slot] = Impl::kEmpty;
  impl_->slot_prefetched[slot] = 0;
  return true;
}

bool FileExpertStore::TryAcquireCached(int32_t expert_id, ExpertBlockLease* out_lease) noexcept {
  if (!initialized_ || !impl_ || out_lease == nullptr || expert_id < 0 ||
      static_cast<int64_t>(expert_id) >= num_experts_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const int slot = impl_->expert_to_slot[expert_id];
  if (slot < 0 || impl_->slot_state[slot] != Impl::kReady) {
    return false;
  }
  if (impl_->slot_prefetched[slot]) {
    impl_->prefetch_hits.fetch_add(1, std::memory_order_relaxed);
    impl_->slot_prefetched[slot] = 0;
  } else {
    impl_->cache_hits.fetch_add(1, std::memory_order_relaxed);
  }
  ++impl_->slot_pins[slot];
  impl_->slot_lru[slot] = ++impl_->lru_clock;
  const uint8_t* ptr = impl_->aligned_base + static_cast<size_t>(slot) * expert_stride_;
  if (!MakeExpertBlockLease(ptr, expert_stride_, impl_->min_alignment, out_lease)) {
    --impl_->slot_pins[slot];
    return false;
  }
  out_lease->content_id = model_content_id_;
  return true;
}

bool FileExpertStore::ReadExpertDirect(int32_t expert_id, uint8_t* dst) noexcept {
  if (!initialized_ || !impl_ || dst == nullptr || expert_id < 0 ||
      static_cast<int64_t>(expert_id) >= num_experts_) {
    return false;
  }
  return impl_->ReadExpertFromDisk(expert_id, dst);
}

bool FileExpertStore::TryAcquireGroupCached(const int32_t* expert_ids, size_t count,
                                            ExpertBlockLease* out_leases) noexcept {
  if (!initialized_ || !impl_ || (count > 0 && (!expert_ids || !out_leases))) return false;
  if (count == 0) return true;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (size_t i = 0; i < count; ++i) {
    const int32_t e = expert_ids[i];
    if (e < 0 || static_cast<int64_t>(e) >= num_experts_) return false;
    const int slot = impl_->expert_to_slot[e];
    if (slot < 0 || impl_->slot_state[slot] != Impl::kReady) {
      return false;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    const int32_t e = expert_ids[i];
    const int slot = impl_->expert_to_slot[e];
    if (impl_->slot_prefetched[slot]) {
      impl_->prefetch_hits.fetch_add(1, std::memory_order_relaxed);
      impl_->slot_prefetched[slot] = 0;
    } else {
      impl_->cache_hits.fetch_add(1, std::memory_order_relaxed);
    }
    ++impl_->slot_pins[slot];
    impl_->slot_lru[slot] = ++impl_->lru_clock;
    const uint8_t* ptr = impl_->aligned_base + static_cast<size_t>(slot) * expert_stride_;
    out_leases[i].data = ptr;
    out_leases[i].length = expert_stride_;
    out_leases[i].alignment = impl_->min_alignment;
    out_leases[i].content_id = model_content_id_;
  }
  return true;
}

bool FileExpertStore::AcquireGroupReady(const int32_t* expert_ids, size_t count,
                                        ExpertBlockLease* out_leases) noexcept {
  if (TryAcquireGroupCached(expert_ids, count, out_leases)) {
    return true;
  }
  if (!initialized_ || !impl_ || (count > 0 && (!expert_ids || !out_leases))) return false;
  for (size_t i = 0; i < count; ++i) {
    if (!AcquireReady(expert_ids[i], &out_leases[i])) {
      for (size_t j = 0; j < i; ++j) ReleaseLease(expert_ids[j]);
      return false;
    }
  }
  return true;
}

void FileExpertStore::ReleaseGroupLeases(const int32_t* expert_ids, size_t count) noexcept {
  if (!initialized_ || !impl_ || count == 0 || !expert_ids) return;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (size_t i = 0; i < count; ++i) {
    const int32_t e = expert_ids[i];
    if (e >= 0 && static_cast<int64_t>(e) < num_experts_) {
      const int slot = impl_->expert_to_slot[e];
      if (slot >= 0 && impl_->slot_pins[slot] > 0) {
        --impl_->slot_pins[slot];
      }
    }
  }
}

bool FileExpertStore::Prefetch(const int32_t* expert_ids, size_t count, uint64_t generation) noexcept {
  if (!initialized_ || !impl_ || (count > 0 && expert_ids == nullptr)) return false;
  for (size_t i = 0; i < count; ++i) {
    if (expert_ids[i] < 0 || static_cast<int64_t>(expert_ids[i]) >= num_experts_) return false;
  }
  bool queued_any = false;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (generation > impl_->latest_generation) {
      impl_->latest_generation = generation;
    }
    for (size_t i = 0; i < count; ++i) {
      const int32_t e = expert_ids[i];
      if (impl_->expert_to_slot[e] < 0) {
        impl_->prefetch_queue.push_back({e, generation});
        queued_any = true;
      }
    }
  }
  if (queued_any) {
    impl_->worker_cv.notify_one();
  }
  return true;
}

void FileExpertStore::WaitPrefetchIdle() noexcept {
  if (!initialized_ || !impl_) return;
  std::unique_lock<std::mutex> lock(impl_->mutex);
  impl_->cv.wait(lock, [this] { return impl_->prefetch_queue.empty() && !impl_->worker_busy; });
}

uint8_t* FileExpertStore::CacheBase() const noexcept {
  return impl_ ? impl_->aligned_base : nullptr;
}

void FileExpertStore::SnapshotExpertToSlot(int32_t* out_slots, int32_t max_experts) const noexcept {
  if (!out_slots || max_experts <= 0) return;
  if (!initialized_ || !impl_) {
    for (int32_t i = 0; i < max_experts; ++i) out_slots[i] = -1;
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const int32_t n = std::min<int32_t>(max_experts, static_cast<int32_t>(impl_->expert_to_slot.size()));
  for (int32_t e = 0; e < n; ++e) {
    const int32_t s = impl_->expert_to_slot[static_cast<size_t>(e)];
    out_slots[e] = (s >= 0 && impl_->slot_state[static_cast<size_t>(s)] == Impl::kReady) ? s : -1;
  }
  for (int32_t e = n; e < max_experts; ++e) {
    out_slots[e] = -1;
  }
}

uint64_t FileExpertStore::CacheHits() const noexcept {
  return impl_ ? impl_->cache_hits.load(std::memory_order_relaxed) : 0;
}
uint64_t FileExpertStore::PrefetchHits() const noexcept {
  return impl_ ? impl_->prefetch_hits.load(std::memory_order_relaxed) : 0;
}
uint64_t FileExpertStore::SsdMisses() const noexcept {
  return impl_ ? impl_->ssd_misses.load(std::memory_order_relaxed) : 0;
}
uint64_t FileExpertStore::BytesReadFromSsd() const noexcept {
  return impl_ ? impl_->bytes_read.load(std::memory_order_relaxed) : 0;
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

bool MakeMxfp4ExpertBlockLayout(int64_t hidden, int64_t inter, Nvfp4ExpertBlockLayout* layout) {
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
  if (!CheckedMulSize(n1, h / 32, &bytes) || !add_bank(off, bytes, &l.fc1_scales, &off)) return false;
  if (!add_bank(off, sizeof(float), &l.fc1_global_scale, &off)) return false;
  if (!CheckedMulSize(h, i / 2, &bytes) || !add_bank(off, bytes, &l.fc2_weights, &off)) return false;
  if (!CheckedMulSize(h, i / 32, &bytes) || !add_bank(off, bytes, &l.fc2_scales, &off)) return false;
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
float DecodeFloat8E8M0(uint8_t code) { return GetE8M0LookupTable()[code]; }

void HalfBitsToFloatRow(const uint16_t* src, float* dst, int64_t count, bool force_scalar) {
#if QMOE_COMPILER_SUPPORTS_AVX512
  if (!force_scalar && CpuHasAvx512()) {
    detail::Avx512HalfBitsToFloatRow(src, dst, count);
    return;
  }
#endif
  (void)force_scalar;
  for (int64_t i = 0; i < count; ++i) dst[i] = HalfToFloat(src[i]);
}

void FloatToHalfBitsRow(const float* src, uint16_t* dst, int64_t count, bool force_scalar) {
#if QMOE_COMPILER_SUPPORTS_AVX512
  if (!force_scalar && CpuHasAvx512()) {
    detail::Avx512FloatToHalfBitsRow(src, dst, count);
    return;
  }
#endif
  (void)force_scalar;
  for (int64_t i = 0; i < count; ++i) dst[i] = FloatToHalf(src[i]);
}

void BFloat16BitsToFloatRow(const uint16_t* src, float* dst, int64_t count, bool force_scalar) {
#if QMOE_COMPILER_SUPPORTS_AVX512
  if (!force_scalar && CpuHasAvx512()) {
    detail::Avx512BFloat16BitsToFloatRow(src, dst, count);
    return;
  }
#endif
  (void)force_scalar;
  for (int64_t i = 0; i < count; ++i) dst[i] = BFloat16ToFloat(src[i]);
}

void FloatToBFloat16BitsRow(const float* src, uint16_t* dst, int64_t count, bool force_scalar) {
#if QMOE_COMPILER_SUPPORTS_AVX512
  if (!force_scalar && CpuHasAvx512()) {
    detail::Avx512FloatToBFloat16BitsRow(src, dst, count);
    return;
  }
#endif
  (void)force_scalar;
  for (int64_t i = 0; i < count; ++i) dst[i] = FloatToBFloat16(src[i]);
}

void GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total,
               int64_t k, const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin,
               int64_t n_end, bool force_scalar) {
#if QMOE_COMPILER_SUPPORTS_AVX512
  if (!force_scalar && CpuHasAvx512()) {
    detail::Avx512GemmRange(weights, scales, bias, n_total, k, a, m, out, out_stride, n_begin, n_end);
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
#if QMOE_COMPILER_SUPPORTS_AVX512
  if (!force_scalar && CpuHasAvx512()) {
    detail::Avx512Nvfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                                 a_reordered, m, out, out_stride, n_begin, n_end, round_scale_bf16);
    return;
  }
#endif
  (void)force_scalar;
  ScalarNvfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                       a_reordered, m, out, out_stride, n_begin, n_end, round_scale_bf16);
}

void Mxfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                    const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                    const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                    int64_t n_begin, int64_t n_end, bool force_scalar, bool round_scale_bf16) {
#if QMOE_COMPILER_SUPPORTS_AVX512
  if (!force_scalar && CpuHasAvx512()) {
    detail::Avx512Mxfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
                                 a_reordered, m, out, out_stride, n_begin, n_end, round_scale_bf16);
    return;
  }
#endif
  (void)force_scalar;
  ScalarMxfp4GemmRange(weights, block_scales, global_scale, bias, bias_is_bf16, n_total, k,
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
    if (view.is_mxfp4) {
      Mxfp4GemmRange(view.fc1_weights, view.fc1_scales, view.fc1_global_scale,
                     view.fc1_bias, view.bias_is_bf16, n_total, hidden, x_reordered, m,
                     tmp, kChunk, nb, ne, force_scalar, round_scale_bf16);
    } else {
      Nvfp4GemmRange(view.fc1_weights, view.fc1_scales, view.fc1_global_scale,
                     view.fc1_bias, view.bias_is_bf16, n_total, hidden, x_reordered, m,
                     tmp, kChunk, nb, ne, force_scalar, round_scale_bf16);
    }
    for (int64_t n = nb; n < ne; n += 2) {
      const int64_t j = n / 2;
      const int pos = Nvfp4ReorderedPosition(static_cast<int>(j & 31));
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
        const float rounded_act = bf16_boundaries   ? RoundBf16Fast(act)
                                  : fp16_boundaries ? BoundaryFloat16::Round(act)
                                                    : act;
        inter_reordered[r * inter + (j & ~int64_t{31}) + pos] = rounded_act;
      }
    }
  }
}

void Nvfp4Fc2Range(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                   const float* inter_reordered, int64_t m, float* out,
                   int64_t n_begin, int64_t n_end, bool force_scalar,
                   bool apply_bias, bool round_scale_bf16) {
  if (view.is_mxfp4) {
    Mxfp4GemmRange(view.fc2_weights, view.fc2_scales, view.fc2_global_scale,
                   apply_bias ? view.fc2_bias : nullptr, view.bias_is_bf16,
                   hidden, inter, inter_reordered, m, out + n_begin, hidden,
                   n_begin, n_end, force_scalar, round_scale_bf16);
  } else {
    Nvfp4GemmRange(view.fc2_weights, view.fc2_scales, view.fc2_global_scale,
                   apply_bias ? view.fc2_bias : nullptr, view.bias_is_bf16,
                   hidden, inter, inter_reordered, m, out + n_begin, hidden,
                   n_begin, n_end, force_scalar, round_scale_bf16);
  }
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

int qmoe_mxfp4_expert_contribution_bf16(const uint8_t* fc1_w, const uint8_t* fc1_s, float fc1_g,
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
  view.is_mxfp4 = true;
  SwiGluParams params{alpha, beta, limit};
  return Nvfp4ExpertForward(hidden, inter, view, x, m, params, out, force_scalar != 0,
                            /*apply_fc2_bias=*/false, /*bf16_boundaries=*/true, /*fp16_boundaries=*/false)
             ? 1
             : 0;
}

int64_t qmoe_mxfp4_block_bytes(int64_t hidden, int64_t inter) {
  Nvfp4ExpertBlockLayout layout;
  return MakeMxfp4ExpertBlockLayout(hidden, inter, &layout) ? static_cast<int64_t>(layout.stride) : -1;
}
}

#if !defined(QMOE_SHARED_SEPARATE_AVX512_TU) && QMOE_COMPILER_SUPPORTS_AVX512
#include "contrib_ops/cpu/moe/qmoe_shared_layout_avx512.cc"
#endif
