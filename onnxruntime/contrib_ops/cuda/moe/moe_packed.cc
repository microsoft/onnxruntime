// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_packed.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_M_AMD64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "core/common/parse_string.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
bool PackedTensorBytesFit(int64_t rows, int64_t columns, size_t element_bytes) {
  size_t row_bytes = 0;
  size_t bytes = 0;
  return rows >= 0 && columns >= 0 &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(columns), element_bytes, &row_bytes) &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(rows), row_bytes, &bytes) &&
         bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
}

size_t AlignUp256(size_t v) {
  return (v + 255u) & ~size_t{255u};
}

inline float RoundBf16Scalar(float x) {
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  if ((bits & 0x7FFFu) < 0x7F800000u) {
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    bits &= 0xFFFF0000u;
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline uint16_t FloatToBf16Bits(float x) {
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  if ((bits & 0x7FFFu) < 0x7F800000u) {
    bits += 0x7FFFu + ((bits >> 16) & 1u);
  }
  return static_cast<uint16_t>(bits >> 16);
}

#if defined(_M_AMD64) || defined(__x86_64__)
inline float HorizontalSumAvx2(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

struct SignXorLutAvx2 {
  alignas(32) uint32_t table[256][8];
  constexpr SignXorLutAvx2() : table{} {
    for (int s = 0; s < 256; ++s) {
      for (int j = 0; j < 8; ++j) {
        table[s][j] = (s & (1 << j)) ? 0x80000000u : 0u;
      }
    }
  }
};
static constexpr SignXorLutAvx2 kSignXorLutAvx2{};

inline __m256 SignXorAvx2(uint8_t signs) {
  return _mm256_load_ps(reinterpret_cast<const float*>(kSignXorLutAvx2.table[signs]));
}
#endif

// Fused AVX2 dot product of one (gate_row, up_row) pair against R token activation vectors x_rows[0..R-1].
void GgufDualDotRowCpu(
    const uint8_t* gate_row,
    const uint8_t* up_row,
    int gate_type,
    int up_type,
    int K,
    const float* const* x_rows,
    int R,
    float* gate_dots,
    float* up_dots) {
  using namespace onnxruntime::contrib::gguf_quant;
#if defined(_M_AMD64) || defined(__x86_64__)
  if (gate_type == up_type && (gate_type == kGgmlTypeIQ2_S || gate_type == kGgmlTypeIQ2_XXS || gate_type == kGgmlTypeIQ2_XS)) {
    __m256 acc_g[kMoEPackedMaxDecodeRows];
    __m256 acc_u[kMoEPackedMaxDecodeRows];
    for (int r = 0; r < R; ++r) {
      acc_g[r] = _mm256_setzero_ps();
      acc_u[r] = _mm256_setzero_ps();
    }
    const int nb = K / 256;
    if (gate_type == kGgmlTypeIQ2_S) {
      for (int i = 0; i < nb; ++i) {
        const uint8_t* blk_g = gate_row + i * 82;
        const uint8_t* blk_u = up_row + i * 82;
        const float d_g = Fp16BitsToFloatCpu(ReadU16Le(blk_g)) * 0.25f;
        const float d_u = Fp16BitsToFloatCpu(ReadU16Le(blk_u)) * 0.25f;
        const uint8_t* qs_g = blk_g + 2;
        const uint8_t* signs_g = qs_g + 32;
        const uint8_t* qh_g = blk_g + 66;
        const uint8_t* scales_g = blk_g + 74;
        const uint8_t* qs_u = blk_u + 2;
        const uint8_t* signs_u = qs_u + 32;
        const uint8_t* qh_u = blk_u + 66;
        const uint8_t* scales_u = blk_u + 74;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float db0_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] & 0xF));
          const float db1_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] >> 4));
          const float db0_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] & 0xF));
          const float db1_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] >> 4));
          const uint8_t qh_byte_g = qh_g[ib32];
          const uint8_t qh_byte_u = qh_u[ib32];
          for (int l = 0; l < 4; ++l) {
            const __m256 v_dl_g = _mm256_set1_ps((l < 2) ? db0_g : db1_g);
            const __m256 v_dl_u = _mm256_set1_ps((l < 2) ? db0_u : db1_u);
            const int idx_g = qs_g[l] | ((static_cast<int>(qh_byte_g) << (8 - 2 * l)) & 0x300);
            const int idx_u = qs_u[l] | ((static_cast<int>(qh_byte_u) << (8 - 2 * l)) & 0x300);
            const __m128i g8_g = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2s_grid + idx_g));
            const __m128i g8_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2s_grid + idx_u));
            const __m256 vf_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_g));
            const __m256 vf_u = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_u));
            const __m256 wg = _mm256_mul_ps(v_dl_g, _mm256_xor_ps(vf_g, SignXorAvx2(signs_g[l])));
            const __m256 wu = _mm256_mul_ps(v_dl_u, _mm256_xor_ps(vf_u, SignXorAvx2(signs_u[l])));
            const int k_off = i * 256 + ib32 * 32 + l * 8;
            for (int r = 0; r < R; ++r) {
              const __m256 xv = _mm256_loadu_ps(x_rows[r] + k_off);
              acc_g[r] = _mm256_fmadd_ps(wg, xv, acc_g[r]);
              acc_u[r] = _mm256_fmadd_ps(wu, xv, acc_u[r]);
            }
          }
          qs_g += 4;
          signs_g += 4;
          qs_u += 4;
          signs_u += 4;
        }
      }
    } else if (gate_type == kGgmlTypeIQ2_XXS) {
      for (int i = 0; i < nb; ++i) {
        const uint8_t* blk_g = gate_row + i * 66;
        const uint8_t* blk_u = up_row + i * 66;
        const float d_g = Fp16BitsToFloatCpu(ReadU16Le(blk_g)) * 0.25f;
        const float d_u = Fp16BitsToFloatCpu(ReadU16Le(blk_u)) * 0.25f;
        const uint8_t* qs_g = blk_g + 2;
        const uint8_t* qs_u = blk_u + 2;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const uint8_t* aux8_g = qs_g + 8 * ib32;
          const uint8_t* aux8_u = qs_u + 8 * ib32;
          const uint32_t aux32_g = ReadU32Le(aux8_g + 4);
          const uint32_t aux32_u = ReadU32Le(aux8_u + 4);
          const __m256 v_db_g = _mm256_set1_ps(d_g * (0.5f + static_cast<float>(aux32_g >> 28)));
          const __m256 v_db_u = _mm256_set1_ps(d_u * (0.5f + static_cast<float>(aux32_u >> 28)));
          for (int l = 0; l < 4; ++l) {
            const __m128i g8_g = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xxs_grid + aux8_g[l]));
            const __m128i g8_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xxs_grid + aux8_u[l]));
            const __m256 vf_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_g));
            const __m256 vf_u = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_u));
            const uint8_t sg = ksigns_iq2xs[(aux32_g >> (7 * l)) & 127];
            const uint8_t su = ksigns_iq2xs[(aux32_u >> (7 * l)) & 127];
            const __m256 wg = _mm256_mul_ps(v_db_g, _mm256_xor_ps(vf_g, SignXorAvx2(sg)));
            const __m256 wu = _mm256_mul_ps(v_db_u, _mm256_xor_ps(vf_u, SignXorAvx2(su)));
            const int k_off = i * 256 + ib32 * 32 + l * 8;
            for (int r = 0; r < R; ++r) {
              const __m256 xv = _mm256_loadu_ps(x_rows[r] + k_off);
              acc_g[r] = _mm256_fmadd_ps(wg, xv, acc_g[r]);
              acc_u[r] = _mm256_fmadd_ps(wu, xv, acc_u[r]);
            }
          }
        }
      }
    } else {
      for (int i = 0; i < nb; ++i) {
        const uint8_t* blk_g = gate_row + i * 74;
        const uint8_t* blk_u = up_row + i * 74;
        const float d_g = Fp16BitsToFloatCpu(ReadU16Le(blk_g)) * 0.25f;
        const float d_u = Fp16BitsToFloatCpu(ReadU16Le(blk_u)) * 0.25f;
        const uint8_t* qs_g = blk_g + 2;
        const uint8_t* scales_g = blk_g + 66;
        const uint8_t* qs_u = blk_u + 2;
        const uint8_t* scales_u = blk_u + 66;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          const float db0_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] & 0xF));
          const float db1_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] >> 4));
          const float db0_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] & 0xF));
          const float db1_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] >> 4));
          for (int l = 0; l < 4; ++l) {
            const uint16_t qg = ReadU16Le(qs_g + (4 * ib32 + l) * 2);
            const uint16_t qu = ReadU16Le(qs_u + (4 * ib32 + l) * 2);
            const __m128i g8_g = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xs_grid + (qg & 511)));
            const __m128i g8_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xs_grid + (qu & 511)));
            const __m256 vf_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_g));
            const __m256 vf_u = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_u));
            const __m256 v_dl_g = _mm256_set1_ps((l < 2) ? db0_g : db1_g);
            const __m256 v_dl_u = _mm256_set1_ps((l < 2) ? db0_u : db1_u);
            const __m256 wg = _mm256_mul_ps(v_dl_g, _mm256_xor_ps(vf_g, SignXorAvx2(ksigns_iq2xs[qg >> 9])));
            const __m256 wu = _mm256_mul_ps(v_dl_u, _mm256_xor_ps(vf_u, SignXorAvx2(ksigns_iq2xs[qu >> 9])));
            const int k_off = i * 256 + ib32 * 32 + l * 8;
            for (int r = 0; r < R; ++r) {
              const __m256 xv = _mm256_loadu_ps(x_rows[r] + k_off);
              acc_g[r] = _mm256_fmadd_ps(wg, xv, acc_g[r]);
              acc_u[r] = _mm256_fmadd_ps(wu, xv, acc_u[r]);
            }
          }
        }
      }
    }
    for (int r = 0; r < R; ++r) {
      gate_dots[r] = HorizontalSumAvx2(acc_g[r]);
      up_dots[r] = HorizontalSumAvx2(acc_u[r]);
    }
    return;
  }
#endif
  alignas(32) float buf_g[kMoEPackedMaxHiddenSize];
  alignas(32) float buf_u[kMoEPackedMaxHiddenSize];
  DequantizeGgufRowCpu(gate_row, gate_type, K, buf_g);
  DequantizeGgufRowCpu(up_row, up_type, K, buf_u);
  for (int r = 0; r < R; ++r) {
    float sg = 0.0f;
    float su = 0.0f;
    const float* xr = x_rows[r];
    for (int k = 0; k < K; ++k) {
      sg += buf_g[k] * xr[k];
      su += buf_u[k] * xr[k];
    }
    gate_dots[r] = sg;
    up_dots[r] = su;
  }
}

// AVX2 dot product of one down_row against R intermediate vectors inter_rows[0..R-1].
void GgufDotRowCpu(
    const uint8_t* down_row,
    int down_type,
    int K,
    const float* const* inter_rows,
    int R,
    float* down_dots) {
  using namespace onnxruntime::contrib::gguf_quant;
#if defined(_M_AMD64) || defined(__x86_64__)
  if (down_type == kGgmlTypeIQ4_NL) {
    __m256 acc[kMoEPackedMaxDecodeRows];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const int nb = K / 32;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 18;
      const __m256 vd = _mm256_set1_ps(Fp16BitsToFloatCpu(ReadU16Le(blk)));
      const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
      const __m128i lo_nib = _mm_and_si128(raw16, mask0f);
      const __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f);
      const __m128i v_lo = _mm_shuffle_epi8(lut, lo_nib);
      const __m128i v_hi = _mm_shuffle_epi8(lut, hi_nib);
      const __m256 w0 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo)));
      const __m256 w1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8))));
      const __m256 w2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi)));
      const __m256 w3 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8))));
      const int k_off = i * 32;
      for (int r = 0; r < R; ++r) {
        const float* xr = inter_rows[r] + k_off;
        acc[r] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xr + 0), acc[r]);
        acc[r] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr + 8), acc[r]);
        acc[r] = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xr + 16), acc[r]);
        acc[r] = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xr + 24), acc[r]);
      }
    }
    for (int r = 0; r < R; ++r) down_dots[r] = HorizontalSumAvx2(acc[r]);
    return;
  } else if (down_type == kGgmlTypeQ2_0) {
    __m256 acc[kMoEPackedMaxDecodeRows];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask3 = _mm256_set1_epi32(3);
    const __m256i ones = _mm256_set1_epi32(1);
    const int nb = K / 64;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 18;
      const __m256 vd = _mm256_set1_ps(Fp16BitsToFloatCpu(ReadU16Le(blk)));
      for (int w = 0; w < 8; ++w) {
        const uint16_t q16 = ReadU16Le(blk + 2 + w * 2);
        const __m256i qv = _mm256_sub_epi32(
            _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(q16), shifts), mask3), ones);
        const __m256 w_vec = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(qv));
        const int k_off = i * 64 + w * 8;
        for (int r = 0; r < R; ++r) {
          acc[r] = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(inter_rows[r] + k_off), acc[r]);
        }
      }
    }
    for (int r = 0; r < R; ++r) down_dots[r] = HorizontalSumAvx2(acc[r]);
    return;
  } else if (down_type == kGgmlTypeQ5_0) {
    __m256 acc[kMoEPackedMaxDecodeRows];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const int nb = K / 32;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 22;
      const __m256 vd = _mm256_set1_ps(Fp16BitsToFloatCpu(ReadU16Le(blk)));
      const uint32_t qh = ReadU32Le(blk + 2);
      const uint8_t* qs = blk + 6;
      alignas(32) float w32[32];
      for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
        w32[j] = static_cast<float>(((qs[j] & 0x0F) | xh_0) - 16);
        w32[j + 16] = static_cast<float>(((qs[j] >> 4) | xh_1) - 16);
      }
      const int k_off = i * 32;
      for (int s = 0; s < 32; s += 8) {
        const __m256 w_vec = _mm256_mul_ps(vd, _mm256_load_ps(w32 + s));
        for (int r = 0; r < R; ++r) {
          acc[r] = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(inter_rows[r] + k_off + s), acc[r]);
        }
      }
      (void)mask0f;
    }
    for (int r = 0; r < R; ++r) down_dots[r] = HorizontalSumAvx2(acc[r]);
    return;
  }
#endif
  alignas(32) float buf_d[kMoEPackedMaxHiddenSize];
  DequantizeGgufRowCpu(down_row, down_type, K, buf_d);
  for (int r = 0; r < R; ++r) {
    float sd = 0.0f;
    const float* xr = inter_rows[r];
    for (int k = 0; k < K; ++k) {
      sd += buf_d[k] * xr[k];
    }
    down_dots[r] = sd;
  }
}

// Lightweight spin-then-park parallel pool shared across MoEPacked offload nodes.
class MoEPackedParallelPool {
 public:
  static MoEPackedParallelPool& Instance() {
    static MoEPackedParallelPool pool;
    return pool;
  }

  int NumThreads() const noexcept { return num_threads_; }

  template <typename Fn>
  void ParallelFor(int num_tasks, const Fn& fn) {
    if (num_tasks <= 0) return;
    if (num_tasks == 1 || num_threads_ <= 1) {
      for (int i = 0; i < num_tasks; ++i) fn(i);
      return;
    }
    std::unique_lock<std::mutex> call_lock(caller_mu_);
    task_fn_ = &fn;
    task_invoker_ = [](const void* ctx, int idx) { (*static_cast<const Fn*>(ctx))(idx); };
    num_tasks_.store(num_tasks, std::memory_order_relaxed);
    next_task_.store(0, std::memory_order_relaxed);
    remaining_.store(num_tasks, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_release);
    if (sleeping_.load(std::memory_order_acquire) > 0) {
      std::lock_guard<std::mutex> lk(wake_mu_);
      wake_cv_.notify_all();
    }
    DrainTasks();
    while (remaining_.load(std::memory_order_acquire) > 0) {
#if defined(_M_AMD64) || defined(__x86_64__)
      _mm_pause();
#endif
    }
  }

 private:
  MoEPackedParallelPool() {
    const unsigned int hw = std::thread::hardware_concurrency();
    num_threads_ = static_cast<int>(std::clamp<unsigned int>(hw > 0 ? (hw * 3) / 4 : 16, 4u, 24u));
    workers_.reserve(static_cast<size_t>(num_threads_ - 1));
    for (int i = 1; i < num_threads_; ++i) {
      workers_.emplace_back([this]() { WorkerLoop(); });
    }
  }

  ~MoEPackedParallelPool() {
    stop_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);
    wake_cv_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) t.detach();
    }
  }

  void DrainTasks() {
    const int total = num_tasks_.load(std::memory_order_relaxed);
    for (;;) {
      const int t = next_task_.fetch_add(1, std::memory_order_relaxed);
      if (t >= total) break;
      task_invoker_(task_fn_, t);
      remaining_.fetch_sub(1, std::memory_order_release);
    }
  }

  void WorkerLoop() {
    uint64_t seen = epoch_.load(std::memory_order_acquire);
    for (;;) {
      auto spin_start = std::chrono::steady_clock::now();
      while (epoch_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() - spin_start > std::chrono::milliseconds(15)) {
          sleeping_.fetch_add(1, std::memory_order_acq_rel);
          {
            std::unique_lock<std::mutex> lk(wake_mu_);
            wake_cv_.wait(lk, [&]() {
              return epoch_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_relaxed);
            });
          }
          sleeping_.fetch_sub(1, std::memory_order_acq_rel);
          break;
        }
#if defined(_M_AMD64) || defined(__x86_64__)
        _mm_pause();
#endif
      }
      if (stop_.load(std::memory_order_relaxed)) return;
      seen = epoch_.load(std::memory_order_acquire);
      DrainTasks();
    }
  }

  int num_threads_{1};
  std::vector<std::thread> workers_;
  std::mutex caller_mu_;
  std::mutex wake_mu_;
  std::condition_variable wake_cv_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> epoch_{0};
  std::atomic<int> sleeping_{0};
  std::atomic<int> num_tasks_{0};
  std::atomic<int> next_task_{0};
  std::atomic<int> remaining_{0};
  const void* task_fn_{nullptr};
  void (*task_invoker_)(const void*, int){nullptr};
};

// Shared prefill double-buffered wave staging across MoEPacked layers on a device.
struct MoEPackedSharedPrefillWaveStaging {
  static constexpr int kWaveExperts = 32;
  std::mutex mu;
  size_t bank_bytes{0};
  uint8_t* pinned_bank[2]{nullptr, nullptr};
  uint8_t* gpu_bank[2]{nullptr, nullptr};
  cudaStream_t copy_stream{nullptr};
  cudaEvent_t h2d_done[2]{nullptr, nullptr};
  cudaEvent_t compute_done[2]{nullptr, nullptr};
  bool bank_in_flight[2]{false, false};
  int32_t* pinned_topk_idx{nullptr};
  float* pinned_topk_scales{nullptr};
  int32_t* pinned_res_sorted_routes{nullptr};
  MoEPackedTile* pinned_res_tiles{nullptr};
  int32_t* pinned_res_num_tiles{nullptr};
  int32_t* pinned_sorted_routes[2]{nullptr, nullptr};
  MoEPackedTile* pinned_tiles[2]{nullptr, nullptr};
  int32_t* pinned_num_tiles[2]{nullptr, nullptr};
  size_t pinned_route_cap{0};

  ~MoEPackedSharedPrefillWaveStaging() = default;

  void ReleaseAll() {
    std::lock_guard<std::mutex> lk(mu);
    for (int b = 0; b < 2; ++b) {
      if (h2d_done[b]) {
        cudaEventDestroy(h2d_done[b]);
        h2d_done[b] = nullptr;
      }
      if (compute_done[b]) {
        cudaEventDestroy(compute_done[b]);
        compute_done[b] = nullptr;
      }
      if (pinned_bank[b]) {
        cudaFreeHost(pinned_bank[b]);
        pinned_bank[b] = nullptr;
      }
      if (gpu_bank[b]) {
        cudaFree(gpu_bank[b]);
        gpu_bank[b] = nullptr;
      }
      if (pinned_sorted_routes[b]) {
        cudaFreeHost(pinned_sorted_routes[b]);
        pinned_sorted_routes[b] = nullptr;
      }
      if (pinned_tiles[b]) {
        cudaFreeHost(pinned_tiles[b]);
        pinned_tiles[b] = nullptr;
      }
      if (pinned_num_tiles[b]) {
        cudaFreeHost(pinned_num_tiles[b]);
        pinned_num_tiles[b] = nullptr;
      }
      bank_in_flight[b] = false;
    }
    if (copy_stream) {
      cudaStreamDestroy(copy_stream);
      copy_stream = nullptr;
    }
    if (pinned_topk_idx) {
      cudaFreeHost(pinned_topk_idx);
      pinned_topk_idx = nullptr;
    }
    if (pinned_topk_scales) {
      cudaFreeHost(pinned_topk_scales);
      pinned_topk_scales = nullptr;
    }
    if (pinned_res_sorted_routes) {
      cudaFreeHost(pinned_res_sorted_routes);
      pinned_res_sorted_routes = nullptr;
    }
    if (pinned_res_tiles) {
      cudaFreeHost(pinned_res_tiles);
      pinned_res_tiles = nullptr;
    }
    if (pinned_res_num_tiles) {
      cudaFreeHost(pinned_res_num_tiles);
      pinned_res_num_tiles = nullptr;
    }
    bank_bytes = 0;
    pinned_route_cap = 0;
  }

  Status EnsureCapacity(size_t required_bank_bytes, size_t max_routes) {
    if (required_bank_bytes > bank_bytes) {
      for (int b = 0; b < 2; ++b) {
        if (bank_in_flight[b] && compute_done[b]) {
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(compute_done[b])));
          bank_in_flight[b] = false;
        }
        if (pinned_bank[b]) {
          cudaFreeHost(pinned_bank[b]);
          pinned_bank[b] = nullptr;
        }
        if (gpu_bank[b]) {
          cudaFree(gpu_bank[b]);
          gpu_bank[b] = nullptr;
        }
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_bank[b]), required_bank_bytes, cudaHostAllocDefault)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&gpu_bank[b]), required_bank_bytes)));
      }
      bank_bytes = required_bank_bytes;
    }
    if (!copy_stream) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking)));
      for (int b = 0; b < 2; ++b) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&h2d_done[b], cudaEventDisableTiming)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&compute_done[b], cudaEventDisableTiming)));
      }
    }
    if (max_routes > pinned_route_cap) {
      if (pinned_topk_idx) cudaFreeHost(pinned_topk_idx);
      if (pinned_topk_scales) cudaFreeHost(pinned_topk_scales);
      if (pinned_res_sorted_routes) cudaFreeHost(pinned_res_sorted_routes);
      if (pinned_res_tiles) cudaFreeHost(pinned_res_tiles);
      if (pinned_res_num_tiles) cudaFreeHost(pinned_res_num_tiles);
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_topk_idx), max_routes * sizeof(int32_t), cudaHostAllocDefault)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_topk_scales), max_routes * sizeof(float), cudaHostAllocDefault)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_res_sorted_routes), max_routes * sizeof(int32_t), cudaHostAllocDefault)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_res_tiles), (max_routes / 8 + 512) * sizeof(MoEPackedTile), cudaHostAllocDefault)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_res_num_tiles), sizeof(int32_t), cudaHostAllocDefault)));
      for (int b = 0; b < 2; ++b) {
        if (pinned_sorted_routes[b]) cudaFreeHost(pinned_sorted_routes[b]);
        if (pinned_tiles[b]) cudaFreeHost(pinned_tiles[b]);
        if (pinned_num_tiles[b]) cudaFreeHost(pinned_num_tiles[b]);
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_sorted_routes[b]), max_routes * sizeof(int32_t), cudaHostAllocDefault)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_tiles[b]), (max_routes / 8 + 512) * sizeof(MoEPackedTile), cudaHostAllocDefault)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_num_tiles[b]), sizeof(int32_t), cudaHostAllocDefault)));
      }
      pinned_route_cap = max_routes;
    }
    return Status::OK();
  }

  static MoEPackedSharedPrefillWaveStaging& Instance() {
    static MoEPackedSharedPrefillWaveStaging inst;
    return inst;
  }
};

}  // namespace

struct MoEPackedOffloadState {
  int32_t layer_index{-1};
  int32_t num_experts{0};
  int32_t hidden_size{0};
  int32_t inter_size{0};
  int32_t k{0};
  bool is_bf16{true};
  bool normalize_routing_weights{true};
  int gate_type{0};
  int up_type{0};
  int down_type{0};
  size_t gate_row_bytes{0};
  size_t up_row_bytes{0};
  size_t down_row_bytes{0};
  size_t gate_exp_bytes{0};
  size_t up_exp_bytes{0};
  size_t down_exp_bytes{0};
  size_t slot_bytes{0};

  const uint8_t* host_gate_w{nullptr};
  const uint8_t* host_up_w{nullptr};
  const uint8_t* host_down_w{nullptr};

  int32_t resident_slots{0};
  bool is_adaptive{false};
  bool cpu_execution_enabled{true};
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> session_controller;
  std::vector<int32_t> host_expert_to_slot;
  std::vector<int32_t> slot_to_expert;
  std::vector<float> expert_ema;
  std::vector<float> x_f32_buf;
  std::vector<float> inter_buf;
  IAllocatorUniquePtr<uint8_t> gpu_resident_buf;
  uint8_t* gpu_gate_bank{nullptr};
  uint8_t* gpu_up_bank{nullptr};
  uint8_t* gpu_down_bank{nullptr};
  IAllocatorUniquePtr<int32_t> d_expert_to_slot;
  IAllocatorUniquePtr<uint8_t> d_decode_ws;

  MoEPackedDecodePinned* h_decode_pinned{nullptr};
  MoEPackedDecodePinned* d_decode_pinned{nullptr};
  int32_t last_seen_seq{0};

  float tail_sub_threshold{0.0f};
  int32_t tail_sub_min_rank{6};
  float min_expert_weight{0.0f};
  bool skip_tail_cold_only{true};

  // Telemetry counters for ORT_QMOE_SHARED_CPU_TIMING / ORT_QMOE_SHARED_RESIDENT_LOG
  std::atomic<bool> live_session_started{false};
  std::atomic<uint64_t> decode_calls{0};
  std::atomic<uint64_t> decode_routes{0};
  std::atomic<uint64_t> decode_cold_routes{0};
  std::atomic<uint64_t> decode_cold_experts{0};
  std::atomic<uint64_t> decode_cpu_us{0};
  std::atomic<uint64_t> decode_prep_us{0};
  std::atomic<uint64_t> decode_fc1_us{0};
  std::atomic<uint64_t> decode_fc2_us{0};
  std::atomic<uint64_t> prefill_routes{0};
  std::atomic<uint64_t> prefill_hit_routes{0};
  std::atomic<uint64_t> total_promotions{0};
  std::atomic<uint64_t> stream_tiles{0};
  std::atomic<uint64_t> stream_experts{0};
  std::atomic<uint64_t> stream_routes{0};
  std::atomic<uint64_t> stream_waves{0};

  bool InWarmup() const noexcept {
    return session_controller != nullptr && session_controller->RunControls().InWarmup();
  }

  ~MoEPackedOffloadState();
  void RunColdDecodeStep();
};

namespace {

class MoEPackedDoorbellService {
 public:
  static MoEPackedDoorbellService& Instance() {
    static MoEPackedDoorbellService svc;
    return svc;
  }

  void Register(MoEPackedOffloadState* state) {
    std::lock_guard<std::mutex> lk(mu_);
    states_.push_back(state);
    if (!running_.load(std::memory_order_relaxed)) {
      stop_.store(false, std::memory_order_relaxed);
      running_.store(true, std::memory_order_release);
      worker_ = std::thread([this]() { Loop(); });
    }
  }

  void Unregister(MoEPackedOffloadState* state) {
    std::thread to_join;
    bool release_staging = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      states_.erase(std::remove(states_.begin(), states_.end(), state), states_.end());
      if (states_.empty() && running_.load(std::memory_order_relaxed)) {
        stop_.store(true, std::memory_order_release);
        running_.store(false, std::memory_order_release);
        to_join = std::move(worker_);
        release_staging = true;
      }
    }
    if (to_join.joinable()) {
      to_join.join();
    }
    if (release_staging) {
      MoEPackedSharedPrefillWaveStaging::Instance().ReleaseAll();
    }
  }

 private:
  MoEPackedDoorbellService() = default;
  ~MoEPackedDoorbellService() {
    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) worker_.detach();
  }

  void Loop() {
    auto last_active = std::chrono::steady_clock::now();
    for (;;) {
      if (stop_.load(std::memory_order_acquire)) return;
      bool did_work = false;
      {
        std::lock_guard<std::mutex> lk(mu_);
        for (MoEPackedOffloadState* st : states_) {
          if (!st || !st->h_decode_pinned) continue;
          const int32_t seq = st->h_decode_pinned->doorbell_seq;
          if (seq > st->last_seen_seq) {
            std::atomic_thread_fence(std::memory_order_acquire);
            st->last_seen_seq = seq;
            st->RunColdDecodeStep();
            std::atomic_thread_fence(std::memory_order_release);
            st->h_decode_pinned->doorbell_done = seq;
            did_work = true;
          }
        }
      }
      if (did_work) {
        last_active = std::chrono::steady_clock::now();
      } else {
        if (std::chrono::steady_clock::now() - last_active > std::chrono::milliseconds(50)) {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        } else {
#if defined(_M_AMD64) || defined(__x86_64__)
          _mm_pause();
#endif
        }
      }
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<MoEPackedOffloadState*> states_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

}  // namespace

MoEPackedOffloadState::~MoEPackedOffloadState() {
  MoEPackedDoorbellService::Instance().Unregister(this);
  const std::string res_env = GetEnvironmentVar("ORT_QMOE_SHARED_RESIDENT_LOG");
  const std::string cpu_env = GetEnvironmentVar("ORT_QMOE_SHARED_CPU_TIMING");
  const bool log_res = !res_env.empty() && res_env != "0";
  const bool log_cpu = !cpu_env.empty() && cpu_env != "0";
  const uint64_t d_calls = decode_calls.load();
  const uint64_t d_routes = decode_routes.load();
  const uint64_t d_cold_routes = decode_cold_routes.load();
  const uint64_t d_hit_routes = d_routes >= d_cold_routes ? (d_routes - d_cold_routes) : 0;
  const uint64_t p_routes = prefill_routes.load();
  const uint64_t p_hits = prefill_hit_routes.load();
  if (log_res && layer_index >= 0) {
    std::fprintf(stderr,
                 "QMoEOffload staging layer=%d weight_banks=2 pinned_bytes=%zu\n"
                 "QMoEOffload resident layer=%d slots=%d hit_routes=%llu routes=%llu promotions=%llu\n"
                 "QMoEOffload resident_phase layer=%d phase=0 hit_routes=%llu routes=%llu\n"
                 "QMoEOffload resident_phase layer=%d phase=1 hit_routes=%llu routes=%llu\n"
                 "QMoEOffload stream layer=%d tiles=%llu experts=%llu routes=%llu cpu_routes=0 waves=%llu reused_promotions=0\n",
                 layer_index, MoEPackedSharedPrefillWaveStaging::kWaveExperts * slot_bytes,
                 layer_index, resident_slots,
                 static_cast<unsigned long long>(p_hits + d_hit_routes),
                 static_cast<unsigned long long>(p_routes + d_routes),
                 static_cast<unsigned long long>(total_promotions.load()),
                 layer_index, static_cast<unsigned long long>(p_hits), static_cast<unsigned long long>(p_routes),
                 layer_index, static_cast<unsigned long long>(d_hit_routes), static_cast<unsigned long long>(d_routes),
                 layer_index,
                 static_cast<unsigned long long>(stream_tiles.load()),
                 static_cast<unsigned long long>(stream_experts.load()),
                 static_cast<unsigned long long>(stream_routes.load()),
                 static_cast<unsigned long long>(stream_waves.load()));
  }
  if (log_cpu && layer_index >= 0) {
    std::fprintf(stderr,
                 "QMoEOffload cpu layer=%d calls=%llu routes=%llu cold_routes=%llu cold_experts=%llu "
                 "cpu_us=%llu classify_us=0 gap_in_us=0 gap_in_n=0 gap_out_us=0 gap_out_n=0 "
                 "t1_us=%llu t2_us=%llu t3_us=0 inter_n=0 prep_us=%llu fc1_us=%llu fc2_us=%llu write_us=0\n",
                 layer_index,
                 static_cast<unsigned long long>(d_calls),
                 static_cast<unsigned long long>(d_routes),
                 static_cast<unsigned long long>(d_cold_routes),
                 static_cast<unsigned long long>(decode_cold_experts.load()),
                 static_cast<unsigned long long>(decode_cpu_us.load()),
                 static_cast<unsigned long long>(decode_fc1_us.load()),
                 static_cast<unsigned long long>(decode_fc2_us.load()),
                 static_cast<unsigned long long>(decode_prep_us.load()),
                 static_cast<unsigned long long>(decode_fc1_us.load()),
                 static_cast<unsigned long long>(decode_fc2_us.load()));
  }
  if (h_decode_pinned) {
    cudaFreeHost(h_decode_pinned);
    h_decode_pinned = nullptr;
  }
}

void MoEPackedOffloadState::RunColdDecodeStep() {
  const auto t0 = std::chrono::steady_clock::now();
  const int nrows = std::clamp<int>(h_decode_pinned->num_rows, 0, kMoEPackedMaxDecodeRows);
  const int ccount = std::clamp<int>(h_decode_pinned->cold_count, 0, kMoEPackedMaxDecodeRoutes);
  if (!live_session_started.load(std::memory_order_acquire) || InWarmup()) {
    if (nrows > 0 && ccount > 0) {
      std::memset(h_decode_pinned->host_cold_output, 0, static_cast<size_t>(nrows * hidden_size) * sizeof(uint16_t));
    }
    return;
  }
  decode_calls.fetch_add(1, std::memory_order_relaxed);
  decode_routes.fetch_add(static_cast<uint64_t>(nrows * k), std::memory_order_relaxed);
  decode_cold_routes.fetch_add(static_cast<uint64_t>(ccount), std::memory_order_relaxed);
  if (ccount <= 0 || nrows <= 0) {
    return;
  }

  // 1. Group cold routes by unique expert
  struct ColdExpertGroup {
    int expert;
    int count;
    int token_rows[kMoEPackedMaxDecodeRows];
    float scales[kMoEPackedMaxDecodeRows];
  };
  ColdExpertGroup groups[kMoEPackedMaxDecodeRoutes];
  int num_groups = 0;
  for (int i = 0; i < ccount; ++i) {
    const int r = h_decode_pinned->cold_routes[i];
    const int exp = h_decode_pinned->cold_experts[i];
    const float sc = h_decode_pinned->cold_scales[i];
    const int trow = r / k;
    if (exp < 0 || exp >= num_experts || trow < 0 || trow >= nrows || sc == 0.0f) continue;
    int g_idx = -1;
    for (int g = 0; g < num_groups; ++g) {
      if (groups[g].expert == exp) {
        g_idx = g;
        break;
      }
    }
    if (g_idx < 0) {
      g_idx = num_groups++;
      groups[g_idx].expert = exp;
      groups[g_idx].count = 0;
    }
    if (groups[g_idx].count < kMoEPackedMaxDecodeRows) {
      const int idx = groups[g_idx].count++;
      groups[g_idx].token_rows[idx] = trow;
      groups[g_idx].scales[idx] = sc;
    }
  }
  decode_cold_experts.fetch_add(static_cast<uint64_t>(num_groups), std::memory_order_relaxed);
  if (num_groups <= 0) {
    std::memset(h_decode_pinned->host_cold_output, 0, static_cast<size_t>(nrows * hidden_size) * sizeof(uint16_t));
    return;
  }

  // 2. Convert input rows from BF16/FP16 to FP32 using pre-allocated x_f32_buf
  for (int r = 0; r < nrows; ++r) {
    const uint16_t* src = h_decode_pinned->host_input + static_cast<size_t>(r) * hidden_size;
    float* dst = x_f32_buf.data() + static_cast<size_t>(r) * kMoEPackedMaxHiddenSize;
    if (is_bf16) {
      for (int c = 0; c < hidden_size; ++c) {
        dst[c] = gguf_quant::Bf16BitsToFloatCpu(src[c]);
      }
    } else {
      for (int c = 0; c < hidden_size; ++c) {
        dst[c] = gguf_quant::Fp16BitsToFloatCpu(src[c]);
      }
    }
  }

  // Assign flat route offset for each group's routes
  int group_route_base[kMoEPackedMaxDecodeRoutes];
  int total_active_routes = 0;
  for (int g = 0; g < num_groups; ++g) {
    group_route_base[g] = total_active_routes;
    total_active_routes += groups[g].count;
  }
  if (inter_buf.size() < static_cast<size_t>(total_active_routes) * inter_size) {
    inter_buf.resize(static_cast<size_t>(total_active_routes) * inter_size);
  }
  const auto t_prep = std::chrono::steady_clock::now();

  // 3. Pass 1: Parallel FC1 (gate + up) + SwiGLU across (group, 32-channel tiles)
  constexpr int kFc1TileCh = 32;
  const int fc1_tiles_per_group = (inter_size + kFc1TileCh - 1) / kFc1TileCh;
  const int total_fc1_tasks = num_groups * fc1_tiles_per_group;
  MoEPackedParallelPool::Instance().ParallelFor(total_fc1_tasks, [&](int task_idx) {
    const int g = task_idx / fc1_tiles_per_group;
    const int tile_idx = task_idx - g * fc1_tiles_per_group;
    const int ch_start = tile_idx * kFc1TileCh;
    const int ch_end = std::min(inter_size, ch_start + kFc1TileCh);
    const int exp = groups[g].expert;
    const int R = groups[g].count;
    const float* x_ptrs[kMoEPackedMaxDecodeRows];
    for (int r = 0; r < R; ++r) {
      x_ptrs[r] = x_f32_buf.data() + static_cast<size_t>(groups[g].token_rows[r]) * kMoEPackedMaxHiddenSize;
    }
    float* group_inter = inter_buf.data() + static_cast<size_t>(group_route_base[g]) * inter_size;
    for (int j = ch_start; j < ch_end; ++j) {
      const size_t row_idx = static_cast<size_t>(exp) * inter_size + j;
      const uint8_t* g_row = host_gate_w + row_idx * gate_row_bytes;
      const uint8_t* u_row = host_up_w + row_idx * up_row_bytes;
      float g_dots[kMoEPackedMaxDecodeRows];
      float u_dots[kMoEPackedMaxDecodeRows];
      GgufDualDotRowCpu(g_row, u_row, gate_type, up_type, hidden_size, x_ptrs, R, g_dots, u_dots);
      for (int r = 0; r < R; ++r) {
        const float gv = is_bf16 ? RoundBf16Scalar(g_dots[r]) : g_dots[r];
        const float uv = is_bf16 ? RoundBf16Scalar(u_dots[r]) : u_dots[r];
        const float swiglu = uv * (gv / (1.0f + std::exp(-gv)));
        group_inter[static_cast<size_t>(r) * inter_size + j] = is_bf16 ? RoundBf16Scalar(swiglu) : swiglu;
      }
    }
  });
  const auto t_fc1 = std::chrono::steady_clock::now();

  // 4. Pass 2: Parallel FC2 partitioned by 64-column output slices across all cold groups (lock-free, L1-resident inter_buf)
  constexpr int kFc2TileCols = 64;
  const int total_fc2_tasks = (hidden_size + kFc2TileCols - 1) / kFc2TileCols;
  MoEPackedParallelPool::Instance().ParallelFor(total_fc2_tasks, [&](int task_idx) {
    const int col_start = task_idx * kFc2TileCols;
    const int col_end = std::min(hidden_size, col_start + kFc2TileCols);
    const int col_cnt = col_end - col_start;
    float acc_out[kMoEPackedMaxDecodeRows][kFc2TileCols];
    for (int r = 0; r < nrows; ++r) {
      for (int c = 0; c < col_cnt; ++c) acc_out[r][c] = 0.0f;
    }
    for (int g = 0; g < num_groups; ++g) {
      const int exp = groups[g].expert;
      const int R = groups[g].count;
      const float* inter_ptrs[kMoEPackedMaxDecodeRows];
      const float* group_inter = inter_buf.data() + static_cast<size_t>(group_route_base[g]) * inter_size;
      for (int r = 0; r < R; ++r) {
        inter_ptrs[r] = group_inter + static_cast<size_t>(r) * inter_size;
      }
      int c = 0;
#if defined(_M_AMD64) || defined(__x86_64__)
      if (R == 1 && down_type == gguf_quant::kGgmlTypeIQ4_NL) {
        const float* xr0 = inter_ptrs[0];
        const int trow0 = groups[g].token_rows[0];
        const float sc0 = groups[g].scales[0];
        const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(gguf_quant::kvalues_iq4nl));
        const __m128i mask0f = _mm_set1_epi8(0x0F);
        const int nb = inter_size / 32;
        for (; c + 4 <= col_cnt; c += 4) {
          const int col = col_start + c;
          const uint8_t* dr0 = host_down_w + (static_cast<size_t>(exp) * hidden_size + col + 0) * down_row_bytes;
          const uint8_t* dr1 = dr0 + down_row_bytes;
          const uint8_t* dr2 = dr1 + down_row_bytes;
          const uint8_t* dr3 = dr2 + down_row_bytes;
          __m256 a0 = _mm256_setzero_ps();
          __m256 a1 = _mm256_setzero_ps();
          __m256 a2 = _mm256_setzero_ps();
          __m256 a3 = _mm256_setzero_ps();
          for (int i = 0; i < nb; ++i) {
            const float* x_blk = xr0 + i * 32;
            const __m256 x0 = _mm256_loadu_ps(x_blk + 0);
            const __m256 x1 = _mm256_loadu_ps(x_blk + 8);
            const __m256 x2 = _mm256_loadu_ps(x_blk + 16);
            const __m256 x3 = _mm256_loadu_ps(x_blk + 24);
            auto step_row = [&](const uint8_t* blk, __m256& acc) {
              const __m256 vd = _mm256_set1_ps(gguf_quant::Fp16BitsToFloatCpu(gguf_quant::ReadU16Le(blk)));
              const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
              const __m128i v_lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw16, mask0f));
              const __m128i v_hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f));
              acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo))), x0, acc);
              acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8)))), x1, acc);
              acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi))), x2, acc);
              acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8)))), x3, acc);
            };
            step_row(dr0 + i * 18, a0);
            step_row(dr1 + i * 18, a1);
            step_row(dr2 + i * 18, a2);
            step_row(dr3 + i * 18, a3);
          }
          const float d0 = HorizontalSumAvx2(a0);
          const float d1 = HorizontalSumAvx2(a1);
          const float d2 = HorizontalSumAvx2(a2);
          const float d3 = HorizontalSumAvx2(a3);
          acc_out[trow0][c + 0] += sc0 * (is_bf16 ? RoundBf16Scalar(d0) : d0);
          acc_out[trow0][c + 1] += sc0 * (is_bf16 ? RoundBf16Scalar(d1) : d1);
          acc_out[trow0][c + 2] += sc0 * (is_bf16 ? RoundBf16Scalar(d2) : d2);
          acc_out[trow0][c + 3] += sc0 * (is_bf16 ? RoundBf16Scalar(d3) : d3);
        }
      } else if (R == 1 && down_type == gguf_quant::kGgmlTypeQ2_0) {
        const float* xr0 = inter_ptrs[0];
        const int trow0 = groups[g].token_rows[0];
        const float sc0 = groups[g].scales[0];
        const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        const __m256i mask3 = _mm256_set1_epi32(3);
        const __m256i ones = _mm256_set1_epi32(1);
        const int nb = inter_size / 64;
        for (; c + 4 <= col_cnt; c += 4) {
          const int col = col_start + c;
          const uint8_t* dr0 = host_down_w + (static_cast<size_t>(exp) * hidden_size + col + 0) * down_row_bytes;
          const uint8_t* dr1 = dr0 + down_row_bytes;
          const uint8_t* dr2 = dr1 + down_row_bytes;
          const uint8_t* dr3 = dr2 + down_row_bytes;
          __m256 a0 = _mm256_setzero_ps();
          __m256 a1 = _mm256_setzero_ps();
          __m256 a2 = _mm256_setzero_ps();
          __m256 a3 = _mm256_setzero_ps();
          for (int i = 0; i < nb; ++i) {
            const uint8_t* b0 = dr0 + i * 18;
            const uint8_t* b1 = dr1 + i * 18;
            const uint8_t* b2 = dr2 + i * 18;
            const uint8_t* b3 = dr3 + i * 18;
            const __m256 vd0 = _mm256_set1_ps(gguf_quant::Fp16BitsToFloatCpu(gguf_quant::ReadU16Le(b0)));
            const __m256 vd1 = _mm256_set1_ps(gguf_quant::Fp16BitsToFloatCpu(gguf_quant::ReadU16Le(b1)));
            const __m256 vd2 = _mm256_set1_ps(gguf_quant::Fp16BitsToFloatCpu(gguf_quant::ReadU16Le(b2)));
            const __m256 vd3 = _mm256_set1_ps(gguf_quant::Fp16BitsToFloatCpu(gguf_quant::ReadU16Le(b3)));
            for (int w = 0; w < 8; ++w) {
              const __m256 xv = _mm256_loadu_ps(xr0 + i * 64 + w * 8);
              auto q_f32 = [&](uint16_t q16, __m256 vd) {
                const __m256i qv = _mm256_sub_epi32(
                    _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(q16), shifts), mask3), ones);
                return _mm256_mul_ps(vd, _mm256_cvtepi32_ps(qv));
              };
              a0 = _mm256_fmadd_ps(q_f32(gguf_quant::ReadU16Le(b0 + 2 + w * 2), vd0), xv, a0);
              a1 = _mm256_fmadd_ps(q_f32(gguf_quant::ReadU16Le(b1 + 2 + w * 2), vd1), xv, a1);
              a2 = _mm256_fmadd_ps(q_f32(gguf_quant::ReadU16Le(b2 + 2 + w * 2), vd2), xv, a2);
              a3 = _mm256_fmadd_ps(q_f32(gguf_quant::ReadU16Le(b3 + 2 + w * 2), vd3), xv, a3);
            }
          }
          const float d0 = HorizontalSumAvx2(a0);
          const float d1 = HorizontalSumAvx2(a1);
          const float d2 = HorizontalSumAvx2(a2);
          const float d3 = HorizontalSumAvx2(a3);
          acc_out[trow0][c + 0] += sc0 * (is_bf16 ? RoundBf16Scalar(d0) : d0);
          acc_out[trow0][c + 1] += sc0 * (is_bf16 ? RoundBf16Scalar(d1) : d1);
          acc_out[trow0][c + 2] += sc0 * (is_bf16 ? RoundBf16Scalar(d2) : d2);
          acc_out[trow0][c + 3] += sc0 * (is_bf16 ? RoundBf16Scalar(d3) : d3);
        }
      }
#endif
      for (; c < col_cnt; ++c) {
        const int col = col_start + c;
        const uint8_t* d_row = host_down_w + (static_cast<size_t>(exp) * hidden_size + col) * down_row_bytes;
        float d_dots[kMoEPackedMaxDecodeRows];
        GgufDotRowCpu(d_row, down_type, inter_size, inter_ptrs, R, d_dots);
        for (int r = 0; r < R; ++r) {
          const float dv = is_bf16 ? RoundBf16Scalar(d_dots[r]) : d_dots[r];
          acc_out[groups[g].token_rows[r]][c] += groups[g].scales[r] * dv;
        }
      }
    }
    for (int r = 0; r < nrows; ++r) {
      uint16_t* dst = h_decode_pinned->host_cold_output + static_cast<size_t>(r) * hidden_size + col_start;
      if (is_bf16) {
        for (int c = 0; c < col_cnt; ++c) {
          dst[c] = FloatToBf16Bits(acc_out[r][c]);
        }
      } else {
        for (int c = 0; c < col_cnt; ++c) {
          dst[c] = MLFloat16(acc_out[r][c]).val;
        }
      }
    }
  });
  const auto t_fc2 = std::chrono::steady_clock::now();

  decode_prep_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_prep - t0).count()),
      std::memory_order_relaxed);
  decode_fc1_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_fc1 - t_prep).count()),
      std::memory_order_relaxed);
  decode_fc2_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_fc2 - t_fc1).count()),
      std::memory_order_relaxed);
  decode_cpu_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_fc2 - t0).count()),
      std::memory_order_relaxed);
}

#define REGISTER_MOE_PACKED_CUDA(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      MoEPacked,                                                         \
      kMSDomain, 1,                                                      \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()), \
      MoEPacked<T>);

REGISTER_MOE_PACKED_CUDA(MLFloat16)
REGISTER_MOE_PACKED_CUDA(BFloat16)

#undef REGISTER_MOE_PACKED_CUDA

template <typename T>
MoEPacked<T>::MoEPacked(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("k", &k_).IsOK() && k_ > 0 && k_ <= 16,
              "MoEPacked requires k in 1..16");
  normalize_routing_weights_ = info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 1);
  activation_type_ = info.GetAttrOrDefault<std::string>("activation_type", "swiglu");
  ORT_ENFORCE(activation_type_ == "swiglu", "MoEPacked only supports activation_type='swiglu'");
  ORT_ENFORCE(info.GetAttr<int64_t>("gate_type", &gate_type_).IsOK(), "MoEPacked requires gate_type");
  ORT_ENFORCE(info.GetAttr<int64_t>("up_type", &up_type_).IsOK(), "MoEPacked requires up_type");
  ORT_ENFORCE(info.GetAttr<int64_t>("down_type", &down_type_).IsOK(), "MoEPacked requires down_type");
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(gate_type_), "MoEPacked: unsupported gate_type=", gate_type_);
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(up_type_), "MoEPacked: unsupported up_type=", up_type_);
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(down_type_), "MoEPacked: unsupported down_type=", down_type_);
  if constexpr (std::is_same_v<T, BFloat16>) {
    ORT_ENFORCE(GetDeviceProp().major >= 8, "MoEPacked BF16 requires compute capability 8.0 or higher");
    ORT_THROW_IF_ERROR(GetMoEPackedBf16KernelSupport(GetDeviceId(), bf16_tc_supported_));
  }

  // Check if inputs 2..4 were placed on CPU by PlaceMoeOffloadInitializersOnCpu for hybrid offload.
  const Tensor* gate_w = nullptr;
  const Tensor* up_w = nullptr;
  const Tensor* down_w = nullptr;
  if (info.TryGetConstantInput(2, &gate_w) && gate_w &&
      info.TryGetConstantInput(3, &up_w) && up_w &&
      info.TryGetConstantInput(4, &down_w) && down_w &&
      gate_w->Location().device.Type() == OrtDevice::CPU &&
      up_w->Location().device.Type() == OrtDevice::CPU &&
      down_w->Location().device.Type() == OrtDevice::CPU &&
      gate_w->Shape().NumDimensions() == 3 &&
      up_w->Shape().NumDimensions() == 3 &&
      down_w->Shape().NumDimensions() == 3) {
    using onnxruntime::moe_offload::MoeOffloadSessionPolicy;
    auto controller = GetQMoESessionController();
    std::filesystem::path model_path = controller ? controller->GetModelPath() : std::filesystem::path{};
    ORT_THROW_IF_ERROR(MoeOffloadSessionPolicy::ParseFromConfigOptions(
        info.GetConfigOptions(), model_path, offload_policy_));
    if (controller != nullptr && offload_policy_.mode != onnxruntime::moe_offload::OffloadMode::kDisabled) {
      auto get_input_name = [&](size_t slot) -> std::string {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
        if (slot < info.GetKernelInfo().GetInputCount()) {
          return info.GetKernelInfo().GetInputName(slot);
        }
#else
        const auto& in_defs = info.node().InputDefs();
        if (slot < in_defs.size() && in_defs[slot] != nullptr && in_defs[slot]->Exists()) {
          return in_defs[slot]->Name();
        }
#endif
        return {};
      };
      int32_t parsed_layer = -1;
      if (MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(info.node().Name(), parsed_layer)) {
        offload_layer_index_ = parsed_layer;
      } else {
        const std::string in2_name = get_input_name(2);
        if (!in2_name.empty() && MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(in2_name, parsed_layer)) {
          offload_layer_index_ = parsed_layer;
        }
      }

      const int32_t num_e = static_cast<int32_t>(gate_w->Shape()[0]);
      const int32_t inter = static_cast<int32_t>(gate_w->Shape()[1]);
      const int32_t hid = static_cast<int32_t>(down_w->Shape()[1]);
      offload_source_desc_.op_type = "MoEPacked";
      offload_source_desc_.quant_type = "gguf";
      offload_source_desc_.stable_node_id = info.node().Name();
      offload_source_desc_.layer_index = offload_layer_index_;
      offload_source_desc_.num_experts = num_e;
      offload_source_desc_.top_k = static_cast<int32_t>(k_);
      offload_source_desc_.hidden_size = hid;
      offload_source_desc_.inter_size = inter;
      offload_source_desc_.block_size = 32;
      offload_source_desc_.weight_bits = 2;
      offload_source_desc_.weights_prepacked = true;
      offload_source_desc_.placement.device_id = controller->DeviceId();
      offload_source_desc_.shard_spec.global_num_experts = num_e;
      offload_source_desc_.shard_spec.local_num_experts = num_e;
      offload_source_desc_.shard_spec.global_hidden_size = hid;
      offload_source_desc_.shard_spec.global_inter_size = inter;
      offload_source_desc_.shard_spec.local_hidden_size = hid;
      offload_source_desc_.shard_spec.local_inter_size = inter;
      offload_source_desc_.shard_spec.quant_block_size = 32;

      auto record_slice = [&](onnxruntime::moe_offload::ExpertBankKind kind, int slot_idx, const Tensor* t) {
        onnxruntime::moe_offload::MoeExpertBankSlice slice;
        slice.kind = kind;
        slice.input_index = slot_idx;
        slice.initializer_name = get_input_name(static_cast<size_t>(slot_idx));
        slice.elem_type = static_cast<ONNXTensorElementDataType>(t->GetElementType());
        slice.full_shape = t->Shape();
        slice.total_bytes = t->SizeInBytes();
        slice.bytes_per_expert = t->SizeInBytes() / static_cast<size_t>(num_e);
        slice.host_data = t->DataRaw();
        slice.is_file_mapped = onnxruntime::moe_offload::IsReloadableFileMapping(slice.host_data, slice.total_bytes);
        offload_source_desc_.banks[static_cast<size_t>(kind)] = std::move(slice);
      };
      record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Weights, 2, gate_w);
      record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Scales, 3, up_w);
      record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc2Weights, 4, down_w);

      const size_t slot_bytes =
          (gate_w->SizeInBytes() + up_w->SizeInBytes() + down_w->SizeInBytes()) / static_cast<size_t>(num_e);
      offload_capabilities_.supports_full_gpu_resident = true;
      offload_capabilities_.supports_hybrid_doorbell_decode = true;
      offload_capabilities_.supports_gpu_streaming_prefill = true;
      offload_capabilities_.supports_file_backed_bounded_host_cache = true;
      offload_capabilities_.supports_cuda_graph_capture = true;
      offload_capabilities_.decode_kernel_skips_negative_slots = true;
      offload_capabilities_.source_bytes_per_expert = slot_bytes;
      offload_capabilities_.gpu_slot_bytes = slot_bytes;
      offload_capabilities_.gpu_fixed_table_bytes = static_cast<size_t>(2) * num_e * sizeof(int32_t);
      offload_capabilities_.cpu_slot_bytes = slot_bytes;
      offload_capabilities_.min_host_cache_slots = 1;

      controller->RegisterMoeOffloadNode(this, offload_policy_, offload_layer_index_);
      offload_node_registered_ = true;
    }
  }
}

template <typename T>
MoEPacked<T>::~MoEPacked() {
  if (offload_node_registered_) {
    if (auto controller = GetQMoESessionController()) {
      if (offload_state_ != nullptr) {
        controller->UnregisterOffloadNode(this);
      }
      controller->UnregisterMoeOffloadNode(this);
    }
  }
}

template <typename T>
Status MoEPacked<T>::InitializeAdmittedPlan(const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                                            const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan) {
  offload_policy_ = policy;
  offload_plan_ = plan;
  offload_layer_index_ = plan.layer_index;

  const Tensor* gate_w = nullptr;
  const Tensor* up_w = nullptr;
  const Tensor* down_w = nullptr;
  ORT_RETURN_IF_NOT(Info().TryGetConstantInput(2, &gate_w) && gate_w &&
                        Info().TryGetConstantInput(3, &up_w) && up_w &&
                        Info().TryGetConstantInput(4, &down_w) && down_w,
                    "MoEPacked InitializeAdmittedPlan requires constant expert weight inputs 2..4.");

  auto st = std::make_shared<MoEPackedOffloadState>();
  st->layer_index = plan.layer_index;
  st->num_experts = static_cast<int32_t>(gate_w->Shape()[0]);
  st->inter_size = static_cast<int32_t>(gate_w->Shape()[1]);
  st->hidden_size = static_cast<int32_t>(down_w->Shape()[1]);
  st->k = static_cast<int32_t>(k_);
  st->is_bf16 = std::is_same_v<T, BFloat16>;
  st->normalize_routing_weights = (normalize_routing_weights_ != 0);
  st->gate_type = static_cast<int>(gate_type_);
  st->up_type = static_cast<int>(up_type_);
  st->down_type = static_cast<int>(down_type_);
  st->gate_row_bytes = static_cast<size_t>(gate_w->Shape()[2]);
  st->up_row_bytes = static_cast<size_t>(up_w->Shape()[2]);
  st->down_row_bytes = static_cast<size_t>(down_w->Shape()[2]);
  st->gate_exp_bytes = static_cast<size_t>(st->inter_size) * st->gate_row_bytes;
  st->up_exp_bytes = static_cast<size_t>(st->inter_size) * st->up_row_bytes;
  st->down_exp_bytes = static_cast<size_t>(st->hidden_size) * st->down_row_bytes;
  st->slot_bytes = st->gate_exp_bytes + st->up_exp_bytes + st->down_exp_bytes;

  st->host_gate_w = static_cast<const uint8_t*>(gate_w->DataRaw());
  st->host_up_w = static_cast<const uint8_t*>(up_w->DataRaw());
  st->host_down_w = static_cast<const uint8_t*>(down_w->DataRaw());

  st->tail_sub_threshold = policy.tail_substitution_threshold;
  st->tail_sub_min_rank = policy.tail_substitution_min_rank;
  st->min_expert_weight = policy.min_expert_weight;
  st->skip_tail_cold_only = policy.skip_tail_cold_only;
  const std::string det_env = GetEnvironmentVar("ORT_QMOE_DETERMINISTIC_REPLAY");
  const bool det_replay = !det_env.empty() && det_env != "0";
  st->is_adaptive = !det_replay && (policy.placement == onnxruntime::moe_offload::PlacementPolicy::kAdaptive);
  st->cpu_execution_enabled = (policy.cpu_execution != onnxruntime::moe_offload::CpuExecutionMode::kDisabled);
  st->session_controller = GetQMoESessionController();
  st->x_f32_buf.resize(static_cast<size_t>(kMoEPackedMaxDecodeRows) * kMoEPackedMaxHiddenSize, 0.0f);
  st->inter_buf.resize(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * st->inter_size, 0.0f);

  st->resident_slots = std::clamp<int32_t>(plan.resident_slots, 0, st->num_experts);
  st->host_expert_to_slot.assign(static_cast<size_t>(st->num_experts), -1);
  st->expert_ema.assign(static_cast<size_t>(st->num_experts), 0.0f);

  st->slot_to_expert.reserve(static_cast<size_t>(st->resident_slots));
  for (int32_t exp : plan.initial_resident_experts) {
    if (static_cast<int32_t>(st->slot_to_expert.size()) >= st->resident_slots) break;
    if (exp >= 0 && exp < st->num_experts && st->host_expert_to_slot[static_cast<size_t>(exp)] < 0) {
      const int32_t slot = static_cast<int32_t>(st->slot_to_expert.size());
      st->host_expert_to_slot[static_cast<size_t>(exp)] = slot;
      st->slot_to_expert.push_back(exp);
      st->expert_ema[static_cast<size_t>(exp)] =
          10.0f - 6.0f * (static_cast<float>(slot) / static_cast<float>(std::max(1, st->resident_slots)));
    }
  }
  for (int32_t exp = 0; exp < st->num_experts && static_cast<int32_t>(st->slot_to_expert.size()) < st->resident_slots; ++exp) {
    if (st->host_expert_to_slot[static_cast<size_t>(exp)] < 0) {
      const int32_t slot = static_cast<int32_t>(st->slot_to_expert.size());
      st->host_expert_to_slot[static_cast<size_t>(exp)] = slot;
      st->slot_to_expert.push_back(exp);
      st->expert_ema[static_cast<size_t>(exp)] = 2.0f;
    }
  }
  st->resident_slots = static_cast<int32_t>(st->slot_to_expert.size());

  AllocatorPtr alloc = Info().GetAllocator(OrtMemTypeDefault);
  ORT_RETURN_IF_NOT(alloc != nullptr, "MoEPacked InitializeAdmittedPlan requires a valid CUDA device allocator.");

  if (st->resident_slots > 0) {
    const size_t total_gate = static_cast<size_t>(st->resident_slots) * st->gate_exp_bytes;
    const size_t total_up = static_cast<size_t>(st->resident_slots) * st->up_exp_bytes;
    const size_t total_down = static_cast<size_t>(st->resident_slots) * st->down_exp_bytes;
    st->gpu_resident_buf = IAllocator::MakeUniquePtr<uint8_t>(alloc, total_gate + total_up + total_down, false, nullptr);
    st->gpu_gate_bank = st->gpu_resident_buf.get();
    st->gpu_up_bank = st->gpu_gate_bank + total_gate;
    st->gpu_down_bank = st->gpu_up_bank + total_up;

    for (int32_t s = 0; s < st->resident_slots; ++s) {
      const int32_t exp = st->slot_to_expert[static_cast<size_t>(s)];
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          st->gpu_gate_bank + static_cast<size_t>(s) * st->gate_exp_bytes,
          st->host_gate_w + static_cast<size_t>(exp) * st->gate_exp_bytes,
          st->gate_exp_bytes, cudaMemcpyHostToDevice)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          st->gpu_up_bank + static_cast<size_t>(s) * st->up_exp_bytes,
          st->host_up_w + static_cast<size_t>(exp) * st->up_exp_bytes,
          st->up_exp_bytes, cudaMemcpyHostToDevice)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          st->gpu_down_bank + static_cast<size_t>(s) * st->down_exp_bytes,
          st->host_down_w + static_cast<size_t>(exp) * st->down_exp_bytes,
          st->down_exp_bytes, cudaMemcpyHostToDevice)));
    }
    onnxruntime::cuda::QMoESessionController::EvictUploadedHostPages(offload_source_desc_, st->slot_to_expert);
  }

  st->d_expert_to_slot = IAllocator::MakeUniquePtr<int32_t>(alloc, static_cast<size_t>(st->num_experts), false, nullptr);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
      st->d_expert_to_slot.get(), st->host_expert_to_slot.data(),
      static_cast<size_t>(st->num_experts) * sizeof(int32_t), cudaMemcpyHostToDevice)));

  // Allocate persistent decode workspace on GPU and mapped pinned doorbell state on host
  size_t decode_ws_bytes = 0;
  decode_ws_bytes += AlignUp256(sizeof(int));
  decode_ws_bytes += AlignUp256(sizeof(int));
  decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(float));
  decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(int));
  decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(int));
  decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(MoEPackedTile));
  decode_ws_bytes += AlignUp256(sizeof(int));
  decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * st->inter_size * sizeof(T));
  decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * st->hidden_size * sizeof(T));
  st->d_decode_ws = IAllocator::MakeUniquePtr<uint8_t>(alloc, decode_ws_bytes, false, nullptr);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemset(st->d_decode_ws.get(), 0, decode_ws_bytes)));

  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(
      reinterpret_cast<void**>(&st->h_decode_pinned), sizeof(MoEPackedDecodePinned), cudaHostAllocMapped)));
  std::memset(st->h_decode_pinned, 0, sizeof(MoEPackedDecodePinned));
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostGetDevicePointer(
      reinterpret_cast<void**>(&st->d_decode_pinned), st->h_decode_pinned, 0)));

  if (st->session_controller) {
    st->session_controller->RegisterOffloadNode();
  }
  MoEPackedDoorbellService::Instance().Register(st.get());
  offload_state_ = std::move(st);
  return Status::OK();
}

template <typename T>
Status MoEPacked<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* gate_w = context->Input<Tensor>(2);
  const Tensor* up_w = context->Input<Tensor>(3);
  const Tensor* down_w = context->Input<Tensor>(4);

  const auto& in_shape = input->Shape();
  const auto& rp_shape = router_probs->Shape();
  const auto& gw_shape = gate_w->Shape();
  const auto& uw_shape = up_w->Shape();
  const auto& dw_shape = down_w->Shape();

  ORT_RETURN_IF_NOT(in_shape.NumDimensions() >= 2, "MoEPacked input must have rank >= 2");
  ORT_RETURN_IF_NOT(rp_shape.NumDimensions() == 2, "MoEPacked router_probs must be 2D [num_rows, num_experts]");
  ORT_RETURN_IF_NOT(gw_shape.NumDimensions() == 3 && uw_shape.NumDimensions() == 3 && dw_shape.NumDimensions() == 3,
                    "MoEPacked expert weights must be 3D [num_experts, out_dim, row_bytes]");

  const size_t last_axis = in_shape.NumDimensions() - 1;
  const int64_t hidden_size = in_shape[last_axis];
  const int64_t num_rows = in_shape.SizeToDimension(last_axis);
  const int64_t num_experts = rp_shape[1];
  const int64_t inter_size = gw_shape[1];

  ORT_RETURN_IF_NOT(hidden_size > 0 && hidden_size <= kMoEPackedMaxDimension &&
                        inter_size > 0 && inter_size <= kMoEPackedMaxDimension,
                    "MoEPacked requires hidden_size and inter_size in 1..", kMoEPackedMaxDimension);
  ORT_RETURN_IF_NOT(num_rows >= 0, "MoEPacked requires a nonnegative row count");
  ORT_RETURN_IF_NOT(rp_shape[0] == num_rows, "MoEPacked router_probs row count mismatch");
  ORT_RETURN_IF_NOT(num_experts > 0 && num_experts <= 512, "MoEPacked supports up to 512 experts");
  ORT_RETURN_IF_NOT(k_ <= num_experts, "MoEPacked requires k <= num_experts");
  ORT_RETURN_IF_NOT(gw_shape[0] == num_experts && uw_shape[0] == num_experts && dw_shape[0] == num_experts,
                    "MoEPacked expert count mismatch");
  ORT_RETURN_IF_NOT(uw_shape[1] == inter_size && dw_shape[1] == hidden_size,
                    "MoEPacked expert inner/outer dimension mismatch");
  ORT_RETURN_IF_NOT((hidden_size % 32) == 0 && (inter_size % 32) == 0,
                    "MoEPacked requires hidden_size and inter_size to be multiples of 32");

  const int64_t exp_gate_rb = gguf_quant::GgmlRowBytes(static_cast<int>(gate_type_), hidden_size);
  const int64_t exp_up_rb = gguf_quant::GgmlRowBytes(static_cast<int>(up_type_), hidden_size);
  const int64_t exp_down_rb = gguf_quant::GgmlRowBytes(static_cast<int>(down_type_), inter_size);
  ORT_RETURN_IF_NOT(exp_gate_rb > 0 && exp_up_rb > 0 && exp_down_rb > 0 &&
                        gw_shape[2] == exp_gate_rb && uw_shape[2] == exp_up_rb && dw_shape[2] == exp_down_rb,
                    "MoEPacked expert weight row_bytes mismatch");
  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  ORT_RETURN_IF_NOT(PackedTensorBytesFit(num_rows, hidden_size, sizeof(T)) &&
                        PackedTensorBytesFit(num_rows, num_experts, sizeof(T)) &&
                        PackedTensorBytesFit(num_experts * inter_size, exp_gate_rb, 1) &&
                        PackedTensorBytesFit(num_experts * inter_size, exp_up_rb, 1) &&
                        PackedTensorBytesFit(num_experts * hidden_size, exp_down_rb, 1),
                    "MoEPacked tensor byte size exceeds the supported address range");

  // Hybrid CPU/GPU Offload path when weights were admitted on CPU by QMoESessionController
  if (offload_state_ != nullptr) {
    MoEPackedOffloadState& st = *offload_state_;
    if (st.InWarmup()) {
      st.live_session_started.store(false, std::memory_order_release);
    } else {
      st.live_session_started.store(true, std::memory_order_release);
    }
    // Read dynamic environment overrides if set (matching QMoEOffload test hooks)
    float eff_tail_sub_thr = st.tail_sub_threshold;
    int eff_tail_sub_rank = st.tail_sub_min_rank;
    float eff_min_weight = st.min_expert_weight;
    const std::string env_thr = GetEnvironmentVar("ORT_QMOE_TAIL_SUBSTITUTION_THRESHOLD");
    if (!env_thr.empty()) static_cast<void>(TryParseStringWithClassicLocale(env_thr, eff_tail_sub_thr));
    const std::string env_rnk = GetEnvironmentVar("ORT_QMOE_TAIL_SUBSTITUTION_MIN_RANK");
    if (!env_rnk.empty()) static_cast<void>(TryParseStringWithClassicLocale(env_rnk, eff_tail_sub_rank));
    const std::string env_mw = GetEnvironmentVar("ORT_QMOE_MIN_EXPERT_WEIGHT");
    if (!env_mw.empty()) static_cast<void>(TryParseStringWithClassicLocale(env_mw, eff_min_weight));

    // 1. Decode path (num_rows <= 16, CUDA-graph-capturable)
    if (st.cpu_execution_enabled &&
        num_rows <= kMoEPackedMaxDecodeRows && num_rows * k_ <= kMoEPackedMaxDecodeRoutes &&
        hidden_size <= kMoEPackedMaxHiddenSize) {
      MoEPackedParams p{};
      p.num_rows = static_cast<int>(num_rows);
      p.num_experts = static_cast<int>(num_experts);
      p.hidden_size = static_cast<int>(hidden_size);
      p.inter_size = static_cast<int>(inter_size);
      p.k = static_cast<int>(k_);
      p.normalize_routing_weights = (normalize_routing_weights_ != 0);
      p.gate_type = static_cast<int>(gate_type_);
      p.up_type = static_cast<int>(up_type_);
      p.down_type = static_cast<int>(down_type_);
      p.gate_row_bytes = exp_gate_rb;
      p.up_row_bytes = exp_up_rb;
      p.down_row_bytes = exp_down_rb;

      return LaunchMoEPackedOffloadDecodeCuda(
          Stream(context),
          is_bf16,
          input->DataRaw(),
          router_probs->DataRaw(),
          st.gpu_gate_bank,
          st.gpu_up_bank,
          st.gpu_down_bank,
          st.d_expert_to_slot.get(),
          st.resident_slots,
          p,
          eff_tail_sub_thr,
          eff_tail_sub_rank,
          eff_min_weight,
          st.skip_tail_cold_only,
          st.d_decode_ws.get(),
          st.d_decode_pinned,
          output->MutableDataRaw());
    }

    // 2. Prefill path (num_rows > 16): resident Tensor-Core launch + double-buffered 32-expert wave streaming
    // Use single-tile 8192-row prefill chunks so an 8K prefill streams cold experts once per layer instead of 4 times.
    constexpr int64_t kMaxPrefillChunkRows = 8192;
    const int64_t max_chunk = std::min<int64_t>(num_rows, kMaxPrefillChunkRows);
    MoEPackedParams max_params{};
    max_params.num_rows = static_cast<int>(max_chunk);
    max_params.num_experts = static_cast<int>(num_experts);
    max_params.hidden_size = static_cast<int>(hidden_size);
    max_params.inter_size = static_cast<int>(inter_size);
    max_params.k = static_cast<int>(k_);
    max_params.normalize_routing_weights = (normalize_routing_weights_ != 0);
    max_params.gate_type = static_cast<int>(gate_type_);
    max_params.up_type = static_cast<int>(up_type_);
    max_params.down_type = static_cast<int>(down_type_);
    max_params.gate_row_bytes = exp_gate_rb;
    max_params.up_row_bytes = exp_up_rb;
    max_params.down_row_bytes = exp_down_rb;

    const size_t ws_bytes = GetMoEPackedWorkspaceBytes(max_params, sizeof(T));
    auto workspace = GetScratchBuffer<uint8_t>(ws_bytes, GetComputeStream(context));

    auto& wave_staging = MoEPackedSharedPrefillWaveStaging::Instance();
    std::lock_guard<std::mutex> wave_lk(wave_staging.mu);
    const int kWaveExperts = MoEPackedSharedPrefillWaveStaging::kWaveExperts;
    const size_t wave_bank_bytes = static_cast<size_t>(kWaveExperts) * st.slot_bytes;
    ORT_RETURN_IF_ERROR(wave_staging.EnsureCapacity(wave_bank_bytes, static_cast<size_t>(max_chunk * k_)));

    const uint8_t* in_bytes = static_cast<const uint8_t*>(input->DataRaw());
    const uint8_t* rp_bytes = static_cast<const uint8_t*>(router_probs->DataRaw());
    uint8_t* out_bytes = static_cast<uint8_t*>(output->MutableDataRaw());
    cudaStream_t stream = Stream(context);

    for (int64_t r0 = 0; r0 < num_rows; r0 += kMaxPrefillChunkRows) {
      const int chunk_rows = static_cast<int>(std::min<int64_t>(kMaxPrefillChunkRows, num_rows - r0));
      const int total_routes = chunk_rows * static_cast<int>(k_);
      const bool use_prefill_tiles = is_bf16 && (total_routes > 128);
      const int tile_cap = use_prefill_tiles ? 32 : 8;
      const int max_tiles_alloc = (total_routes + tile_cap - 1) / tile_cap + static_cast<int>(num_experts);

      MoEPackedParams chunk_params = max_params;
      chunk_params.num_rows = chunk_rows;

      uint8_t* ws_ptr = workspace.get();
      float* d_topk_scales = reinterpret_cast<float*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * sizeof(float));
      int* d_topk_indices = reinterpret_cast<int*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * sizeof(int));
      int* d_sorted_routes = reinterpret_cast<int*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * sizeof(int));
      MoEPackedTile* d_tiles = reinterpret_cast<MoEPackedTile*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(max_tiles_alloc) * sizeof(MoEPackedTile));
      int* d_num_tiles = reinterpret_cast<int*>(ws_ptr);
      ws_ptr += AlignUp256(sizeof(int));
      void* d_fc1_out = ws_ptr;
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * inter_size * sizeof(T));
      void* d_fc2_out = ws_ptr;

      const void* chunk_in = in_bytes + static_cast<size_t>(r0 * hidden_size) * sizeof(T);
      const void* chunk_rp = rp_bytes + static_cast<size_t>(r0 * num_experts) * sizeof(T);
      void* chunk_out = out_bytes + static_cast<size_t>(r0 * hidden_size) * sizeof(T);

      // 2a. Softmax + TopK on GPU and copy indices/scales to pinned host staging
      ORT_RETURN_IF_ERROR(LaunchMoEPackedSoftmaxTopKCuda(
          stream, is_bf16, chunk_rp, d_topk_scales, d_topk_indices, chunk_rows,
          static_cast<int>(num_experts), static_cast<int>(k_), chunk_params.normalize_routing_weights));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
          wave_staging.pinned_topk_idx, d_topk_indices, static_cast<size_t>(total_routes) * sizeof(int32_t),
          cudaMemcpyDeviceToHost, stream)));
      if (eff_min_weight > 0.0f) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            wave_staging.pinned_topk_scales, d_topk_scales, static_cast<size_t>(total_routes) * sizeof(float),
            cudaMemcpyDeviceToHost, stream)));
      }
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamSynchronize(stream)));

      // 2b. Bucket routes by expert on CPU
      std::vector<std::vector<int>> exp_routes(static_cast<size_t>(num_experts));
      bool any_pruned = false;
      for (int r = 0; r < total_routes; ++r) {
        const int e = wave_staging.pinned_topk_idx[r];
        if (eff_min_weight > 0.0f && (r % static_cast<int>(k_)) > 0 &&
            wave_staging.pinned_topk_scales[r] < eff_min_weight) {
          wave_staging.pinned_topk_scales[r] = 0.0f;
          any_pruned = true;
          continue;
        }
        if (e >= 0 && e < num_experts) {
          exp_routes[static_cast<size_t>(e)].push_back(r);
        }
      }
      if (any_pruned) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_topk_scales, wave_staging.pinned_topk_scales, static_cast<size_t>(total_routes) * sizeof(float),
            cudaMemcpyHostToDevice, stream)));
      }

      const bool in_warmup = st.InWarmup();
      if (!in_warmup) {
        st.prefill_routes.fetch_add(static_cast<uint64_t>(total_routes), std::memory_order_relaxed);
        st.stream_tiles.fetch_add(1, std::memory_order_relaxed);
        for (int e = 0; e < num_experts; ++e) {
          st.expert_ema[static_cast<size_t>(e)] =
              0.85f * st.expert_ema[static_cast<size_t>(e)] +
              static_cast<float>(exp_routes[static_cast<size_t>(e)].size());
        }
      }

      // 2c. Build & launch GPU-resident expert tiles first
      int res_route_cnt = 0;
      int res_tile_cnt = 0;
      std::vector<int> cold_experts;
      cold_experts.reserve(static_cast<size_t>(num_experts));
      for (int e = 0; e < num_experts; ++e) {
        const auto& rvec = exp_routes[static_cast<size_t>(e)];
        if (rvec.empty()) continue;
        const int slot = st.host_expert_to_slot[static_cast<size_t>(e)];
        if (slot >= 0 && slot < st.resident_slots) {
          const int start_r = res_route_cnt;
          for (int r : rvec) {
            wave_staging.pinned_res_sorted_routes[res_route_cnt++] = r;
          }
          const int c = static_cast<int>(rvec.size());
          const int n_t = (c + tile_cap - 1) / tile_cap;
          for (int t = 0; t < n_t; ++t) {
            MoEPackedTile tile{};
            tile.expert = slot;
            tile.start_route = start_r + t * tile_cap;
            tile.count = std::min(tile_cap, c - t * tile_cap);
            tile.pad = 0;
            wave_staging.pinned_res_tiles[res_tile_cnt++] = tile;
          }
        } else {
          cold_experts.push_back(e);
        }
      }
      if (!in_warmup) {
        st.prefill_hit_routes.fetch_add(static_cast<uint64_t>(res_route_cnt), std::memory_order_relaxed);
        // Sort cold experts by descending EMA score so Wave 0 holds the hottest cold experts for promotion
        std::sort(cold_experts.begin(), cold_experts.end(), [&](int a, int b) {
          return st.expert_ema[static_cast<size_t>(a)] > st.expert_ema[static_cast<size_t>(b)];
        });
#if defined(_WIN32)
        if (r0 == 0 && !cold_experts.empty()) {
          MEMORYSTATUSEX mem_stat{};
          mem_stat.dwLength = sizeof(mem_stat);
          if (GlobalMemoryStatusEx(&mem_stat) && mem_stat.ullAvailPhys >= (2ULL << 30)) {
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            ranges.reserve(cold_experts.size() * 3);
            for (int ce : cold_experts) {
              ranges.push_back({const_cast<uint8_t*>(st.host_gate_w + static_cast<size_t>(ce) * st.gate_exp_bytes), st.gate_exp_bytes});
              ranges.push_back({const_cast<uint8_t*>(st.host_up_w + static_cast<size_t>(ce) * st.up_exp_bytes), st.up_exp_bytes});
              ranges.push_back({const_cast<uint8_t*>(st.host_down_w + static_cast<size_t>(ce) * st.down_exp_bytes), st.down_exp_bytes});
            }
            static_cast<void>(PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0));
          }
        }
#endif
      }

      if (res_tile_cnt > 0) {
        *wave_staging.pinned_res_num_tiles = res_tile_cnt;
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_sorted_routes, wave_staging.pinned_res_sorted_routes, static_cast<size_t>(res_route_cnt) * sizeof(int),
            cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_tiles, wave_staging.pinned_res_tiles, static_cast<size_t>(res_tile_cnt) * sizeof(MoEPackedTile),
            cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_num_tiles, wave_staging.pinned_res_num_tiles, sizeof(int), cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(LaunchMoEPackedGroupTilesCuda(
            stream, is_bf16, use_prefill_tiles, chunk_in,
            st.gpu_gate_bank, st.gpu_up_bank, st.gpu_down_bank,
            d_sorted_routes, d_tiles, d_num_tiles, res_tile_cnt, chunk_params, d_fc1_out, d_fc2_out));
      }

      // 2d. Double-buffered streaming waves for cold experts (CPU parallel staging + H2D copy_stream + GPU Tensor-Core compute)
      // During dummy-token Warmup(), run at most 1 wave to warm wave buffers and Tensor-Core kernels without faulting 40 GiB.
      const int num_cold = in_warmup ? std::min<int>(static_cast<int>(cold_experts.size()), kWaveExperts)
                                     : static_cast<int>(cold_experts.size());
      const int num_waves = (num_cold + kWaveExperts - 1) / kWaveExperts;
      if (!in_warmup) {
        st.stream_experts.fetch_add(static_cast<uint64_t>(num_cold), std::memory_order_relaxed);
        st.stream_waves.fetch_add(static_cast<uint64_t>(num_waves), std::memory_order_relaxed);
      }

      auto stage_and_enqueue_wave = [&](int w_idx) -> Status {
        const int b = w_idx & 1;
        const int e_begin = w_idx * kWaveExperts;
        const int w_cnt = std::min(kWaveExperts, num_cold - e_begin);
        const size_t w_gate_bytes = static_cast<size_t>(w_cnt) * st.gate_exp_bytes;
        const size_t w_up_bytes = static_cast<size_t>(w_cnt) * st.up_exp_bytes;
        const size_t w_down_bytes = static_cast<size_t>(w_cnt) * st.down_exp_bytes;

        if (wave_staging.bank_in_flight[b]) {
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(wave_staging.compute_done[b])));
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(wave_staging.h2d_done[b])));
          wave_staging.bank_in_flight[b] = false;
        }
        uint8_t* p_gate = wave_staging.pinned_bank[b];
        uint8_t* p_up = p_gate + w_gate_bytes;
        uint8_t* p_down = p_up + w_up_bytes;

        MoEPackedParallelPool::Instance().ParallelFor(w_cnt, [&](int local_s) {
          const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
          std::memcpy(p_gate + static_cast<size_t>(local_s) * st.gate_exp_bytes,
                      st.host_gate_w + static_cast<size_t>(exp) * st.gate_exp_bytes, st.gate_exp_bytes);
          std::memcpy(p_up + static_cast<size_t>(local_s) * st.up_exp_bytes,
                      st.host_up_w + static_cast<size_t>(exp) * st.up_exp_bytes, st.up_exp_bytes);
          std::memcpy(p_down + static_cast<size_t>(local_s) * st.down_exp_bytes,
                      st.host_down_w + static_cast<size_t>(exp) * st.down_exp_bytes, st.down_exp_bytes);
        });

        int w_route_cnt = 0;
        int w_tile_cnt = 0;
        for (int local_s = 0; local_s < w_cnt; ++local_s) {
          const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
          const auto& rvec = exp_routes[static_cast<size_t>(exp)];
          const int start_r = w_route_cnt;
          for (int r : rvec) {
            wave_staging.pinned_sorted_routes[b][w_route_cnt++] = r;
          }
          const int c = static_cast<int>(rvec.size());
          const int n_t = (c + tile_cap - 1) / tile_cap;
          for (int t = 0; t < n_t; ++t) {
            MoEPackedTile tile{};
            tile.expert = local_s;
            tile.start_route = start_r + t * tile_cap;
            tile.count = std::min(tile_cap, c - t * tile_cap);
            tile.pad = 0;
            wave_staging.pinned_tiles[b][w_tile_cnt++] = tile;
          }
        }
        *wave_staging.pinned_num_tiles[b] = w_tile_cnt;

        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            wave_staging.gpu_bank[b], wave_staging.pinned_bank[b],
            w_gate_bytes + w_up_bytes + w_down_bytes,
            cudaMemcpyHostToDevice, wave_staging.copy_stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(wave_staging.h2d_done[b], wave_staging.copy_stream)));
        wave_staging.bank_in_flight[b] = true;
        return Status::OK();
      };

      if (num_waves > 0) {
        ORT_RETURN_IF_ERROR(stage_and_enqueue_wave(0));
      }
      for (int w_idx = 0; w_idx < num_waves; ++w_idx) {
        const int b = w_idx & 1;
        const int e_begin = w_idx * kWaveExperts;
        const int w_cnt = std::min(kWaveExperts, num_cold - e_begin);
        const size_t w_gate_bytes = static_cast<size_t>(w_cnt) * st.gate_exp_bytes;
        const size_t w_up_bytes = static_cast<size_t>(w_cnt) * st.up_exp_bytes;

        int w_route_cnt = 0;
        for (int local_s = 0; local_s < w_cnt; ++local_s) {
          const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
          w_route_cnt += static_cast<int>(exp_routes[static_cast<size_t>(exp)].size());
        }
        const int w_tile_cnt = *wave_staging.pinned_num_tiles[b];
        if (!in_warmup) {
          st.stream_routes.fetch_add(static_cast<uint64_t>(w_route_cnt), std::memory_order_relaxed);
        }

        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamWaitEvent(stream, wave_staging.h2d_done[b], 0)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_sorted_routes, wave_staging.pinned_sorted_routes[b], static_cast<size_t>(w_route_cnt) * sizeof(int),
            cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_tiles, wave_staging.pinned_tiles[b], static_cast<size_t>(w_tile_cnt) * sizeof(MoEPackedTile),
            cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_num_tiles, wave_staging.pinned_num_tiles[b], sizeof(int), cudaMemcpyHostToDevice, stream)));

        const uint8_t* g_bank = wave_staging.gpu_bank[b];
        const uint8_t* u_bank = g_bank + w_gate_bytes;
        const uint8_t* d_bank = u_bank + w_up_bytes;
        ORT_RETURN_IF_ERROR(LaunchMoEPackedGroupTilesCuda(
            stream, is_bf16, use_prefill_tiles, chunk_in,
            g_bank, u_bank, d_bank,
            d_sorted_routes, d_tiles, d_num_tiles, w_tile_cnt, chunk_params, d_fc1_out, d_fc2_out));

        // Adaptive D2D promotion of hottest cold experts already in gpu_bank[b] into coldest unpinned resident slots
        if (!in_warmup && st.is_adaptive && st.resident_slots > 32 && w_idx == 0) {
          const int min_evict_slot = std::max(16, st.resident_slots - 32);
          int promoted_in_wave = 0;
          for (int local_s = 0; local_s < w_cnt && promoted_in_wave < 8; ++local_s) {
            const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
            const float cand_score = st.expert_ema[static_cast<size_t>(exp)];
            if (cand_score < 2.0f) continue;
            int best_victim_slot = -1;
            float lowest_score = cand_score - 0.5f;
            for (int vs = min_evict_slot; vs < st.resident_slots; ++vs) {
              const int v_exp = st.slot_to_expert[static_cast<size_t>(vs)];
              if (v_exp >= 0 && exp_routes[static_cast<size_t>(v_exp)].empty() &&
                  st.expert_ema[static_cast<size_t>(v_exp)] < lowest_score) {
                lowest_score = st.expert_ema[static_cast<size_t>(v_exp)];
                best_victim_slot = vs;
              }
            }
            if (best_victim_slot >= 0) {
              const int v_exp = st.slot_to_expert[static_cast<size_t>(best_victim_slot)];
              st.host_expert_to_slot[static_cast<size_t>(v_exp)] = -1;
              st.host_expert_to_slot[static_cast<size_t>(exp)] = best_victim_slot;
              st.slot_to_expert[static_cast<size_t>(best_victim_slot)] = exp;
              ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                  st.gpu_gate_bank + static_cast<size_t>(best_victim_slot) * st.gate_exp_bytes,
                  g_bank + static_cast<size_t>(local_s) * st.gate_exp_bytes,
                  st.gate_exp_bytes, cudaMemcpyDeviceToDevice, stream)));
              ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                  st.gpu_up_bank + static_cast<size_t>(best_victim_slot) * st.up_exp_bytes,
                  u_bank + static_cast<size_t>(local_s) * st.up_exp_bytes,
                  st.up_exp_bytes, cudaMemcpyDeviceToDevice, stream)));
              ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                  st.gpu_down_bank + static_cast<size_t>(best_victim_slot) * st.down_exp_bytes,
                  d_bank + static_cast<size_t>(local_s) * st.down_exp_bytes,
                  st.down_exp_bytes, cudaMemcpyDeviceToDevice, stream)));
              ++promoted_in_wave;
              st.total_promotions.fetch_add(1, std::memory_order_relaxed);
            }
          }
          if (promoted_in_wave > 0) {
            ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                st.d_expert_to_slot.get(), st.host_expert_to_slot.data(),
                static_cast<size_t>(st.num_experts) * sizeof(int32_t), cudaMemcpyHostToDevice, stream)));
          }
        }

        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(wave_staging.compute_done[b], stream)));
        // Stage and enqueue Wave w_idx + 1 on CPU + copy_stream IN PARALLEL while GPU executes Wave w_idx on stream!
        if (w_idx + 1 < num_waves) {
          ORT_RETURN_IF_ERROR(stage_and_enqueue_wave(w_idx + 1));
        }
      }

      // 2e. Finalize combine across all routes (resident + streamed waves)
      ORT_RETURN_IF_ERROR(LaunchMoEPackedFinalizeOnlyCuda(
          stream, is_bf16, d_fc2_out, d_topk_scales, chunk_rows,
          static_cast<int>(hidden_size), static_cast<int>(k_), chunk_out));
    }
    return Status::OK();
  }

  // Process in chunks of at most 2048 rows so workspace stays compact during large prefills.
  const int64_t max_chunk = std::min<int64_t>(num_rows, kMoEPackedMaxChunkRows);
  const int64_t max_routes = max_chunk * k_;
  const int64_t max_tiles = max_routes <= 128 ? max_routes : (max_routes + 7) / 8 + num_experts;
  ORT_RETURN_IF_NOT((hidden_size + 7) / 8 <= GetDeviceProp().maxGridSize[0] &&
                        (inter_size + 7) / 8 <= GetDeviceProp().maxGridSize[0] &&
                        max_tiles <= GetDeviceProp().maxGridSize[1] &&
                        max_chunk <= GetDeviceProp().maxGridSize[1],
                    "MoEPacked dimensions exceed the CUDA grid limits");

  MoEPackedParams max_params{};
  max_params.num_rows = static_cast<int>(max_chunk);
  max_params.num_experts = static_cast<int>(num_experts);
  max_params.hidden_size = static_cast<int>(hidden_size);
  max_params.inter_size = static_cast<int>(inter_size);
  max_params.k = static_cast<int>(k_);
  max_params.normalize_routing_weights = (normalize_routing_weights_ != 0);
  max_params.bf16_tc_supported = bf16_tc_supported_;
  max_params.gate_type = static_cast<int>(gate_type_);
  max_params.up_type = static_cast<int>(up_type_);
  max_params.down_type = static_cast<int>(down_type_);
  max_params.gate_row_bytes = exp_gate_rb;
  max_params.up_row_bytes = exp_up_rb;
  max_params.down_row_bytes = exp_down_rb;

  size_t ws_bytes = 0;
  ORT_RETURN_IF_ERROR(GetMoEPackedWorkspaceBytes(max_params, sizeof(T), ws_bytes));
  Tensor* output = context->Output(0, in_shape);
  if (num_rows == 0) return Status::OK();
  auto workspace = GetScratchBuffer<uint8_t>(ws_bytes, GetComputeStream(context));

  const uint8_t* in_bytes = static_cast<const uint8_t*>(input->DataRaw());
  const uint8_t* rp_bytes = static_cast<const uint8_t*>(router_probs->DataRaw());
  uint8_t* out_bytes = static_cast<uint8_t*>(output->MutableDataRaw());

  for (int64_t r0 = 0; r0 < num_rows;) {
    const int64_t chunk_rows = std::min<int64_t>(kMoEPackedMaxChunkRows, num_rows - r0);
    MoEPackedParams chunk_params = max_params;
    chunk_params.num_rows = static_cast<int>(chunk_rows);
    const size_t input_offset = SafeInt<size_t>(r0) * hidden_size * sizeof(T);
    const size_t router_offset = SafeInt<size_t>(r0) * num_experts * sizeof(T);
    ORT_RETURN_IF_ERROR(LaunchMoEPackedCuda(
        Stream(context),
        is_bf16,
        in_bytes + input_offset,
        rp_bytes + router_offset,
        gate_w->Data<uint8_t>(),
        up_w->Data<uint8_t>(),
        down_w->Data<uint8_t>(),
        chunk_params,
        workspace.get(),
        out_bytes + input_offset));
    r0 += chunk_rows;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
