
#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "contrib_ops/cpu/vec/vec_base.h"
#include "contrib_ops/cpu/vec/functional_base.h"

namespace onnxruntime {

namespace {

// 1) out = exp(a - val)
// 2) val = sum(out)
template <typename T1, typename T2>
inline void _exp_reduce_sum_fusion_kernel(
    T1* a,
    const int& size,
    T2* out,
    T1& val) {
  auto vec_size = vec::Vectorized<T1>::size();
  auto vec_max = vec::Vectorized<T1>(val);
  T1 tmp_sum = 0;
  auto vec_tmp_sum = vec::Vectorized<T1>(tmp_sum);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<T1>::loadu(a + i);
    auto tmp1 = tmp0 - vec_max;
    auto tmp2 = tmp1.exp_u20();
    vec_tmp_sum += tmp2;
    _store(out + i, tmp2);
  }
  tmp_sum = vec::vec_reduce_all<T1>(
      [](vec::Vectorized<T1>& x, vec::Vectorized<T1>& y) {
        return x + y;
      },
      vec_tmp_sum);
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 - val;
    auto tmp2 = exp(tmp1);
    tmp_sum += tmp2;
    out[i] = tmp2;
  }
  val = tmp_sum;
}

// 1) out = a * scale
// 2) max = max(out)
template <typename scalar_t>
inline void _mul_reduce_max_fusion_kernel(
    const scalar_t* a,
    const scalar_t& scale,
    const int size,
    scalar_t* out,
    scalar_t& max) {
  auto vec_size = vec::Vectorized<scalar_t>::size();
  auto vec_scale = vec::Vectorized<scalar_t>(scale);
  scalar_t tmp_max = -std::numeric_limits<scalar_t>::infinity();
  auto vec_tmp_max = vec::Vectorized<scalar_t>(tmp_max);
  for (long i = 0; i < vec_size * (size / vec_size); i += vec_size) {
    auto tmp0 = vec::Vectorized<scalar_t>::loadu(a + i);
    auto tmp1 = tmp0 * vec_scale;
    vec_tmp_max = vec::maximum(vec_tmp_max, tmp1);
    _store(out + i, tmp1);
  }
  for (long i = vec_size * (size / vec_size); i < size; i++) {
    auto tmp0 = a[i];
    auto tmp1 = tmp0 * scale;
    tmp_max = std::max(tmp_max, tmp1);
    out[i] = tmp1;
  }
  max = std::max(
      tmp_max,
      vec::vec_reduce_all<scalar_t>(
          [](vec::Vectorized<scalar_t>& x, vec::Vectorized<scalar_t>& y) {
            return vec::maximum(x, y);
          },
          vec_tmp_max));
}

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_CHECK(ptr2 == nullptr);
  return ptr;
}

template <typename scalar_t,
          typename std::enable_if_t<vec::is_reduced_floating_point_v<scalar_t>, int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

template <typename scalar_t>
inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }

  // #if defined(__GNUC__)
  // #pragma unroll
  // #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

// void reshape_attn_mask_to_4d(
//     Tensor & attn_mask,
//     int64_t batchSize,
//     int64_t num_head,
//     int64_t qSize,
//     int64_t kvSize) {
//   // Support mask shapes:
//   // 2d: ({q_seq_len, 1}  x {kv_seq_len, 1})
//   // 4d: ({batch, 1} x {num_heads, 1} x {q_seq_len, 1}  x {kv_seq_len, 1})
//   // Guaranteed in check_attn_mask_shape
//   int64_t attn_mask_size_0 = 1;
//   int64_t attn_mask_size_1 = 1;
//   if (attn_mask.Dim() == 4) {
//     if (attn_mask.Size(0) == batchSize) {
//       attn_mask_size_0 = batchSize;
//     }
//     if (attn_mask.Size(1) == num_head) {
//       attn_mask_size_1 = num_head;
//     }
//   }
//   attn_mask = attn_mask
//                 .view({attn_mask_size_0, attn_mask_size_1, attn_mask.Size(-2), attn_mask.Size(-1)})
//                 .expand({attn_mask_size_0, attn_mask_size_1, qSize, kvSize});
// }

template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention(
    Tensor& output,          // batch x q_seq_len  x num_heads  x head_size
    Tensor& logsumexp,       // batch x q_seq_len  x num_heads
    const Tensor& query,     // batch x q_seq_len  x num_heads  x head_size or
                                    // batch x num_heads  x q_seq_len  x head_size (when is_q_bnsh is True)
    const Tensor& key,       // batch x kv_seq_len x num_heads  x head_size or
                                    // batch x num_heads  x kv_seq_len x head_size (when is_kv_bnsh is True)
    const Tensor& value,     // batch x kv_seq_len x num_heads  x head_size or
                                    // batch x num_heads  x kv_seq_len x head_size (when is_kv_bnsh is True)
    bool is_causal,
    const Tensor* attn_mask, // batch x num_heads q_seq_len x kv_seq_len, optional
    double scale,
    concurrency::ThreadPool* thread_pool,
    AllocatorPtr allocator,
    bool is_q_bnsh,
    bool is_kv_bnsh) {

  constexpr bool is_reduced_type = vec::is_reduced_floating_point_v<scalar_t>;
  static_assert(!is_reduced_type);
  using accum_t = scalar_t; // Need update this for reduced type.

  using Vec = vec::Vectorized<accum_t>;
  accum_t scaling_factor = scale > 0
                               ? static_cast<accum_t>(scale)
                               : static_cast<accum_t>(1.0f / sqrtf(static_cast<float>(query.Size(-1))));

  ORT_ENFORCE((query.Size(3) == value.Size(3)) && (key.Size(3) == value.Size(3)),
        "Q/K/V should have the same head size");

  int64_t batchSize = query.Size(0);
  int64_t qSize = query.Size(is_q_bnsh ? 2 : 1);
  int64_t kvSize = value.Size(is_kv_bnsh ? 2 : 1);
  int64_t num_head = query.Size(is_q_bnsh ? 1 : 2);
  int64_t headSize = query.Size(3);

  // attention mask is optional
  bool has_attn_mask = attn_mask != nullptr && attn_mask->NumberOfElements() > 0;
  // if (has_attn_mask) {
  //   // if (is_reduced_type) {
  //   //   attn_mask = attn_mask.to(onnxruntime::kFloat);
  //   // }
  //   reshape_attn_mask_to_4d(attn_mask.value(), batchSize, num_head, qSize, kvSize);
  // }

  // Strides
  int64_t qStrideB = query.Stride(0);
  int64_t qStrideM = query.Stride(is_q_bnsh ? 2 : 1);
  int64_t qStrideH = query.Stride(is_q_bnsh ? 1 : 2);
  int64_t kStrideB = key.Stride(0);
  int64_t kStrideN = key.Stride(is_kv_bnsh ? 2 : 1);
  int64_t kStrideH = key.Stride(is_kv_bnsh ? 1 : 2);
  int64_t vStrideB = value.Stride(0);
  int64_t vStrideN = value.Stride(is_kv_bnsh ? 2 : 1);
  int64_t vStrideH = value.Stride(is_kv_bnsh ? 1 : 2);
  int64_t oStrideB = output.Stride(0);
  int64_t oStrideM = output.Stride(1);
  int64_t oStrideH = output.Stride(2);
  int64_t lStrideB = logsumexp.Stride(0);
  int64_t lStrideM = logsumexp.Stride(1);
  int64_t lStrideH = logsumexp.Stride(2);
  int64_t mStrideB =
      (has_attn_mask && attn_mask->Size(0) > 1)
      ? attn_mask->Stride(0)
      : 0;
  int64_t mStrideH =
      (has_attn_mask && attn_mask->Size(1) > 1)
      ? attn_mask->Stride(1)
      : 0;
  int64_t mStrideM =
      has_attn_mask ? attn_mask->Stride(2) : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  //int64_t num_thread = concurrency::ThreadPool::DegreeOfParallelism(thread_pool);

  // const auto dtype = query.scalar_type();
  // const auto accumulate_dtype = toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  // Tensor buf = at::empty({num_thread, size_per_thread}, accum_t);
  // Tensor buf_reduced = at::empty({num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0}, query.options());

  // Data ptrs
  const scalar_t* q_data = query.Data<scalar_t>();
  const scalar_t* k_data = key.Data<scalar_t>();
  const scalar_t* v_data = value.Data<scalar_t>();
  const accum_t* mask_data = has_attn_mask ? attn_mask->Data<accum_t>() : nullptr;
  scalar_t* out_data = output.MutableData<scalar_t>();
  accum_t* lse_data = logsumexp.MutableData<accum_t>();
  // accum_t* buf_data = buf.MutableData<accum_t>();
  // scalar_t* buf_reduced_data = is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  double cost_per_slice = 1.0;
  concurrency::ThreadPool::TryParallelFor(
    thread_pool, batchSize * num_head * qSlice, cost_per_slice, [&](ptrdiff_t begin, ptrdiff_t end) {
    int64_t i = 0, j = 0, k = 0;
    vec::data_index_init(begin, i, batchSize, j, num_head, k, qSlice);

    // We cannot get current thread ID from thread pool so we have to allocate in each thread.
    // int thread_id = thread_pool->CurrentThreadId();
    // accum_t* buf_ptr = buf_data + thread_id * size_per_thread;
    void* buffer = allocator->Alloc(sizeof(accum_t) * size_per_thread);
    BufferUniquePtr thread_buffer(buffer, BufferDeleter(std::move(allocator)));

    accum_t* qk_data = reinterpret_cast<accum_t*>(buffer);
    accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
    accum_t* qk_sum_data = qk_max_data + qSplitSize;
    accum_t* dst_data = qk_sum_data + qSplitSize;
    // scalar_t* qk_reduced_data = is_reduced_type ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize : nullptr;

    for (int64_t z = begin; z < end; z++) {
      int64_t m = k * qSplitSize;
      int64_t qBlockSize = std::min(qSplitSize, qSize - m);
      // Initialize max and sum
      fill_stub(qk_max_data, -std::numeric_limits<accum_t>::infinity(), qBlockSize);
      fill_stub(qk_sum_data, static_cast<accum_t>(0), qBlockSize);
      int64_t num_keys = is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
      for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
        int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
        // Calculate scale * q @ k.T

        // // Compute Q*K' + AttentionMask
        // //                     original                 transposed             each iteration
        // // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
        // // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
        // // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
        // math::Gemm<T, ThreadPool>(CblasNoTrans,                  // transA
        //                           CblasTrans,                    // transB
        //                           sequence_length,               // m
        //                           total_sequence_length,         // n
        //                           head_size,                     // k
        //                           alpha,                         // alpha
        //                           Q + q_input_chunk_length * i,  // A
        //                           k,                             // B
        //                           mask_data != nullptr ? 1.0f : 0.0f, // beta
        //                           output,                          // C
        //                           nullptr);

        // cpublas::gemm(
        //     TransposeType::Transpose,     // transa
        //     TransposeType::NoTranspose,   // transb
        //     kvBlockSize,                  // m
        //     qBlockSize,                   // n
        //     headSize,                     // k
        //     static_cast<accum_t>(1),      // alpha
        //     k_data + i * kStrideB + j * kStrideH + n * kStrideN, // a
        //     kStrideN,                                            // lda  (stride of a)
        //     q_data + i * qStrideB + j * qStrideH + m * qStrideM, // b
        //     qStrideM,                                            // ldb  (stride of b)
        //     static_cast<accum_t>(0),                             // beta
        //     qk_data,                                             // c
        //     kvBlockSize);                                        // ldc  (stride of c)


        //                     original                 transposed             each iteration
        // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
        // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
        // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
        math::GemmEx<scalar_t, concurrency::ThreadPool>(
            CblasNoTrans,
            CblasTrans,
            qBlockSize,
            kvBlockSize,
            headSize,
            static_cast<accum_t>(1),
            q_data + i * qStrideB + j * qStrideH + m * qStrideM,
            static_cast<int>(qStrideM),
            k_data + i * kStrideB + j * kStrideH + n * kStrideN,
            static_cast<int>(kStrideN),
            static_cast<accum_t>(0),
            qk_data,
            static_cast<int>(kvBlockSize),
            nullptr);  // thread pool

        // Apply causal mask, fill unused with -inf
        if (is_causal && num_keys - n <= kvSplitSize) {
          for (int64_t row =0; row < qBlockSize; row++) {
            int64_t last_col = m + row - n;
            accum_t* row_ptr = qk_data + row * kvBlockSize;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<accum_t>::infinity(), kvBlockSize - last_col - 1);
          }
        }

        // Update attention weights with attention mask
        // And apply scaling factor
        // qk <- qk * scaling + attn_mask
        if (has_attn_mask) {
          for (int64_t row = 0; row < qBlockSize; ++row) {
            onnxruntime::vec::map2<accum_t>(
                [scaling_factor](Vec x, Vec y) {
                  return x * Vec(scaling_factor) + y;
                },
                qk_data + row * kvBlockSize,
                qk_data + row * kvBlockSize,
                mask_data + i * mStrideB + j * mStrideH + (m + row) * mStrideM + n,
                kvBlockSize);
          }
        }

        // Update coefficients with Softmax
        accum_t tmp_max = 0, tmp_sum = 0, exp_tmp = 0;
        for (int64_t row = 0; row < qBlockSize; ++row) {
          if (has_attn_mask) {
            // max per row
            tmp_max = onnxruntime::vec::reduce_all<accum_t>(
                [](Vec& x, Vec& y) { return onnxruntime::vec::maximum(x, y); },
                qk_data + row * kvBlockSize,
                kvBlockSize);
          } else {
            // apply scaling factor and max per row in fusion
            _mul_reduce_max_fusion_kernel(
                qk_data + row * kvBlockSize,
                scaling_factor,
                static_cast<int>(kvBlockSize),
                qk_data + row * kvBlockSize,
                tmp_max);
          }
          tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
          // qk <- exp(qk - max) and sum per row
          tmp_sum = tmp_max;
          _exp_reduce_sum_fusion_kernel(
              qk_data + row * kvBlockSize,
              static_cast<int>(kvBlockSize),
              qk_data + row * kvBlockSize, //conditional_data_ptr(qk_data, qk_reduced_data) + row * kvBlockSize,
              tmp_sum);
          // exp_tmp <- exp(max[row] - max)
          exp_tmp = std::exp(qk_max_data[row] - tmp_max);
          // sum[row] <- sum + exp_tmp * sum[row]
          qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
          // max[row] <- max
          qk_max_data[row] = tmp_max;
          // dst <- dst * exp_tmp
          if (n > 0) {
            vec::map<accum_t>(
              [exp_tmp](Vec x) { return x * Vec(exp_tmp); },
              dst_data + row * headSize, dst_data + row * headSize, headSize);
          }
        }
        // Calculate Softmax(q @ k.T) @ v
        // cpublas::gemm(
        //     TransposeType::NoTranspose,
        //     TransposeType::NoTranspose,
        //     headSize,
        //     qBlockSize,
        //     kvBlockSize,
        //     static_cast<accum_t>(1),
        //     v_data + i * vStrideB + j * vStrideH + n * vStrideN,
        //     vStrideN,
        //     qk_data, // conditional_data_ptr(qk_data, qk_reduced_data),
        //     kvBlockSize,
        //     n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
        //     dst_data,
        //     headSize);

        //  math::MatMul<T>(sequence_length, v_head_size, total_sequence_length,
        //                 attention_probs + attention_probs_offset,
        //                 v, current_tmp_data, nullptr);
        math::GemmEx<scalar_t, concurrency::ThreadPool>(
            CblasNoTrans,
            CblasNoTrans,
            qBlockSize,
            headSize,
            kvBlockSize,
            static_cast<accum_t>(1),
            qk_data,  // conditional_data_ptr(qk_data, qk_reduced_data),
            static_cast<int>(kvBlockSize),
            v_data + i * vStrideB + j * vStrideH + n * vStrideN,
            static_cast<int>(kStrideN),
            n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
            dst_data,
            static_cast<int>(headSize),
            nullptr);  // thread pool
      }
      // dst <- dst / sum[row]
      // reorder MHA output with strides
      for (int64_t row = 0; row < qBlockSize; ++row) {
        accum_t sum_reciprocal = 1 / qk_sum_data[row];
        vec::map<scalar_t>(
          [sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
          out_data + i * oStrideB + j * oStrideH + m * oStrideM + row * oStrideM,
          dst_data + row * headSize,
          headSize);
      }
      // Store logsumexp for backward
      accum_t* lse_ptr = lse_data + i * lStrideB + j * lStrideH + m * lStrideM;
      for (int64_t row =0; row < qBlockSize; row++) {
        lse_ptr[row * lStrideM] = qk_max_data[row]
            + std::log(qk_sum_data[row]);
      }

      // Move to the next query
      vec::data_index_step(i, batchSize, j, num_head, k, qSlice);
    }
  });
}
} // anonymous namespace

namespace contrib {

// Note that CPU_CAPABILITY is a macro, which will be replaced by "cpu_default", "cpu_avx2", "cpu_avx512" etc to support different CPU capabilities.
namespace CPU_CAPABILITY {

void flash_attention_kernel_impl(
    Tensor& output,
    Tensor& logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,
    const Tensor* attn_mask,
    double scale,
    concurrency::ThreadPool* thread_pool,
    AllocatorPtr allocator,
    bool is_q_bnsh,
    bool is_kv_bnsh) {
  auto q_seq_len = query.Size(2);

  if (query.IsDataType<float>()) {
    if (q_seq_len >= 768) {
      cpu_flash_attention<float, 256, 512>(
          output, logsumexp, query, key, value,
          is_causal, attn_mask, scale, thread_pool, allocator, is_q_bnsh, is_kv_bnsh);
    } else if (q_seq_len >= 192) {
      cpu_flash_attention<float, 64, 512>(
          output, logsumexp, query, key, value,
          is_causal, attn_mask, scale, thread_pool, allocator, is_q_bnsh, is_kv_bnsh);
    } else {
      cpu_flash_attention<float, 32, 512>(
          output, logsumexp, query, key, value,
          is_causal, attn_mask, scale, thread_pool, allocator, is_q_bnsh, is_kv_bnsh);
    }
  }
}
}  // namespace CPU_CAPABILITY
}  // namespace contrib

} // onnxruntime
