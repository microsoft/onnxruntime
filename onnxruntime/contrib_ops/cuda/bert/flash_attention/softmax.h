/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cmath>
#include <limits>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "contrib_ops/cuda/bert/flash_attention/utils.h"

namespace onnxruntime {
namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
constexpr float kInfinity = std::numeric_limits<float>::infinity();

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); mi++) {
    summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0>& dst, Tensor<Engine1, Layout1>& src, Operator& op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op) {
  thread_reduce_<zero_init>(tensor, summary, op);
  quad_allreduce_(summary, summary, op);
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& max) {
  MaxOp<float> max_op;
  reduce_<zero_init>(tensor, max, max_op);
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& sum) {
  SumOp<float> sum_op;
  thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1> const& max, const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    // If max is -inf, then all elements must have been -inf (possibly due to masking).
    // We don't want (-inf - (-inf)) since that would give NaN.
    // If we don't have float around M_LOG2E the multiplication is done in fp64.
    const float max_scaled = max(mi) == -kInfinity ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
      // max * log_2(e)) This allows the compiler to use the ffma
      // instruction instead of fadd and fmul separately.
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum;

  __forceinline__ __device__ Softmax() {};

  template <bool Is_first, bool Check_inf = false, typename Tensor0, typename Tensor1>
  __forceinline__ __device__ void softmax_rescale_o(Tensor0& acc_s, Tensor1& acc_o, float softmax_scale_log2, float sink) {
    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == kNRows);

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      if (Is_first)
        printf("Softmax first iteration:\n");
      else
        printf("Softmax subsequent iteration:\n");

      printf("row_max values:\n");
      for (int mi = 0; mi < size<0>(row_max); ++mi) {
        printf("%f ", (float)row_max(mi));
      }
      printf("\n");

      printf("row_sum values:\n");
      for (int mi = 0; mi < size<0>(row_sum); ++mi) {
        printf("%f ", (float)row_sum(mi));
      }
      printf("\n");

      printf("scores values:\n");
      for (int mi = 0; mi < size<0>(scores); ++mi) {
        for (int ni = 0; ni < size<1>(scores); ++ni) {
          printf("%f ", (float)scores(mi, ni));
        }
        printf("\n");
      }
      printf("\n");
    }

    const bool use_sink = (sink != -kInfinity);
    if (Is_first) {
      flash::template reduce_max</*zero_init=*/true>(scores, row_max);

      if (use_sink) {
#pragma unroll
        for (int mi = 0; mi < size<0>(row_max); ++mi) {
          row_max(mi) = max(row_max(mi), sink);  // Sink value ensures that row_max cannot be -inf
        }
      }

      flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
      flash::reduce_sum</*zero_init=*/true>(scores, row_sum);

      //       if (use_sink) {
      // #pragma unroll
      //         for (int mi = 0; mi < size<0>(row_sum); ++mi) {
      //           float sink_exp = exp2f((sink - row_max(mi)) * softmax_scale_log2);
      //           row_sum(mi) += sink_exp;
      //         }
      //       }
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);

      flash::template reduce_max</*zero_init=*/false>(scores, row_max);

      if (use_sink) {
#pragma unroll
        for (int mi = 0; mi < size<0>(row_max); ++mi) {
          row_max(mi) = max(row_max(mi), sink);
        }
      }

      Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
      static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);

#pragma unroll
      for (int mi = 0; mi < size<0>(row_max); ++mi) {
        // TODO: set Check_inf to false when there is sink.
        float scores_max_cur = !Check_inf
                                   ? row_max(mi)
                                   : (row_max(mi) == -kInfinity ? 0.0f : row_max(mi));
        float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale;

#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
          acc_o_rowcol(mi, ni) *= scores_scale;
        }
      }

      flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
      // We don't do the reduce across threads here since we don't need to use the row_sum.
      // We do that reduce at the end when we need to normalize the softmax.
      flash::reduce_sum</*zero_init=*/false>(scores, row_sum);

      //       if (use_sink) {
      //         const float sink_scaled = sink * softmax_scale_log2;
      // #pragma unroll
      //         for (int mi = 0; mi < size<0>(row_sum); ++mi) {
      //           const float max_scaled = row_max(mi) == -kInfinity ? 0.f : row_max(mi) * softmax_scale_log2;
      //           float sink_exp = exp2f(sink_scaled - max_scaled);
      //           row_sum(mi) += sink_exp;
      //         }
      //       }
    }

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      printf("updated row_max values:\n");
      for (int mi = 0; mi < size<0>(row_max); ++mi) {
        printf("%f ", (float)row_max(mi));
      }
      printf("\n");

      printf("updated row_sum values:\n");
      for (int mi = 0; mi < size<0>(row_sum); ++mi) {
        printf("%f ", (float)row_sum(mi));
      }
      printf("\n");

      printf("updated scores values:\n");
      for (int mi = 0; mi < size<0>(scores); ++mi) {
        for (int ni = 0; ni < size<1>(scores); ++ni) {
          printf("%f ", (float)scores(mi, ni));
        }
        printf("\n");
      }
      printf("\n");
    }
  }

  template <bool Split = false, typename Tensor0>
  __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0& acc_o, float softmax_scale, float sink) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
    TensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);

    const bool use_sink = (sink != -kInfinity);

#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      float sum = row_sum(mi);

      if (use_sink) {
        sum += expf((sink - row_max(mi)) * softmax_scale);
      }

      float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
      lse(mi) = (sum == 0.f || sum != sum)
                    ? (Split ? -kInfinity : kInfinity)
                    : row_max(mi) * softmax_scale + __logf(sum);
      float scale = inv_sum;
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scale;
      }
    }
    return lse;
  };
};

}  // namespace flash
}  // namespace onnxruntime
