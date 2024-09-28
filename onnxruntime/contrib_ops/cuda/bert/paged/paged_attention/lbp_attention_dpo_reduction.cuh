// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "contrib_ops/cuda/bert/paged/atomics.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention_dp.cuh"

namespace onnxruntime::contrib::paged {

__forceinline__ __device__ float2
single_pass_flash_acc(const float2& acc_max_sum, const float2& inc_max_sum) {
  float prev_max = acc_max_sum.x;
  float prev_sum = acc_max_sum.y;

  float curr_max = inc_max_sum.x;
  float curr_sum = inc_max_sum.y;

  float new_max = max(prev_max, curr_max);

  float prev_factor = new_max == prev_max ? 1.0f : __expf(prev_max - new_max);
  float curr_factor = new_max == curr_max ? 1.0f : __expf(curr_max - new_max);

  float new_sum = prev_factor * prev_sum + curr_factor * curr_sum;
  return float2{new_max, new_sum};
}

template <int NumThreads, int GroupSize = constant::WarpSize, int Strided = false>
__forceinline__ __device__ float2
flash_acc_global_stats_reduction_cta(float2 global_max_sum) {
  constexpr int NumWarps = NumThreads / constant::WarpSize;

  __shared__ float2 reduction_buffer[NumWarps];

  global_max_sum = warp::reduce<GroupSize, Strided>(global_max_sum, single_pass_flash_acc);
  if (lane_id() == 0) {
    reduction_buffer[warp_id()] = global_max_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    CUTE_UNROLL
    for (int i = 1; i < NumWarps; i++) {
      global_max_sum = single_pass_flash_acc(global_max_sum, reduction_buffer[i]);
    }
    reduction_buffer[0] = global_max_sum;
  }
  __syncthreads();
  global_max_sum = reduction_buffer[0];
  return global_max_sum;
}

template <int NumThreads, int HeadSize, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void
flash_acc_with_global_stats(
    float* acc, float2 global_max_sum,               // accumulative part
    const Tensor1& inc, const Tensor2& inc_max_sum,  // incremental part, indexed as (chunk) and (chunk,dim)
    int task_chunk_start, int task_chunk_end
) {
  float global_max = global_max_sum.x;
  float global_sum = global_max_sum.y;
  float global_sum_inv = 1.0f / global_sum;

  CUTE_UNROLL
  for (int i = threadIdx.x; i < HeadSize; i += NumThreads) {
    float gloabl_acc = acc[i];
    for (int t = task_chunk_start; t < task_chunk_end; t++) {
      float curr_max = inc_max_sum(t).x;
      float curr_sum = inc_max_sum(t).y;
      float curr_factor = __expf(curr_max - global_max);
      gloabl_acc += curr_factor * (curr_sum * global_sum_inv) * inc(t, i);
    }
    acc[i] = gloabl_acc;
  }
}

template <int NumThreads, int HeadSize, typename TIO, typename Config>
__global__ void
lbp_attention_reduction_kernel(
    void* __restrict__ workspace,
    TIO* __restrict__ out,                 // [num_seqs, num_heads, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int num_seqs,
    const int num_heads,
    const int max_context_len
)
#if !defined(PAGED_SPLIT_COMPILATION) || defined(PAGED_ATTENTION_KERNEL_IMPL)
{
  static_assert(!Config::InplaceFlashAcc);
  static_assert(!Config::UseHeadSeq);

  using TI = TIO;
  using TO = TIO;
  using WorkspaceT = DataParallelWorkspace<Config>;

  using FashAccCta = FlashAccCta<NumThreads, HeadSize>;

  auto ws_max_sum = WorkspaceT::max_sum(workspace, num_seqs, num_heads, max_context_len);
  auto ws_out = WorkspaceT::out(workspace, num_seqs, num_heads, HeadSize, max_context_len);

  __shared__ float smem_out[HeadSize];

  int seq_head_idx = blockIdx.x;
  int seq_idx = seq_head_idx / num_heads;
  int head_idx = seq_head_idx % num_heads;
  int task_chunk_start = 0;
  constexpr int TaskChunkSeqLen = Config::TaskChunkSeqLen;
  int task_chunk_end = ceil_div(context_lens[seq_idx], TaskChunkSeqLen);

  CUTE_UNROLL
  for (int i = threadIdx.x; i < HeadSize; i += NumThreads) {
    smem_out[i] = 0.0f;
  }
  // __syncthreads();

  auto max_sum_view = ws_max_sum(_, seq_idx, head_idx);
  float2 global_max_sum;
  init_max_sum(&global_max_sum);

  // compute the global max and global sum of multiple TaskChunks in single pass
  for (int t = task_chunk_start + threadIdx.x; t < task_chunk_end; t += NumThreads) {
    global_max_sum = single_pass_flash_acc(global_max_sum, max_sum_view(t));
  }

  global_max_sum = flash_acc_global_stats_reduction_cta<NumThreads>(global_max_sum);
  flash_acc_with_global_stats<NumThreads, HeadSize>(
      smem_out, global_max_sum,
      ws_out(_, seq_idx, head_idx, _), ws_max_sum(_, seq_idx, head_idx),
      task_chunk_start, task_chunk_end
  );

  auto gO = make_tensor(make_gmem_ptr(out), make_layout(make_shape(num_seqs, num_heads, Int<HeadSize>{}), LayoutRight{}));
  FashAccCta::write_inc(
      &gO(seq_idx, head_idx, _0{}), nullptr,
      smem_out, nullptr
  );
}
#else
    ;
#endif

template <int VecSize, typename TAcc, typename TInc>
__forceinline__ __device__ void
flash_acc_thread(
    TAcc* acc, float2* acc_max_sum,             // accumulative part
    const TInc& inc, const float2* inc_max_sum  // incremental part
) {
  float prev_max = acc_max_sum->x;
  float prev_sum = acc_max_sum->y;

  float curr_max = inc_max_sum->x;
  float curr_sum = inc_max_sum->y;

  float new_max = max(prev_max, curr_max);

  float prev_factor = new_max == prev_max ? 1.0f : __expf(prev_max - new_max);
  float curr_factor = new_max == curr_max ? 1.0f : __expf(curr_max - new_max);

  float new_sum = prev_factor * prev_sum + curr_factor * curr_sum;
  float new_sum_inv = 1.0f / new_sum;

  *acc_max_sum = float2{new_max, new_sum};
  bool load_prev_acc = prev_sum != 0.0f;
  CUTE_UNROLL
  for (int i = 0; i < VecSize; i++) {
    float old_val = load_prev_acc ? type_convert<float>(acc[i]) : 0.0f;
    float new_val = prev_factor * (prev_sum * new_sum_inv) * old_val +
                    curr_factor * (curr_sum * new_sum_inv) * type_convert<float>(inc[i]);
    acc[i] = type_convert<TAcc>(new_val);
  }
}

template <typename TIO, typename Config>
__global__ void
lbp_attention_split_head_reduction_kernel(
    void* __restrict__ workspace,
    TIO* __restrict__ out,                 // [num_seqs, num_heads, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int num_seqs,
    const int num_heads,
    const int head_size,
    const int max_context_len
)
#if !defined(PAGED_SPLIT_COMPILATION) || defined(PAGED_ATTENTION_KERNEL_IMPL)
{
  static_assert(!Config::InplaceFlashAcc);
  static_assert(!Config::UseHeadSeq);

  constexpr int NumThreads = 128;
  constexpr int NumWarps = NumThreads / constant::WarpSize;

  constexpr int CachelineSize = 32;
  constexpr int LdgSize = 16;
  constexpr int VecSize = LdgSize / sizeof(TIO);
  constexpr int NumVecH = CachelineSize / LdgSize;  // number of vectors along HeadSize
  constexpr int NumVecT = NumThreads / NumVecH;     // number of vectors along each TaskChunk
  auto dim_idx = (blockIdx.x * NumVecH + threadIdx.x % NumVecH) * VecSize;

  int seq_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  using WorkspaceT = DataParallelWorkspace<Config>;
  auto ws_max_sum = WorkspaceT::max_sum(workspace, num_seqs, num_heads, max_context_len)(_, seq_idx, head_idx);        // (task_chunk_idx,seq_idx,head_idx)
  auto ws_out = WorkspaceT::out(workspace, num_seqs, num_heads, head_size, max_context_len)(_, seq_idx, head_idx, _);  // (task_chunk_idx,seq_idx,head_idx,dim_idx)
  auto cta_ws_out = local_tile(ws_out, make_shape(max_context_len, Int<VecSize>{}), make_coord(_0{}, dim_idx / VecSize));

  int task_chunk_start = 0;
  constexpr int TaskChunkSeqLen = Config::TaskChunkSeqLen;
  int task_chunk_end = ceil_div(context_lens[seq_idx], TaskChunkSeqLen);
  auto task_chunk_idx = task_chunk_start + threadIdx.x / NumVecH;

  auto acc = make_tensor<float>(Int<VecSize>{});
  float2 acc_max_sum;
  init_max_sum(&acc_max_sum);
  for (int t = task_chunk_idx; t < task_chunk_end; t += NumVecT) {
    auto inc = make_tensor<TIO>(Int<VecSize>{});
    copy(cta_ws_out(t, _), inc);
    float2 inc_max_sum = ws_max_sum(t);
    flash_acc_thread<VecSize>(&acc(_0{}), &acc_max_sum, &inc(_0{}), &inc_max_sum);
  }

  __shared__ float reduction_buffer[cute::max(VecSize * NumVecH, NumWarps)];

  // re-normalize local acc vector with global stats
  float2 global_max_sum = flash_acc_global_stats_reduction_cta<NumThreads, constant::WarpSize / NumVecH, true>(acc_max_sum);
  float curr_max = acc_max_sum.x;
  float curr_sum = acc_max_sum.y;
  float global_max = global_max_sum.x;
  float global_sum = global_max_sum.y;
  float curr_factor = __expf(curr_max - global_max);
  float factor = curr_factor * (curr_sum / global_sum);
  transform(acc, [&](auto v) { return v * factor; });

  // local acc -> global acc
  acc = warp::reduce<constant::WarpSize / NumVecH, true>(acc, [](float a, float b) { return a + b; });
  for (int warp = 0; warp < NumWarps; warp++) {
    __syncthreads();
    if (warp_id() == warp && lane_id() < NumVecH) {
      if (warp == 0) {
        CUTE_UNROLL
        for (int i = 0; i < VecSize; i++) {
          reduction_buffer[lane_id() * VecSize + i] = acc(i);
        }
      } else {
        CUTE_UNROLL
        for (int i = 0; i < VecSize; i++) {
          reduction_buffer[lane_id() * VecSize + i] += acc(i);
        }
      }
    }
  }
  __syncthreads();

  // write global acc
  if (threadIdx.x < NumVecH) {
    auto gO = make_tensor(make_gmem_ptr(out), make_layout(make_shape(num_seqs, num_heads, head_size), LayoutRight{}));
    CUTE_UNROLL
    for (int i = 0; i < VecSize; i++) {
      gO(seq_idx, head_idx, dim_idx + i) = reduction_buffer[threadIdx.x * VecSize + i];
    }
  }
}
#else
    ;
#endif

template <int NumThreads, int HeadSize, typename TIO, typename Config>
inline void
launch_lbp_attention_reduction_kernel(
    stream_t stream,
    dev_props_ptr dev_props,
    void* __restrict__ workspace,
    TIO* __restrict__ out_ptr,                 // [num_seqs, num_heads, head_size]
    const int* __restrict__ context_lens_ptr,  // [num_seqs]
    const int num_seqs,
    const int num_heads,
    const int max_context_len
) {
  constexpr int CachelineSize = 32;
  constexpr int LdgSize = 16;
  constexpr int VecSize = LdgSize / sizeof(TIO);
  constexpr int NumVecH = CachelineSize / LdgSize;  // number of vectors along HeadSize
  constexpr int NumCtaPerHead = HeadSize / (VecSize * NumVecH);
  static_assert(HeadSize % (VecSize * NumVecH) == 0);

  float num_blk_per_sm = float(num_heads * num_seqs) / dev_props->multiProcessorCount;
  // float num_blk_per_sm_split_head = float(num_heads * num_seqs * NumCtaPerHead) / dev_props->multiProcessorCount;

  if (num_blk_per_sm >= 2) {
    lbp_attention_reduction_kernel<NumThreads, HeadSize, TIO, Config>
        <<<num_heads * num_seqs, NumThreads, 0, stream>>>(
            workspace,
            out_ptr,
            context_lens_ptr,
            num_seqs,
            num_heads,
            max_context_len
        );
  } else {
    lbp_attention_split_head_reduction_kernel<TIO, Config>
        <<<dim3(NumCtaPerHead, num_heads, num_seqs), NumThreads, 0, stream>>>(
            workspace,
            out_ptr,
            context_lens_ptr,
            num_seqs,
            num_heads,
            HeadSize,
            max_context_len
        );
  }
}

}  // namespace onnxruntime::contrib::paged
