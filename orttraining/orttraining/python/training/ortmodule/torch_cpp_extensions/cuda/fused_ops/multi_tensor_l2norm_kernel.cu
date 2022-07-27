// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright NVIDIA/apex
// This file is adapted from NVIDIA/apex, commit 3ff1a10f72ec07067c4e44759442329804ac5162

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template <typename T>
__device__ __forceinline__ bool is_aligned(T* p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset) {
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <typename x_t>
struct L2NormFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int* noop_gmem, TensorListMetadata<1>& tl,
                                             float* output, float* output_per_tensor, bool per_tensor,
                                             int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size; i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] += next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            vals[ii] += next * next;
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final)) *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) * max_chunks_per_tensor + chunk_idx] = final;
    }
  }
};

__global__ void cleanup(float* output, float* output_per_tensor, float* ret, float* ret_per_tensor, bool per_tensor,
                        int max_chunks_per_tensor) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < 320) val = output[threadIdx.x];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) *ret = sqrt(final);
  }

  if (per_tensor) {
    float* output_this_tensor = output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    float val = 0;
    for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x) val += output_this_tensor[i];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) ret_per_tensor[blockIdx.x] = sqrt(final);
  }
}

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(int chunk_size, at::Tensor noop_flag,
                                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                                            at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor) max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  } else {
    ret_per_tensor = at::empty({0}, float_options);
  }

  DISPATCH_DOUBLE_FLOAT_AND_HALF(
      tensor_lists[0][0].scalar_type(), 0, "multi_tensor_l2norm_cuda",
      multi_tensor_apply<1>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists, L2NormFunctor<scalar_t_0>(),
                            output.data_ptr<float>(), per_tensor ? output_per_tensor.data_ptr<float>() : nullptr,
                            per_tensor, max_chunks_per_tensor);)

  AT_CUDA_CHECK(cudaGetLastError());
  // AT_CUDA_CHECK(cudaDeviceSynchronize());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  auto ret = at::empty({1}, output.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup<<<per_tensor ? ntensors : 1, 512, 0, stream>>>(
      output.data_ptr<float>(), per_tensor ? output_per_tensor.data_ptr<float>() : nullptr, ret.data_ptr<float>(),
      per_tensor ? ret_per_tensor.data_ptr<float>() : nullptr, per_tensor, max_chunks_per_tensor);

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}
