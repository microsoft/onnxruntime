// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright NVIDIA/apex
// This file is adapted from NVIDIA/apex, commit 0c7d8e3fa9a095a1641a2290877436d0314b69c6

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template<typename x_t, typename y_t, typename out_t>
struct AxpbyFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    float a,
    float b,
    int arg_to_check)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx*chunk_size;

    y_t* y = (y_t*)tl.addresses[1][tensor_loc];
    y += chunk_idx*chunk_size;

    out_t* out = (out_t*)tl.addresses[2][tensor_loc];
    out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    bool finite = true;
    x_t r_x[ILP];
    y_t r_y[ILP];
    out_t r_out[ILP];

    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(out))
    {
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_x, x, 0 , i_start);
        load_store(r_y, y, 0 , i_start);
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_out[ii] = a*static_cast<float>(r_x[ii]) + b*static_cast<float>(r_y[ii]);
          if(arg_to_check == -1)
            finite = finite && (isfinite(r_x[ii]) && isfinite(r_y[ii]));
          if(arg_to_check == 0)
            finite = finite && isfinite(r_x[ii]);
          if(arg_to_check == 1)
            finite = finite && isfinite(r_y[ii]);
        }
        // store
        load_store(out, r_out, i_start , 0);
      }
    }
    else
    {
      // Non-divergent exit condition for __syncthreads, not necessary here
      for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x*ILP)
      {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_x[ii] = 0;
          r_y[ii] = 0;
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_x[ii] = x[i];
            r_y[ii] = y[i];
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_out[ii] = a*static_cast<float>(r_x[ii]) + b*static_cast<float>(r_y[ii]);
          if(arg_to_check == -1)
            finite = finite && (isfinite(r_x[ii]) && isfinite(r_y[ii]));
          if(arg_to_check == 0)
            finite = finite && isfinite(r_x[ii]);
          if(arg_to_check == 1)
            finite = finite && isfinite(r_y[ii]);
        }
        // see note in multi_tensor_scale_kernel.cu
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
            out[i] = r_out[ii];
        }
      }
    }
    if(!finite)
      *noop_gmem = 1; // Blindly fire off a write.  These will race but that's ok.
  }
};

void multi_tensor_axpby_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>>& tensor_lists,
  float a,
  float b,
  int arg_to_check)
{
  using namespace at;
  // The output (downscaled) type is always float.
  // If build times suffer, think about where to put this dispatch,
  // and what logic should be moved out of multi_tensor_apply.

  DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "multi_tensor_axpby_cuda",
    DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 1, "multi_tensor_axpby_cuda",
      DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[2][0].scalar_type(), 2, "multi_tensor_axpby_cuda",
           multi_tensor_apply<3>(
             BLOCK_SIZE,
             chunk_size,
             noop_flag,
             tensor_lists,
             AxpbyFunctor<scalar_t_0, scalar_t_1, scalar_t_2>(),
             a,
             b,
             arg_to_check); )))

  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
