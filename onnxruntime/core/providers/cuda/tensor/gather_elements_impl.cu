// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_elements_impl.h"
#include "core/providers/cuda/tensor/scatter_elements_impl.h"
#ifdef ENABLE_TRAINING_OPS
#include "orttraining/training_ops/cuda/tensor/gather_elements_grad_impl.h"
#endif

#include "core/providers/cuda/atomic/common.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
#ifdef USE_ROCM
constexpr int kThreadsPerBlock = 256;
#else
constexpr int kThreadsPerBlock = GPU_WARP_SIZE * 4;
#endif
constexpr int kThreadWorkSize = 4;

// General case to compute the input(for Gather)/output(for Scatter) and indices data offset given the thread ID
// using strides and fast_divmods. The offsets are returned in a 2-element TArray.
template <bool IsStridedIndices>
struct OffsetCalculator {
  OffsetCalculator(const int rank, const TArray<int64_t> masked_input_strides, const TArray<fast_divmod> indices_fdms,
                   const TArray<int64_t> indices_strides)
      : rank_(rank), indices_fdms_(indices_fdms) {
    masked_input_strides_.SetSize(rank);
    if (IsStridedIndices) indices_strides_.SetSize(rank);
    for (int dim = 0; dim < rank; ++dim) {
      masked_input_strides_[dim] = static_cast<CUDA_LONG>(masked_input_strides[dim]);
      if (IsStridedIndices) indices_strides_[dim] = static_cast<CUDA_LONG>(indices_strides[dim]);
    }
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    offsets[0] = 0;
    offsets[1] = IsStridedIndices ? 0 : linear_idx;
    CUDA_LONG q, r = linear_idx;
#pragma unroll
    for (int dim = 0; dim < indices_fdms_.Capacity(); ++dim) {
      if (dim == rank_) break;
      indices_fdms_[dim].divmod(r, q, r);
      offsets[0] += masked_input_strides_[dim] * q;
      if (IsStridedIndices) offsets[1] += indices_strides_[dim] * q;
    }
    return offsets;
  }

  int rank_;
  TArray<fast_divmod> indices_fdms_;
  TArray<CUDA_LONG> masked_input_strides_;
  TArray<CUDA_LONG> indices_strides_;
};

// Optimization for 2D case to compute the input(for Gather)/output(for Scatter) and indices data offset
// given the thread ID so we don't need FOR loop for fast_divmod computes.
// The offsets are returned in a 2-element TArray.
template <bool IsOuterAxis, bool IsStridedIndices>
struct OffsetCalculatorFor2D {
  OffsetCalculatorFor2D(const fast_divmod indices_row_size_fdm, const int64_t input_row_size,
                        const TArray<int64_t> indices_strides)
      : indices_row_size_fdm_(indices_row_size_fdm), input_row_size_(static_cast<CUDA_LONG>(input_row_size)) {
    if (IsStridedIndices) {
      indices_strides_.SetSize(2);
      indices_strides_[0] = static_cast<CUDA_LONG>(indices_strides[0]);
      indices_strides_[1] = static_cast<CUDA_LONG>(indices_strides[1]);
    }
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    if (IsStridedIndices) {
      CUDA_LONG q, r = linear_idx;
      indices_row_size_fdm_.divmod(r, q, r);
      offsets[0] = IsOuterAxis ? r : q * input_row_size_;
      offsets[1] = q * indices_strides_[0] + r * indices_strides_[1];
    } else {
      offsets[0] =
          IsOuterAxis ? indices_row_size_fdm_.mod(linear_idx) : indices_row_size_fdm_.div(linear_idx) * input_row_size_;
      offsets[1] = linear_idx;
    }
    return offsets;
  }

  fast_divmod indices_row_size_fdm_;
  CUDA_LONG input_row_size_;
  TArray<CUDA_LONG> indices_strides_;
};
}  // namespace

template <class T>
struct FuncAssignment {
  __device__ __inline__ void operator()(T* start_addr, size_t index, T value) const { start_addr[index] = value; }
};

template <typename T, typename TIndex, bool IsGather, typename OffsetCalcT, typename TFunc>
__global__ void _GatherScatterElementsKernel(const T* src_data, const TIndex* indices_data, T* output_data,
                                             const int64_t input_dim_along_axis, const int64_t input_stride_along_axis,
                                             const OffsetCalcT offset_calc, const TFunc func, CUDA_LONG N) {
  CUDA_LONG start = kThreadsPerBlock * kThreadWorkSize * blockIdx.x + threadIdx.x;
  CUDA_LONG id;
  T value[kThreadWorkSize];

  if (!IsGather) {
    id = start;
#pragma unroll
    for (int work = 0; work < kThreadWorkSize; ++work) {
      if (id < N) {
        value[work] = src_data[id];
        id += kThreadsPerBlock;
      }
    }
  }

  id = start;
#pragma unroll
  for (int work = 0; work < kThreadWorkSize; ++work) {
    if (id < N) {
      TArray<CUDA_LONG, 2> offsets = offset_calc.get(id);
      int64_t input_offset_along_axis = static_cast<int64_t>(indices_data[offsets[1]]);
      if (input_offset_along_axis >= -input_dim_along_axis && input_offset_along_axis < input_dim_along_axis) {
        if (input_offset_along_axis < 0) input_offset_along_axis += input_dim_along_axis;
        CUDA_LONG input_offset = offsets[0] + static_cast<CUDA_LONG>(input_offset_along_axis * input_stride_along_axis);

        if (IsGather) {
          func(value, static_cast<size_t>(work), src_data[input_offset]);
        } else {
          func(output_data, static_cast<size_t>(input_offset), value[work]);
        }
      }

      id += kThreadsPerBlock;
    }
  }

  if (IsGather) {
    id = start;
#pragma unroll
    for (int work = 0; work < kThreadWorkSize; ++work) {
      if (id < N) {
        output_data[id] = value[work];
        id += kThreadsPerBlock;
      }
    }
  }
}

#define LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(src_data, is_outer_axis, is_strided_indices, is_gather)               \
  auto offset_calc = OffsetCalculatorFor2D<is_outer_axis, is_strided_indices>(args.indices_fdms[0], input_row_size,    \
                                                                              args.indices_strides);                   \
  _GatherScatterElementsKernel<T, TIndex, is_gather, decltype(offset_calc), decltype(func)>                            \
      <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(src_data, indices_data, output_data, args.input_dim_along_axis, \
                                                       args.input_stride_along_axis, offset_calc, func, N)

#define LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(src_data, is_strided_indices, is_gather)                                 \
  auto offset_calc =                                                                                                   \
      OffsetCalculator<is_strided_indices>(rank, args.masked_input_strides, args.indices_fdms, args.indices_strides);  \
  _GatherScatterElementsKernel<T, TIndex, is_gather, decltype(offset_calc), decltype(func)>                            \
      <<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(src_data, indices_data, output_data, args.input_dim_along_axis, \
                                                       args.input_stride_along_axis, offset_calc, func, N)

#define HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(src_data, is_outer_axis, is_gather) \
  if (args.indices_strides.Size() > 0) {                                                         \
    LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(src_data, is_outer_axis, true, is_gather);          \
  } else {                                                                                       \
    LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(src_data, is_outer_axis, false, is_gather);         \
  }

template <typename T, typename TIndex>
void GatherElementsImpl(cudaStream_t stream, const T* input_data, const TIndex* indices_data, T* output_data,
                        const GatherScatterElementsArgs& args) {
  CUDA_LONG N = static_cast<CUDA_LONG>(args.indices_size);
  int blocksPerGrid = static_cast<int>(CeilDiv(N, kThreadsPerBlock * kThreadWorkSize));
  auto func = FuncAssignment<T>();
  if (args.rank == 2) {
    int64_t input_row_size = args.masked_input_strides[0];
    if (args.axis == 0) {
      HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(input_data, true, true);
    } else {
      HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(input_data, false, true);
    }
    return;
  }

  int rank = static_cast<int>(args.rank);
  if (args.indices_strides.Size() > 0) {
    LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(input_data, true, true);
  } else {
    // Save one divmod in kernel if axis is the last dim.
    if (args.rank == args.axis + 1) rank -= 1;
    LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(input_data, false, true);
  }
}

template <typename T, typename TIndex, typename TFunc>
Status ScatterElementsImplInternal(cudaStream_t stream, const T* input_data, const TIndex* indices_data,
                                   const T* updates_data, T* output_data, const GatherScatterElementsArgs& args,
                                   const TFunc func) {
  if (input_data != output_data) {
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(output_data, input_data, args.input_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }

  if (args.indices_size == 0) return Status::OK();

  CUDA_LONG N = static_cast<CUDA_LONG>(args.indices_size);
  int blocksPerGrid = static_cast<int>(CeilDiv(N, kThreadsPerBlock * kThreadWorkSize));
  if (args.rank == 2) {
    int64_t input_row_size = args.masked_input_strides[0];
    if (args.axis == 0) {
      HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(updates_data, true, false);
    } else {
      HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES(updates_data, false, false);
    }
    return Status::OK();
  }

  int rank = static_cast<int>(args.rank);
  if (args.indices_strides.Size() > 0) {
    LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(updates_data, true, false);
  } else {
    // Save one divmod in kernel if axis is the last dim.
    if (args.rank == args.axis + 1) rank -= 1;
    LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL(updates_data, false, false);
  }
  return Status::OK();
}

#undef HANDLE_GATHER_SCATTER_ELEMENTS_2D_IS_STRIDED_INDICES
#undef LAUNCH_GATHER_SCATTER_ELEMENTS_KERNEL
#undef LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL

template <typename T, typename TIndex>
Status ScatterElementsImpl(cudaStream_t stream, const T* input_data, const TIndex* indices_data, const T* updates_data,
                           T* output_data, const GatherScatterElementsArgs& args) {
  return ScatterElementsImplInternal(stream, input_data, indices_data, updates_data, output_data, args,
                                     FuncAssignment<T>());
}

#define GATHER_SCATTER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, TIndex)                                                     \
  template void GatherElementsImpl<T, TIndex>(cudaStream_t stream, const T* input_data, const TIndex* indices_data,    \
                                              T* output_data, const GatherScatterElementsArgs& args);                  \
  template Status ScatterElementsImpl<T, TIndex>(cudaStream_t stream, const T* input_data, const TIndex* indices_data, \
                                                 const T* updates_data, T* output_data,                                \
                                                 const GatherScatterElementsArgs& args);

#define GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(T)           \
  GATHER_SCATTER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, int32_t) \
  GATHER_SCATTER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, int64_t)

// GatherElementsGrad needs atomic_add which supports float types only, so use half, float and double for 16, 32, and 64
// bits data respectively.
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(int8_t)
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(half)
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(float)
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(double)

#undef GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL
#undef GATHER_SCATTER_ELEMENTS_SPECIALIZED_TINDEX_IMPL

#ifdef ENABLE_TRAINING_OPS

template <class T>
struct FuncAtomicAdd {
  FuncAtomicAdd(const size_t numel) : numel_(numel) {}
  __device__ __inline__ void operator()(T* start_addr, size_t index, T value) const {
    AtomicAdd(start_addr, index, numel_, value);
  }

  const size_t numel_;
};

template <typename T, typename TIndex>
Status GatherElementsGradNonDeterministicImpl(cudaStream_t stream, const TIndex* indices_data, const T* updates_data,
                                              T* output_data,
                                              const GatherScatterElementsArgs& args) {
  // Be noted: usage of AtomicAdd is not deterministic if there are duplicated indices to update.
  // That's the reason we name this function as non-deterministic.

  // Give output_data as the input_data parameter by intention,
  // to skip input_data copy, which is not applicable for GatherElementsGrad.
  // output_data's numel is same as input_data's numel.
  return ScatterElementsImplInternal(stream, output_data, indices_data, updates_data, output_data, args,
                                     FuncAtomicAdd<T>(static_cast<size_t>(args.input_size)));
}

#define GATHER_ELEMENTS_GRAD_SPECIALIZED_TINDEX_IMPL(T, TIndex)                                                      \
  template Status GatherElementsGradNonDeterministicImpl<T, TIndex>(cudaStream_t stream, const TIndex* indices_data, \
                                                                    const T* updates_data, T* output_data,           \
                                                                    const GatherScatterElementsArgs& args);

#define GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(T) \
  GATHER_ELEMENTS_GRAD_SPECIALIZED_TINDEX_IMPL(T, int32_t)   \
  GATHER_ELEMENTS_GRAD_SPECIALIZED_TINDEX_IMPL(T, int64_t)

GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(half)
GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(float)
GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(double)

#undef GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL
#undef GATHER_ELEMENTS_GRAD_SPECIALIZED_TINDEX_IMPL

#endif

}  // namespace cuda
}  // namespace onnxruntime
