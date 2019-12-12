// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "scatter_elements_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename Tin, bool OUTERAXIS>
__global__ void _ScatterElementsKernel2D(
    const int max_dim,  // max dim on the scattered axis
    const T* input_data,
    const Tin* indices_data,
    const int64_t indices_size,
    const fast_divmod indices_stride_row,
    const T* updates,
    const int64_t output_row_size,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, indices_size);

  int row, col, data_idx;
  indices_stride_row.divmod(indices_index, row, col);
  int dim = (int)(indices_data[indices_index]);
  if (dim >= -max_dim && dim < max_dim) {
    if (dim < 0) dim += max_dim;
    if (OUTERAXIS) {
      data_idx = dim * output_row_size + col;
    } else {
      data_idx = row * output_row_size + dim;
    }
    output_data[data_idx] = updates[indices_index];
  }
  // else invalid index
}

template <typename T, typename Tin>
__global__ void _ScatterElementsKernel(
    const int rank,
    const T* input_data,
    const int64_t* input_dims,
    const int64_t* input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    const int64_t* indices_dims,
    const fast_divmod* indices_strides,
    const T* updates,
    const int axis,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(indices_index, indices_size);
  int dim, remain = indices_index;
  size_t data_idx = 0;
  for (int i = 0; i < rank; ++i) {
    indices_strides[i].divmod(remain, dim, remain);
    if (i == axis) {
      dim = (int)(indices_data[indices_index]);
      if (dim < -input_dims[i] || dim >= input_dims[i]) {
        return;  // Invalid index
      }
      if (dim < 0) dim += input_dims[i];
    }
    data_idx += input_strides[i] * dim;
  }
  output_data[data_idx] = updates[indices_index];
}

// From the innermost axis (largest) check equality of dim value of input and indices.
// If same, merge it and continue. Otherwise, copy remaining. The scatter axis need
// to be keep.
static int CompactInputIndicesDims(
    int rank, int axis, int64_t* input_dims, int64_t* indices_dims,
    std::vector<int64_t>& eff_input_dims,
    std::vector<int64_t>& eff_indices_dims) {
  eff_input_dims.clear();
  eff_indices_dims.clear();

  bool could_continue_merge = true;
  if (axis < rank - 1) {
    eff_input_dims.push_back(1);
    eff_indices_dims.push_back(1);
    int i = rank - 1;
    for (; i > axis; --i) {
      if (input_dims[i] == indices_dims[i]) {
        eff_input_dims.back() *= input_dims[i];
        eff_indices_dims.back() *= indices_dims[i];
      } else {
        could_continue_merge = false;
        break;
      }
    }
    if (eff_input_dims.back() == 1) {
      eff_input_dims.pop_back();
      eff_indices_dims.pop_back();
    }
    if (!could_continue_merge) {
      for (; i > axis; --i) {
        eff_input_dims.push_back(input_dims[i]);
        eff_indices_dims.push_back(indices_dims[i]);
      }
    }
  }
  could_continue_merge = could_continue_merge && (input_dims[axis] == indices_dims[axis]);
  eff_input_dims.push_back(input_dims[axis]);
  eff_indices_dims.push_back(indices_dims[axis]);
  int new_axis = (int)(eff_input_dims.size());
  if (axis > 0) {
    if (could_continue_merge) {
      eff_input_dims.push_back(1);
      eff_indices_dims.push_back(1);
    }
    int i = axis - 1;
    for (; i >= 0 && could_continue_merge; --i) {
      if (input_dims[i] == indices_dims[i]) {
        eff_input_dims.back() *= input_dims[i];
        eff_indices_dims.back() *= indices_dims[i];
      } else {
        could_continue_merge = false;
        break;
      }
    }
    if (new_axis < (int)eff_indices_dims.size() && eff_input_dims.back() == 1) {
      eff_input_dims.pop_back();
      eff_indices_dims.pop_back();
    }
    if (!could_continue_merge) {
      for (; i >= 0 && could_continue_merge; --i) {
        eff_input_dims.push_back(input_dims[i]);
        eff_indices_dims.push_back(indices_dims[i]);
      }
    }
  }
  new_axis = eff_input_dims.size() - new_axis;
  std::reverse(eff_input_dims.begin(), eff_input_dims.end());
  std::reverse(eff_indices_dims.begin(), eff_indices_dims.end());
  return new_axis;
}

template <typename T, typename Tin>
Status ScatterElementsImpl2D(
    const T* input_data,
    const std::vector<int64_t>& input_dims,
    const Tin* indices_data,
    const int64_t indices_size,
    const std::vector<int64_t>& indices_dims,
    const T* updates,
    const int axis,
    T* output_data) {
  int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(indices_size, GridDim::maxThreadsPerBlock));
  fast_divmod indices_stride_row(indices_dims[1]);
  if (axis == 0) {
    _ScatterElementsKernel2D<T, Tin, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        gsl::narrow_cast<int>(input_dims[0]), input_data,
        indices_data, indices_size, indices_stride_row,
        updates, input_dims[1], output_data);
  } else {
    _ScatterElementsKernel2D<T, Tin, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        gsl::narrow_cast<int>(input_dims[1]), input_data,
        indices_data, indices_size, indices_stride_row,
        updates, input_dims[1], output_data);
  }
  return Status::OK();
}

template <typename T, typename Tin>
Status ScatterElementsImpl(
    const int rank,
    const T* input_data,
    const int64_t input_size,
    CudaKernel::CudaAsyncBuffer<int64_t>& buffer_input_dims,
    CudaKernel::CudaAsyncBuffer<int64_t>& buffer_input_strides,
    const Tin* indices_data,
    const int64_t indices_size,
    CudaKernel::CudaAsyncBuffer<int64_t>& buffer_indices_dims,
    CudaKernel::CudaAsyncBuffer<fast_divmod>& fdm_indices_strides,
    const T* updates,
    const int axis,
    T* output_data) {
  if (input_data != output_data) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, input_size * sizeof(T), cudaMemcpyDeviceToDevice, 0));
  }

  if (indices_size > 0) {
    std::vector<int64_t> eff_input_dims;
    std::vector<int64_t> eff_indices_dims;
    int new_axis = CompactInputIndicesDims(
        rank, axis, buffer_input_dims.CpuPtr(), buffer_indices_dims.CpuPtr(), eff_input_dims, eff_indices_dims);
    if (eff_input_dims.size() == 2) {
      return ScatterElementsImpl2D(
          input_data, eff_input_dims, indices_data, indices_size, eff_indices_dims, updates, new_axis, output_data);
    }

    ORT_RETURN_IF_ERROR(buffer_input_dims.CopyToGpu());
    ORT_RETURN_IF_ERROR(buffer_input_strides.CopyToGpu());
    ORT_RETURN_IF_ERROR(buffer_indices_dims.CopyToGpu());
    ORT_RETURN_IF_ERROR(fdm_indices_strides.CopyToGpu());
    int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(indices_size, GridDim::maxThreadsPerBlock));
    _ScatterElementsKernel<T, Tin><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        rank, input_data, buffer_input_dims.GpuPtr(), buffer_input_strides.GpuPtr(),
        indices_data, indices_size, buffer_indices_dims.GpuPtr(), fdm_indices_strides.GpuPtr(),
        updates, axis, output_data);
  }
  return Status::OK();
}

#define SPECIALIZED_TINDEX_IMPL(T, TIndex)                        \
  template Status ScatterElementsImpl<T, TIndex>(                 \
      const int rank,                                             \
      const T* input_data,                                        \
      const int64_t input_size,                                   \
      CudaKernel::CudaAsyncBuffer<int64_t>& buffer_input_dims,    \
      CudaKernel::CudaAsyncBuffer<int64_t>& buffer_input_strides, \
      const TIndex* indices_data,                                 \
      const int64_t indices_size,                                 \
      CudaKernel::CudaAsyncBuffer<int64_t>& buffer_indices_dims,  \
      CudaKernel::CudaAsyncBuffer<fast_divmod>& indices_strides,  \
      const T* updates,                                           \
      const int axis,                                             \
      T* output_data)

#define SPECIALIZED_IMPL(T)            \
  SPECIALIZED_TINDEX_IMPL(T, int32_t); \
  SPECIALIZED_TINDEX_IMPL(T, int64_t);

SPECIALIZED_IMPL(int8_t)
SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(uint16_t)
SPECIALIZED_IMPL(uint32_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(bool)

}  // namespace cuda
}  // namespace onnxruntime
