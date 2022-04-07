// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/mask_fill.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime;
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename CudaT>
__global__ void MaskFillOnAxis(
    CudaT* output,
    const int* mask,
    fast_divmod fdm_dim_on_axis,
    fast_divmod fdm_element_per_axis,
    int axis,
    int64_t total_size)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, total_size);
    int idx_on_axis = fdm_dim_on_axis.mod(fdm_element_per_axis.div(id));
    if (mask[idx_on_axis] == 0)
    {
        output[id] = static_cast<CudaT>(0.);
    }
}

template <typename T>
void MaskFillCudaImpl(
    cudaStream_t stream,
    Tensor* output_tensor,
    const Tensor* mask_tensor,
    int axis)
{
    typedef typename ToCudaType<T>::MappedType CudaT;

    auto* output = reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>());
    const auto& output_dims = output_tensor->Shape().GetDims();
    const auto total_size = output_tensor->Shape().Size();

    const auto* mask = mask_tensor->template Data<int>();
    const auto& mask_shape = mask_tensor->Shape();

    int blocksPerGrid = (int)(ceil(static_cast<float>(total_size) / GridDim::maxThreadsPerBlock));
    int element_per_axis = 1;
    for (auto i = output_dims.begin() + axis + 1; i != output_dims.end(); i++)
    {
        element_per_axis *= static_cast<int>(*i);
    }
    fast_divmod fdm_element_per_axis(element_per_axis);
    fast_divmod fdm_dim_on_axis(static_cast<int>(output_dims[axis]));
    MaskFillOnAxis<CudaT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        output, mask, fdm_dim_on_axis, fdm_element_per_axis, axis, output_tensor->Shape().Size());
}

#define MASK_FILL_CUDA_IMPL(T)                       \
    template void MaskFillCudaImpl<T>(               \
        cudaStream_t stream,                         \
        Tensor * output_tensor,                      \
        const Tensor* mask_tensor,                   \
        int axis);

MASK_FILL_CUDA_IMPL(double)
MASK_FILL_CUDA_IMPL(float)
MASK_FILL_CUDA_IMPL(MLFloat16)
MASK_FILL_CUDA_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
