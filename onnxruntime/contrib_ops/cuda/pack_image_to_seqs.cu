// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/PACK_IMAGE_TO_SEQS.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

#include <chrono>

using namespace onnxruntime;
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename CudaT>
__global__ void PackImageToSeqsOnDevice(
    CudaT* output,
    const CudaT* input,
    const int* seq_output_indexs_gpu,
    int* seq_offset_gpu,
    fast_divmod fdm_HW,
    fast_divmod fdm_C,
    int64_t output_CHW_size,
    int64_t output_HW_size,
    int64_t output_W_size,
    int64_t input_total_size)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, input_total_size);
    int channel_index;
    int height_index;
    int width_index;
    int hw_index;
    fdm_C.divmod(id, channel_index, hw_index);
    fdm_HW.divmod(hw_index, height_index, width_index);

    int batch_output = seq_output_indexs_gpu[width_index];
    if (batch_output == 0)
    {
        return;
    }
    batch_output --;
    int width_output = width_index - seq_offset_gpu[batch_output];
    int64_t output_pos = batch_output * output_CHW_size +
                        channel_index * output_HW_size +
                        height_index * output_W_size +
                        width_output;
    output[output_pos] = input[id];
}


template <typename T>
void PackImageToSeqsCudaImpl(
    cudaStream_t stream,
    const Tensor* input_tensor,
    Tensor* output_tensor,
    int* seq_offset_gpu,
    int* seq_output_indexs_gpu)
{
    typedef typename ToCudaType<T>::MappedType CudaT;
    const auto& input_shape = input_tensor->Shape();
    const auto input_total_size = input_shape.Size();

    const auto* input = reinterpret_cast<const CudaT*>(input_tensor->template Data<T>());
    auto* output = reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>());

    int blocksPerGrid = (int)(ceil(static_cast<float>(input_total_size) / GridDim::maxThreadsPerBlock));

    auto output_dimensions = output_tensor->Shape().GetDims();

    int64_t output_HW_size = output_dimensions[3] * output_dimensions[2];
    int64_t output_CHW_size = output_HW_size * output_dimensions[1];
    fast_divmod fdm_HW(static_cast<int>(input_shape.GetDims()[3]));
    fast_divmod fdm_C(static_cast<int>(input_shape.GetDims()[3] * input_shape.GetDims()[2]));

    PackImageToSeqsOnDevice<CudaT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        output, input, seq_output_indexs_gpu, seq_offset_gpu, fdm_HW, fdm_C,
        output_CHW_size, output_HW_size, output_dimensions[3], input_total_size);

    return;
}

#define PACK_IMAGE_TO_SEQS_IMPL(T)                   \
    template void PackImageToSeqsCudaImpl<T>(        \
                    cudaStream_t stream,            \
                    const Tensor* input_tensor,     \
                    Tensor* output_tensor,          \
                    int* seq_offset_gpu,            \
                    int* seq_output_indexs_gpu);

PACK_IMAGE_TO_SEQS_IMPL(MLFloat16)
PACK_IMAGE_TO_SEQS_IMPL(double)
PACK_IMAGE_TO_SEQS_IMPL(float)
PACK_IMAGE_TO_SEQS_IMPL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
