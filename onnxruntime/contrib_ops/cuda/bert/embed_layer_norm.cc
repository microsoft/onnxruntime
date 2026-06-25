// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/embed_layer_norm_helper.h"
#include "embed_layer_norm.h"
#include "embed_layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormalization,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
EmbedLayerNorm<T>::EmbedLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);
}

template <typename T>
Status EmbedLayerNorm<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(embed_layer_norm::CheckInputs(context));

  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);  // optional. nullptr if it's distill-bert
  const Tensor* word_embedding = context->Input<Tensor>(2);
  const Tensor* position_embedding = context->Input<Tensor>(3);
  const Tensor* segment_embedding = context->Input<Tensor>(4);  // optional. nullptr if it's distill-bert
  const Tensor* gamma = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* mask = context->Input<Tensor>(7);          // optional. nullptr if not provided
  const Tensor* position_ids = context->Input<Tensor>(8);  // optional. nullptr if not provided

  const auto& input_dims = input_ids->Shape().GetDims();
  int64_t hidden_size = word_embedding->Shape()[1];

  TensorShape output_shape({input_dims[0], input_dims[1], hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({input_dims[0]});
  Tensor* mask_index = context->Output(1, mask_index_shape);

  Tensor* embedding_sum = context->Output(2, output_shape);

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);
  size_t element_size = sizeof(T);

  int word_embedding_length = static_cast<int>(word_embedding->Shape()[0]);
  int position_embedding_length = static_cast<int>(position_embedding->Shape()[0]);
  int segment_embedding_length =
      (nullptr == segment_embedding) ? 0 : static_cast<int>(segment_embedding->Shape()[0]);

  const bool broadcast_position_ids = (nullptr != position_ids && position_ids->Shape()[0] == 1);

  // Device flag raised by the kernel when an input id is outside its embedding table.
  auto error_flag = GetScratchBuffer<int>(1, context->GetComputeStream());
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(error_flag.get(), 0, sizeof(int), Stream(context)));

  ORT_RETURN_IF_ERROR(LaunchEmbedLayerNormKernel(
      Stream(context),
      output->MutableData<T>(),
      nullptr == mask_index ? nullptr : mask_index->MutableData<int32_t>(),
      input_ids->Data<int32_t>(),
      nullptr == segment_ids ? nullptr : segment_ids->Data<int32_t>(),
      nullptr == mask ? nullptr : mask->Data<int32_t>(),
      gamma->Data<T>(),
      beta->Data<T>(),
      word_embedding->Data<T>(),
      position_embedding->Data<T>(),
      nullptr == segment_embedding ? nullptr : segment_embedding->Data<T>(),
      epsilon_,
      static_cast<int>(hidden_size),
      batch_size,
      sequence_length,
      element_size,
      embedding_sum == nullptr ? nullptr : embedding_sum->MutableData<T>(),
      position_ids == nullptr ? nullptr : position_ids->Data<int32_t>(),
      broadcast_position_ids,
      word_embedding_length,
      position_embedding_length,
      segment_embedding_length,
      error_flag.get()));

  // The kernel always validates ids device-side and skips the embedding reads for any
  // out-of-range id, so input safety does not depend on the readback below; the readback only
  // upgrades a skipped (silently zeroed) row into a clean error status. cudaStreamSynchronize and
  // device-to-host copies are not permitted on a stream that is capturing a CUDA graph, so skip the
  // status readback while the stream is capturing. Ids are still kept in range device-side, and
  // error surfacing resumes on normal (non-capturing) runs.
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(Stream(context), &capture_status));
  if (capture_status == cudaStreamCaptureStatusNone) {
    auto host_error_flag = AllocateBufferOnCPUPinned<int>(1);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_error_flag.get(), error_flag.get(), sizeof(int),
                                         cudaMemcpyDeviceToHost, Stream(context)));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(Stream(context)));
    ORT_RETURN_IF(*host_error_flag.get() != 0,
                  "input id is out of range of the corresponding embedding table.");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
