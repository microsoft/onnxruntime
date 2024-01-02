// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_decoder.h"

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ImageDecoder,                                                \
      kOnnxDomain,                                                \
      20,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create()) \
          .InputMemoryType(OrtMemTypeCPUInput, 0) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ImageDecoder);

REGISTER_KERNEL_TYPED(uint8_t)

ImageDecoder::ImageDecoder(const OpKernelInfo& info) : CudaKernel(info) {
  pixel_format_ = info.GetAttrOrDefault<std::string>("pixel_format", "RGB");
}

ImageDecoder ::~ImageDecoder() {

}

Status ImageDecoder::ComputeInternal(OpKernelContext* context) const {
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
