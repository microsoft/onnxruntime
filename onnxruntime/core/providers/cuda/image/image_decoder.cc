// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_decoder.h"
#include "nvjpeg.h"
#include "core/providers/cuda/cuda_common.h"

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
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      ImageDecoder);

REGISTER_KERNEL_TYPED(uint8_t)

int dev_malloc(void** p, size_t s) { return (int)cudaMalloc(p, s); }

int dev_free(void* p) { return (int)cudaFree(p); }

int host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }

int host_free(void* p) { return (int)cudaFreeHost(p); }

ImageDecoder::ImageDecoder(const OpKernelInfo& info) : CudaKernel(info) {
  pixel_format_ = info.GetAttrOrDefault<std::string>("pixel_format", "RGB");
  if (pixel_format_ == "RGB") {
    // convert to planar RGB
    fmt_ = NVJPEG_OUTPUT_RGB;
  } else if (pixel_format_ == "BGR") {
    // convert to planar BGR
    fmt_ = NVJPEG_OUTPUT_BGR;
  } else if (pixel_format_ == "Grayscale") {
    // return luma component only, if YCbCr colorspace,
    // or try to convert to grayscale,
    // writes to 1-st channel of nvjpegImage_t
    fmt_ = NVJPEG_OUTPUT_Y;
  } else {
    ORT_THROW("pixel_format is expected to be RGB, BGR, or Grayscale, got: ", pixel_format_);
  }

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};
  int flags = 0;
  NVJPEG_CALL_THROW(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                                 &pinned_allocator, flags, &nvjpeg_handle_));

  NVJPEG_CALL_THROW(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_));
}

ImageDecoder ::~ImageDecoder() {
  NVJPEG_CALL_THROW(nvjpegJpegStateDestroy(nvjpeg_state_));
  NVJPEG_CALL_THROW(nvjpegDestroy(nvjpeg_handle_));
}

Status ImageDecoder::ComputeInternal(OpKernelContext* context) const {
  const Tensor* encoded_stream = context->Input<Tensor>(0);
  const uint8_t* encoded_stream_data = encoded_stream->Data<uint8_t>();
  const auto& dims = encoded_stream->Shape().GetDims();

  if (dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input is expected to have 1 dimension, got ", dims.size());
  }

  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  NVJPEG_CALL_THROW(nvjpegGetImageInfo(
      nvjpeg_handle_, encoded_stream_data, dims[0],
      &channels, &subsampling, widths, heights));

  if (fmt_ == NVJPEG_OUTPUT_Y) {
    channels = 1;
  }

  // we do not work with yuv output type so the output per-channel image size is always the original image size.
  int64_t width = widths[0], height = heights[0];
  TensorShape output_shape{static_cast<int64_t>(channels), static_cast<int64_t>(height), static_cast<int64_t>(width)};
  Tensor* image = context->Output(0, output_shape);
  uint8_t* image_data = image->MutableData<uint8_t>();
  nvjpegImage_t out_planes;
  for (int c = 0; c < channels; c++) {
    out_planes.pitch[c] = width;
    out_planes.channel[c] = image_data + c * height * width;
  }

  NVJPEG_CALL_THROW(nvjpegDecode(nvjpeg_handle_, nvjpeg_state_,
                                 encoded_stream_data,
                                 dims[0], fmt_, &out_planes,
                                 Stream(context)));
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
