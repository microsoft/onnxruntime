// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_encoder.h"
#include "nvjpeg.h"
#include "core/providers/cuda/cuda_common.h"
#include <fstream>

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ImageEncoder,                                                \
      kMSDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ImageEncoder);

REGISTER_KERNEL_TYPED(uint8_t)

static int dev_malloc(void** p, size_t s) { return (int)cudaMalloc(p, s); }
static int dev_free(void* p) { return (int)cudaFree(p); }
static nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};

ImageEncoder::ImageEncoder(const OpKernelInfo& info) : CudaKernel(info) {
  pixel_format_ = info.GetAttrOrDefault<std::string>("pixel_format", "RGB");
  ORT_ENFORCE(
    pixel_format_ == "RGB" ||
    pixel_format_ == "BGR" ||
    pixel_format_ == "RGBI" ||
    pixel_format_ == "BGRI" ||
    pixel_format_ == "YUV" ||
    pixel_format_ == "Y",
    "ImageEncoder: invalid pixel_format: ", pixel_format_);

  if (pixel_format_ == "RGB" || pixel_format_ == "BGR") {
    input_rgb_format_ = pixel_format_ == "RGB" ? NVJPEG_INPUT_RGB : NVJPEG_INPUT_BGR;
  } else if (pixel_format_ == "RGBI" || pixel_format_ == "BGRI") {
    input_rgb_format_ = pixel_format_ == "RGB" ? NVJPEG_INPUT_RGBI : NVJPEG_INPUT_BGRI;
  } else if (pixel_format_ == "YUV" || pixel_format_ == "Y") {
    // nvjpegEncodeYUV does not need input_rgb_format_.
  }

  NVJPEG_CALL_THROW(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle_));
  NVJPEG_CALL_THROW(nvjpegJpegStateCreate(nvjpeg_handle_, &jpeg_state_));
  NVJPEG_CALL_THROW(nvjpegEncoderStateCreate(nvjpeg_handle_, &encoder_state_, NULL));
  NVJPEG_CALL_THROW(nvjpegEncoderParamsCreate(nvjpeg_handle_, &encode_params_, NULL));

  // sample input parameters
  NVJPEG_CALL_THROW(nvjpegEncoderParamsSetQuality(encode_params_, quality_, NULL));
  NVJPEG_CALL_THROW(nvjpegEncoderParamsSetOptimizedHuffman(encode_params_, huf_, NULL));

  if (subsampling_attr_ == "444") {
    NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_444, NULL));
    subsampling_ = NVJPEG_CSS_444;
  } else if (subsampling_attr_ == "422") {
    NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_422, NULL));
    subsampling_ = NVJPEG_CSS_422;
  } else if (subsampling_attr_ == "420") {
    NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_420, NULL));
    subsampling_ = NVJPEG_CSS_420;
  } else if (subsampling_attr_ == "440") {
    NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_440, NULL));
    subsampling_ = NVJPEG_CSS_440;
  } else if (subsampling_attr_ == "411") {
    NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_411, NULL));
    subsampling_ = NVJPEG_CSS_411;
  } else if (subsampling_attr_ == "410") {
    NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_410, NULL));
    subsampling_ = NVJPEG_CSS_410;
  } else if (subsampling_attr_ == "400") {
    NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_GRAY, NULL));
    subsampling_ = NVJPEG_CSS_GRAY;
  } else {

  }
}

ImageEncoder::~ImageEncoder() {
  NVJPEG_CALL_THROW(nvjpegEncoderParamsDestroy(encode_params_));
  NVJPEG_CALL_THROW(nvjpegEncoderStateDestroy(encoder_state_));
  NVJPEG_CALL_THROW(nvjpegJpegStateDestroy(jpeg_state_));
  NVJPEG_CALL_THROW(nvjpegDestroy(nvjpeg_handle_));
}

Status ImageEncoder::ComputeInternal(OpKernelContext* context) const {
  const Tensor* image = context->Input<Tensor>(0);
  const uint8_t* image_data = image->Data<uint8_t>();
  const auto& dims = image->Shape().GetDims();

  if (dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input is expected to have four dimensions corresponding to [N,C,H,W], got ", dims.size());
  }

  const int64_t N = dims[0];
  const int64_t C = dims[1];
  const int64_t H = dims[2];
  const int64_t W = dims[3];

  bool is_interleaved = false;
  bool is_rgb;
  if (pixel_format_ == "RGB" || pixel_format_ == "BGR") {
    is_rgb = true;
  }
  else if (pixel_format_ == "RGBI" || pixel_format_ == "BGRI") {
    is_interleaved = true;
    is_rgb = true;
  }
  else if (pixel_format_ == "YUV") {
    is_rgb = false;
  }
  else if (pixel_format_ == "Y") {
    is_rgb = false;
  }

  uint8_t* image_data_ = const_cast<uint8_t*>(image_data);
  nvjpegImage_t imgdesc =
  {
    {
      image_data_,
      image_data_ + W * H,
      image_data_ + W * H * 2,
      image_data_ + W * H * 3,
    },
    {
      (unsigned int)(is_interleaved ? W * 3 : W),
      (unsigned int)W,
      (unsigned int)W,
      (unsigned int)W
    }
  };

  if (is_rgb) {
    NVJPEG_CALL_THROW(nvjpegEncodeImage(
      nvjpeg_handle_,
      encoder_state_,
      encode_params_,
      &imgdesc,
      input_rgb_format_,
      W,
      H,
      NULL));
  } else {
    NVJPEG_CALL_THROW(nvjpegEncodeYUV(
      nvjpeg_handle_,
      encoder_state_,
      encode_params_,
      &imgdesc,
      subsampling_,
      W,
      H,
      NULL));
  }

  size_t length;
  NVJPEG_CALL_THROW(nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_,
      encoder_state_,
      NULL,
      &length,
      NULL));
  std::vector<unsigned char> obuffer;
  obuffer.resize(length);
  NVJPEG_CALL_THROW(nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_,
      encoder_state_,
      obuffer.data(),
      &length,
      NULL));


  // ...

  // Save the binary data to a file
  std::ofstream outfile("c:/temp/encoded.jpg", std::ofstream::binary);
  outfile.write(reinterpret_cast<const char*>(obuffer.data()), obuffer.size());
  outfile.close();

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
