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
          .OutputMemoryType(OrtMemTypeCPUOutput, 0) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ImageEncoder);

REGISTER_KERNEL_TYPED(uint8_t)

static int dev_malloc(void** p, size_t s) { return (int)cudaMalloc(p, s); }
static int dev_free(void* p) { return (int)cudaFree(p); }
static nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};

ImageEncoder::ImageEncoder(const OpKernelInfo& info) : CudaKernel(info) {
  pixel_format_ = info.GetAttrOrDefault<std::string>("pixel_format", "RGB");
  quality_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("quality", 70));
  std::string subsampling = info.GetAttrOrDefault<std::string>("subsampling", "420");

  ORT_ENFORCE(
    pixel_format_ == "RGB" ||
    pixel_format_ == "BGR" ||
    pixel_format_ == "Grayscale",
    "ImageEncoder: invalid pixel_format: ", pixel_format_);
  input_rgb_format_ = pixel_format_ == "RGB" ? NVJPEG_INPUT_RGB : NVJPEG_INPUT_BGR;

  nvjpegChromaSubsampling_t chroma_subsampling;
  if (subsampling == "444") {
    chroma_subsampling = NVJPEG_CSS_444;
  } else if (subsampling == "422") {
    chroma_subsampling = NVJPEG_CSS_422;
  } else if (subsampling == "420") {
    chroma_subsampling = NVJPEG_CSS_420;
  } else if (subsampling == "440") {
    chroma_subsampling = NVJPEG_CSS_440;
  } else if (subsampling == "411") {
    chroma_subsampling = NVJPEG_CSS_411;
  } else if (subsampling == "410") {
    chroma_subsampling = NVJPEG_CSS_410;
  } else if (subsampling == "400") {
    chroma_subsampling = NVJPEG_CSS_GRAY;
  } else {
    ORT_THROW("Unknown or unsupported subsampling: ", subsampling);
  }

  if (pixel_format_ == "Grayscale" && chroma_subsampling != NVJPEG_CSS_GRAY) {
    // log warning attribute mismatch
    chroma_subsampling = NVJPEG_CSS_GRAY;
  }
  NVJPEG_CALL_THROW(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle_));
  NVJPEG_CALL_THROW(nvjpegJpegStateCreate(nvjpeg_handle_, &jpeg_state_));
  NVJPEG_CALL_THROW(nvjpegEncoderStateCreate(nvjpeg_handle_, &encoder_state_, NULL));
  NVJPEG_CALL_THROW(nvjpegEncoderParamsCreate(nvjpeg_handle_, &encode_params_, NULL));

  NVJPEG_CALL_THROW(nvjpegEncoderParamsSetSamplingFactors(encode_params_, chroma_subsampling, NULL));
  NVJPEG_CALL_THROW(nvjpegEncoderParamsSetQuality(encode_params_, quality_, NULL));
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

  // TODO: process N images
  // const int64_t N = dims[0];
  // const int64_t C = dims[1];
  const int64_t H = dims[2];
  const int64_t W = dims[3];

  uint8_t* image_data_ = const_cast<uint8_t*>(image_data);
  nvjpegImage_t imgdesc =
  {
    {
      image_data_,
      image_data_ + W * H,
      image_data_ + W * H * 2,
      //image_data_ + W * H * 3,
    },
    {
      (unsigned int)W,
      (unsigned int)W,
      (unsigned int)W,
      //(unsigned int)W
    }
  };

  if (pixel_format_ == "RGB" || pixel_format_ == "BGR") {
    NVJPEG_CALL_THROW(nvjpegEncodeImage(
      nvjpeg_handle_,
      encoder_state_,
      encode_params_,
      &imgdesc,
      input_rgb_format_,
      static_cast<int>(W),
      static_cast<int>(H),
      NULL));
  } else { // pixel_format_ == "Grayscale"
    NVJPEG_CALL_THROW(nvjpegEncodeYUV(
      nvjpeg_handle_,
      encoder_state_,
      encode_params_,
      &imgdesc,
      NVJPEG_CSS_GRAY,
      static_cast<int>(W),
      static_cast<int>(H),
      NULL));
  }

  size_t length;
  NVJPEG_CALL_THROW(nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_,
      encoder_state_,
      NULL,
      &length,
      NULL));

  TensorShape output_shape{static_cast<int64_t>(length)};
  Tensor* encoded_stream = context->Output(0, output_shape);
  uint8_t* encoded_stream_data = encoded_stream->MutableData<uint8_t>();

  std::vector<unsigned char> obuffer;
  obuffer.resize(length);
  NVJPEG_CALL_THROW(nvjpegEncodeRetrieveBitstream(
      nvjpeg_handle_,
      encoder_state_,
      encoded_stream_data,
      &length,
      NULL));

  //// Save the binary data to a file
  //{
  //  std::string out_filename;
  //  if (pixel_format_ == "Grayscale")
  //    out_filename = "c:/temp/encoder_test_grayscale_encoded.jpg";
  //  else if(pixel_format_ == "RGB")
  //    out_filename = "c:/temp/encoder_test_rgb_encoded.jpg";
  //  else
  //    out_filename = "c:/temp/encoder_test_bgr_encoded.jpg";

  //  std::ofstream outfile(out_filename, std::ofstream::binary);
  //  outfile.write(reinterpret_cast<const char*>(encoded_stream_data), length);
  //  outfile.close();
  //}
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
