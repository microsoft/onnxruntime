// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "nvjpeg.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class ImageEncoder final : public CudaKernel {
 public:
  ImageEncoder(const OpKernelInfo& info);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ImageEncoder);
  ~ImageEncoder();

  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  std::string pixel_format_;
  nvjpegInputFormat_t input_rgb_format_;
  int quality_{70};
  int huf_{0};

  std::string subsampling_attr_{"420"};
  nvjpegChromaSubsampling_t subsampling_;

  nvjpegEncoderParams_t encode_params_;
  nvjpegHandle_t nvjpeg_handle_;
  nvjpegJpegState_t jpeg_state_;
  nvjpegEncoderState_t encoder_state_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
