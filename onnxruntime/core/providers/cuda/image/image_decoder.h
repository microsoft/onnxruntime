// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

using namespace onnxruntime::cuda;

class ImageDecoder final : public CudaKernel {
 public:
  ImageDecoder(const OpKernelInfo& info);
  ~ImageDecoder();
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string pixel_format_;
  nvjpegOutputFormat_t fmt_;

  nvjpegJpegState_t nvjpeg_state_;
  nvjpegHandle_t nvjpeg_handle_;
};

}  // namespace cuda
}  // namespace onnxruntime
