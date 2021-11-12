// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_transfer.h"

#include "opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class OpenCLDataTransfer : public IDataTransfer {
 public:
  explicit OpenCLDataTransfer(cl::CommandQueue cmd_queue);
  ~OpenCLDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
  common::Status CopyTensorToBuffer(const Tensor& src, Tensor& dst);
  common::Status CopyTensor1DToImage2D(const Tensor& src, Tensor& dst);
  common::Status CopyTensor2DToImage2D(const Tensor& src, Tensor& dst);
  common::Status CopyTensorNCHWToImage2D(const Tensor& src, Tensor& dst);
  common::Status CopyTensorNCHWcToImage2D(const Tensor& src, Tensor& dst);

 private:
  cl::CommandQueue cmd_queue_;
};

}  // namespace opencl
}  // namespace onnxruntime
