// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_transfer.h"

#include "opencl_execution_provider.h"
#include "opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class OpenCLDataTransfer : public IDataTransfer {
 public:
  OpenCLDataTransfer(const OpenCLExecutionProvider* exec);
  ~OpenCLDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
  common::Status CopyTensorToBuffer(const Tensor& src, Tensor& dst);

  common::Status CopyTensor1DToImage2D(const Tensor& src, Tensor& dst) const;
  common::Status CopyTensor2DToImage2D(const Tensor& src, Tensor& dst) const;
  common::Status CopyTensorNCHWToImage2D(const Tensor& src, Tensor& dst) const;
  common::Status CopyTensorNCHWcToImage2D(const Tensor& src, Tensor& dst) const;

  common::Status CopyImage2DToTensor1D(const Tensor& src, Tensor& dst) const;
  common::Status CopyImage2DToTensor2D(const Tensor& src, Tensor& dst) const;
  common::Status CopyImage2DToTensorNCHW(const Tensor& src, Tensor& dst) const;
  common::Status CopyImage2DToTensorNCHWc(const Tensor& src, Tensor& dst) const;

 private:
  const OpenCLExecutionProvider* exec_;
};

}  // namespace opencl
}  // namespace onnxruntime
