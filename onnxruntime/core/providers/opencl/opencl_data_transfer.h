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
  using Status = common::Status;
  using Buffer = cl::Buffer;
  using Image2D = cl::Image2D;

 public:
  OpenCLDataTransfer(const OpenCLExecutionProvider* exec, const OpenCLKernelHolder* kernels);
  ~OpenCLDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
  Status CopyTensorToBuffer(const Tensor& src, Tensor& dst);

  // Tensor* means ort Tensor holds transparent memory address. Buffer/Image2D
  // means ort Tensor holds opaque Image2D handle, peal them out of the Tensor
  // before passing it into the func
  Status CopyTensor1DToImage2D(const Tensor& src, const Image2D& dst, const Image2DDesc& desc) const;
  Status CopyTensor2DToImage2D(const Tensor& src, const Image2D& dst, const Image2DDesc& desc) const;
  Status CopyTensorNCHWcToImage2D(const Tensor& src, const Image2D& dst, const Image2DDesc& desc) const;
  Status CopyImage2DToTensor1D(const Image2D& src, const Image2DDesc& desc, Tensor& dst) const;
  Status CopyImage2DToTensor2D(const Image2D& src, const Image2DDesc& desc, Tensor& dst) const;
  Status CopyImage2DToTensorNCHWc(const Image2D& src, const Image2DDesc& desc, Tensor& dst) const;

  // dev to dev copy are all async!
  Status CopyBuffer1DToImage2D(const Buffer& src, const TensorShape shape, const Image2D& dst, const Image2DDesc& desc) const;
  Status CopyBuffer2DToImage2D(const Buffer& src, const TensorShape shape, const Image2D& dst, const Image2DDesc& desc) const;
  Status CopyBufferNCHWcToImage2D(const Buffer& src, const TensorShape shape, const Image2D& dst, const Image2DDesc& desc) const;
  Status CopyImage2DToBuffer1D(const Image2D& src, const Image2DDesc& desc, const Buffer& dst, const TensorShape shape) const;
  Status CopyImage2DToBuffer2D(const Image2D& src, const Image2DDesc& desc, const Buffer& dst, const TensorShape shape) const;
  Status CopyImage2DToBufferNCHWc(const Image2D& src, const Image2DDesc& desc, const Buffer& dst, const TensorShape shape) const;

  Status UnimplementedCopy() const;

 private:
  const OpenCLExecutionProvider* exec_;
  const OpenCLKernelHolder* kernels_;
};

}  // namespace opencl
}  // namespace onnxruntime
