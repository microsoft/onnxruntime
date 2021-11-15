// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_data_transfer.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"

#include "opencl_utils.h"

namespace onnxruntime {
namespace opencl {

OpenCLDataTransfer::OpenCLDataTransfer(const OpenCLExecutionProvider* exec) : exec_(exec) {}

OpenCLDataTransfer::~OpenCLDataTransfer() {}

bool OpenCLDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU);
}

common::Status OpenCLDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  ORT_ENFORCE(exec_queue_id == 0);

  // *.Location().mem_type == *.Location().device.MemType() for OpenCL EP
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  // HOST --> DEV
  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_ENFORCE(src.ByteOffset() == 0);
    if (dst_device.MemType() == CLMemType::OPENCL_BUFFER) {
      auto dst_buffer = CL_BUFFER_FROM_TENSOR(dst);
      std::cerr << "OpenCL copy host " << src.DataRaw() << " --> device cl::Buffer(" << dst_buffer() << ")\n";
      OPENCL_CHECK_ERROR(exec_->GetCommandQueue().enqueueWriteBuffer(dst_buffer, CL_TRUE, 0, src.SizeInBytes(), src.DataRaw()));
      return Status::OK();
    } else if (dst_device.MemType() == CLMemType::OPENCL_IMAGE_2D) {
      switch (src.Shape().NumDimensions()) {
        case 1:
          return CopyTensor1DToImage2D(src, dst);
        case 2:
          return CopyTensor2DToImage2D(src, dst);
        case 4:
          return CopyTensorNCHWToImage2D(src, dst);
        case 5:
          return CopyTensorNCHWcToImage2D(src, dst);
      }
    }
  }

  // DEV --> HOST
  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_ENFORCE(dst.ByteOffset() == 0);
    if (src_device.MemType() == CLMemType::OPENCL_BUFFER) {
      auto src_buffer = CL_BUFFER_FROM_TENSOR(src);
      std::cerr << "OpenCL copy host " << src.DataRaw() << " <-- device cl::Buffer(" << src_buffer() << ")\n";
      OPENCL_CHECK_ERROR(exec_->GetCommandQueue().enqueueReadBuffer(src_buffer, CL_TRUE, 0, dst.SizeInBytes(), dst.MutableDataRaw()));
      return Status::OK();
    } else if (src_device.MemType() == CLMemType::OPENCL_IMAGE_2D) {
      switch (src.Shape().NumDimensions()) {
        case 1:
          return CopyImage2DToTensor1D(src, dst);
        case 2:
          return CopyImage2DToTensor2D(src, dst);
        case 4:
          return CopyImage2DToTensorNCHW(src, dst);
        case 5:
          return CopyImage2DToTensorNCHWc(src, dst);
      }
    }
  }

  return ORT_MAKE_STATUS(NONE, EP_FAIL, "Cannot copy tensor from ", src_device, " to ", dst_device);
}

common::Status OpenCLDataTransfer::CopyTensor1DToImage2D(const Tensor& src, Tensor& dst) const {
  // NOTE: if we use enqueueWriteImage here, we need to correctly handle the
  // tail data. The part of data that cannot fill the image width is tail data.
  // That is:
  // 1. If the Image2D is of shape [1, 1024], src tensor has 8 elements in it.
  //    we can launch enqueueWriteImage with only 2 pixel (2 float4 is 8
  //    elements) for this case.
  // 2. If the Image2D is of shape [2, 1024], src tensor has 4096 + 8 elements,
  //    we can launch two enqueueWriteImage with 1024 pixels and 2 pixels.
  //
  // However, we are unable use this method to handle the part of data that
  // cannot fill a pixel! E.g., tail data has only 7 element. Use
  // enqueueWriteImage will cause opencl driver out-of-bound read.
  //
  // Again, however, if the p_data of src is allocated as aligned to the
  // pagesize boundary, then the tail will always sit in a page owned by us.
  // Thus, no segfault will happen in this case.
  //
  // So copy the data to a buffer and then copy it with kernel might be a better
  // solution here...

  cl_int err{};
  float* b = const_cast<float*>(src.Data<float>());
  cl::Buffer tmp_buffer(exec_->GetCommandQueue(), b, b + src.SizeInBytes(), /*readOnly=*/true, /*useHostPtr=*/true, &err);
  OPENCL_CHECK_ERROR(err);

  const auto desc = Image2DDesc::PackFromTensor1D(src.Shape());

  KernelLauncher{exec_->GetCopyTensor1DToImage2DKernel()}
      .setBuffer(tmp_buffer)
      .setArg<cl_int>(CeilDiv(src.Shape().Size(), 4))
      .setImage2D(CL_IMAGE2D_FROM_TENSOR(dst))
      .Launch(exec_->GetCommandQueue(), cl::NDRange(desc.Width(), desc.Height()));

  return Status::OK();
}

common::Status OpenCLDataTransfer::CopyTensor2DToImage2D(const Tensor& src, Tensor& dst) const {
  ORT_NOT_IMPLEMENTED();
}

common::Status OpenCLDataTransfer::CopyTensorNCHWToImage2D(const Tensor& src, Tensor& dst) const {
  ORT_NOT_IMPLEMENTED();
}

common::Status OpenCLDataTransfer::CopyTensorNCHWcToImage2D(const Tensor& src, Tensor& dst) const {
  ORT_NOT_IMPLEMENTED();
}

common::Status OpenCLDataTransfer::CopyImage2DToTensor1D(const Tensor& src, Tensor& dst) const {
  // NOTE: See CopyTensor1DToImage2D for the underlying issue of using
  // enqueueReadImage
  ORT_NOT_IMPLEMENTED();
}

common::Status OpenCLDataTransfer::CopyImage2DToTensor2D(const Tensor& src, Tensor& dst) const {
  ORT_NOT_IMPLEMENTED();
}

common::Status OpenCLDataTransfer::CopyImage2DToTensorNCHW(const Tensor& src, Tensor& dst) const {
  ORT_NOT_IMPLEMENTED();
}

common::Status OpenCLDataTransfer::CopyImage2DToTensorNCHWc(const Tensor& src, Tensor& dst) const {
  ORT_NOT_IMPLEMENTED();
}

}  // namespace opencl
}  // namespace onnxruntime
