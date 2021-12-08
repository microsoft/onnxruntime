// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_utils.h"
#include "opencl_kernel_holder.h"
#include "opencl_data_transfer.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace opencl {

OpenCLDataTransfer::OpenCLDataTransfer(const OpenCLExecutionProvider* exec, const OpenCLKernelHolder* kernels) : exec_(exec), kernels_(kernels) {}

OpenCLDataTransfer::~OpenCLDataTransfer() {}

bool OpenCLDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU && src_device.MemType() != src_device.MemType());
}

Status OpenCLDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  ORT_ENFORCE(exec_queue_id == 0);
  // *.Location().mem_type == *.Location().device.MemType() for OpenCL EP
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  // Tensors when used as weight, some OPs use special layout optimization,
  // need to be specially handled.
  if (dst.Usage() != TensorUsage::Generic) {
    ORT_ENFORCE(src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU);
    switch (dst.Usage()) {
      case TensorUsage::ConvWeight:
        return CopyConvWeight(src, dst);
      case TensorUsage::DepthwiseConvWeight:
        return CopyDepthwiseConvWeight(src, dst);
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "unhandled tensor copy of special usage");
    }
  }

  // HOST ==> DEV
  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_ENFORCE(src.ByteOffset() == 0);
    if (dst_device.MemType() == CLMemType::OPENCL_BUFFER) {
      VLOGF_DEFAULT(0, "[CL] copy    host(%p) ---> Buffer(%p)", src.DataRaw(), CL_BUFFER_FROM_TENSOR(dst)());
      ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueWriteBuffer(CL_BUFFER_FROM_TENSOR(dst), CL_TRUE, 0, src.SizeInBytes(), src.DataRaw()));
      return Status::OK();
    }

    if (dst_device.MemType() == CLMemType::OPENCL_IMAGE_2D) {
      auto desc = Image2DDesc::PackFromTensor(src.Shape());
      switch (src.Shape().NumDimensions()) {
        case 1:
          return CopyTensor1DToImage2D(src, CL_IMAGE2D_FROM_TENSOR(dst), desc);
        case 2:
          return CopyTensor2DToImage2D(src, CL_IMAGE2D_FROM_TENSOR(dst), desc);
        case 3:
          return UnimplementedCopy();
        case 4:
          return CopyTensorNCHWToImage2D(src, CL_IMAGE2D_FROM_TENSOR(dst), desc);
        case 5:
          return CopyTensorNCHWcToImage2D(src, CL_IMAGE2D_FROM_TENSOR(dst), desc);
      }
    }
  }

  // DEV ==> HOST
  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_ENFORCE(dst.ByteOffset() == 0);
    if (src_device.MemType() == CLMemType::OPENCL_BUFFER) {
      VLOGF_DEFAULT(0, "[CL] copy  Buffer(%p) -----> host(%p)", CL_BUFFER_FROM_TENSOR(src)(), dst.DataRaw());
      ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueReadBuffer(CL_BUFFER_FROM_TENSOR(src), CL_TRUE, 0, dst.SizeInBytes(), dst.MutableDataRaw()));
      return Status::OK();
    }

    if (src_device.MemType() == CLMemType::OPENCL_IMAGE_2D) {
      auto desc = Image2DDesc::PackFromTensor(src.Shape());
      switch (src.Shape().NumDimensions()) {
        case 1:
          return CopyImage2DToTensor1D(CL_IMAGE2D_FROM_TENSOR(src), desc, dst);
        case 2:
          return CopyImage2DToTensor2D(CL_IMAGE2D_FROM_TENSOR(src), desc, dst);
        case 3:
          return UnimplementedCopy();
        case 4:
          return CopyImage2DToTensorNCHW(CL_IMAGE2D_FROM_TENSOR(src), desc, dst);
        case 5:
          return CopyImage2DToTensorNCHWc(CL_IMAGE2D_FROM_TENSOR(src), desc, dst);
      }
    }
  }

  // Buffer <==> Image2D (dev --> dev)
  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU) {
    if (src_device.MemType() == CLMemType::OPENCL_BUFFER && dst_device.MemType() == CLMemType::OPENCL_IMAGE_2D) {
      auto desc = Image2DDesc::PackFromTensor(src.Shape());
      switch (src.Shape().NumDimensions()) {
        case 1:
          return CopyBuffer1DToImage2D(CL_BUFFER_FROM_TENSOR(dst), dst.Shape(), CL_IMAGE2D_FROM_TENSOR(src), desc);
        case 2:
          return CopyBuffer2DToImage2D(CL_BUFFER_FROM_TENSOR(dst), dst.Shape(), CL_IMAGE2D_FROM_TENSOR(src), desc);
        case 3:
          return UnimplementedCopy();
        case 4:
          return CopyBufferNCHWToImage2D(CL_BUFFER_FROM_TENSOR(dst), dst.Shape(), CL_IMAGE2D_FROM_TENSOR(src), desc);
        case 5:
          return CopyBufferNCHWcToImage2D(CL_BUFFER_FROM_TENSOR(dst), dst.Shape(), CL_IMAGE2D_FROM_TENSOR(src), desc);
      }
    }

    if (src_device.MemType() == CLMemType::OPENCL_IMAGE_2D && dst_device.MemType() == CLMemType::OPENCL_BUFFER) {
      auto desc = Image2DDesc::PackFromTensor(src.Shape());
      switch (src.Shape().NumDimensions()) {
        case 1:
          return CopyImage2DToBuffer1D(CL_IMAGE2D_FROM_TENSOR(src), desc, CL_BUFFER_FROM_TENSOR(dst), dst.Shape());
        case 2:
          return CopyImage2DToBuffer2D(CL_IMAGE2D_FROM_TENSOR(src), desc, CL_BUFFER_FROM_TENSOR(dst), dst.Shape());
        case 3:
          return UnimplementedCopy();
        case 4:
          return CopyImage2DToBufferNCHW(CL_IMAGE2D_FROM_TENSOR(src), desc, CL_BUFFER_FROM_TENSOR(dst), dst.Shape());
        case 5:
          return CopyImage2DToBufferNCHWc(CL_IMAGE2D_FROM_TENSOR(src), desc, CL_BUFFER_FROM_TENSOR(dst), dst.Shape());
      }
    }
  }

  return ORT_MAKE_STATUS(NONE, EP_FAIL, "Cannot copy tensor from ", src_device, " to ", dst_device);
}

Status OpenCLDataTransfer::CopyTensor1DToImage2D(const Tensor& src, const Image2D& dst, const Image2DDesc& desc) const {
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
  VLOGF_DEFAULT(0, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), dst());
  auto tmp = exec_->GetScratchBuffer(src.SizeInBytes());
  ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueWriteBuffer(*tmp, /*blocking=*/CL_FALSE, 0, src.SizeInBytes(), src.DataRaw()));
  ORT_RETURN_IF_ERROR(CopyBuffer1DToImage2D(*tmp, src.Shape(), dst, desc));
  exec_->GetCommandQueue().finish();  // do sync copy, since we cannot extend the lifetime of src or tmp
  return Status::OK();
}

Status OpenCLDataTransfer::CopyTensor2DToImage2D(const Tensor& src, const Image2D& dst, const Image2DDesc& desc) const {
  VLOGF_DEFAULT(0, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), dst());
  ORT_NOT_IMPLEMENTED("CopyTensor2DToImage2D");
}

Status OpenCLDataTransfer::CopyTensorNCHWToImage2D(const Tensor& src, const Image2D& dst, const Image2DDesc& desc) const {
  VLOGF_DEFAULT(0, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), dst());
  auto tmp = exec_->GetScratchBuffer(src.SizeInBytes());
  ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueWriteBuffer(*tmp, /*blocking=*/CL_FALSE, 0, src.SizeInBytes(), src.DataRaw()));
  ORT_RETURN_IF_ERROR(CopyBufferNCHWToImage2D(*tmp, src.Shape(), dst, desc));
  exec_->GetCommandQueue().finish();  // do sync copy, since we cannot extend the lifetime of src or tmp
  return Status::OK();
}

Status OpenCLDataTransfer::CopyTensorNCHWcToImage2D(const Tensor& src, const Image2D& dst, const Image2DDesc& desc) const {
  VLOGF_DEFAULT(0, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), dst());
  ORT_NOT_IMPLEMENTED("CopyTensorNCHWcToImage2D");
}

Status OpenCLDataTransfer::CopyImage2DToTensor1D(const Image2D& src, const Image2DDesc& desc, Tensor& dst) const {
  // NOTE: See CopyTensor1DToImage2D for the underlying issue of using enqueueReadImage
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) -----> host(%p)", src(), dst.DataRaw());
  auto tmp = exec_->GetScratchBuffer(dst.SizeInBytes());
  ORT_RETURN_IF_ERROR(CopyImage2DToBuffer1D(src, desc, *tmp, dst.Shape()));
  // do sync copy, since we cannot extend the lifetime of src or tmp
  ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueReadBuffer(*tmp, /*blocking=*/CL_TRUE, /*offset=*/0, dst.SizeInBytes(), dst.MutableDataRaw()));
  return Status::OK();
}

Status OpenCLDataTransfer::CopyImage2DToTensor2D(const Image2D& src, const Image2DDesc& desc, Tensor& dst) const {
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) -----> host(%p)", src(), dst.DataRaw());
  ORT_NOT_IMPLEMENTED("CopyImage2DToTensor2D");
}

Status OpenCLDataTransfer::CopyImage2DToTensorNCHW(const Image2D& src, const Image2DDesc& desc, Tensor& dst) const {
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) -----> host(%p)", src(), dst.DataRaw());
  auto tmp = exec_->GetScratchBuffer(dst.SizeInBytes());
  ORT_RETURN_IF_ERROR(CopyImage2DToBufferNCHW(src, desc, *tmp, dst.Shape()));
  // do sync copy, since we cannot extend the lifetime of src or tmp
  ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueReadBuffer(*tmp, /*blocking=*/CL_TRUE, /*offset=*/0, dst.SizeInBytes(), dst.MutableDataRaw()));
  return Status::OK();
}

Status OpenCLDataTransfer::CopyImage2DToTensorNCHWc(const Image2D& src, const Image2DDesc& desc, Tensor& dst) const {
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) -----> host(%p)", src(), dst.DataRaw());
  ORT_NOT_IMPLEMENTED("CopyImage2DToTensorNCHWc");
}

Status OpenCLDataTransfer::CopyBuffer1DToImage2D(
    const Buffer& src,
    const TensorShape shape,
    const Image2D& dst,
    const Image2DDesc& desc) const {
  VLOGF_DEFAULT(0, "[CL] copy  Buffer(%p) --> Image2D(%p)", src(), dst());
  ORT_RETURN_IF_ERROR(
      KernelLauncher{kernels_->GetKernel("CopyBuffer1DToImage2D")}
          .setArg<cl_int>(desc.Width())
          .setArg<cl_int>(desc.Height())
          .setBuffer(src)
          .setArg<cl_int>(shape.Size())  // nelem
          .setImage2D(dst)
          .Launch(exec_->GetCommandQueue(), desc.AsNDRange()));
  return Status::OK();
}

Status OpenCLDataTransfer::CopyBuffer2DToImage2D(
    const Buffer& src,
    const TensorShape shape,
    const Image2D& dst,
    const Image2DDesc& desc) const {
  VLOGF_DEFAULT(0, "[CL] copy  Buffer(%p) --> Image2D(%p)", src(), dst());
  ORT_NOT_IMPLEMENTED("CopyBuffer2DToImage2D");
}

Status OpenCLDataTransfer::CopyBufferNCHWToImage2D(
    const Buffer& src,
    const TensorShape shape,
    const Image2D& dst,
    const Image2DDesc& desc) const {
  VLOGF_DEFAULT(0, "[CL] copy  Buffer(%p) --> Image2D(%p)", src(), dst());
  ORT_RETURN_IF_ERROR(
      KernelLauncher{kernels_->GetKernel("CopyBufferNCHWToImage2D")}
          .setArg<cl_int>(desc.Width())
          .setArg<cl_int>(desc.Height())
          .setBuffer(src)
          .setArg<cl_int>(shape[1])  // C
          .setArg<cl_int>(shape[2])  // H
          .setArg<cl_int>(shape[3])  // W
          .setImage2D(dst)
          .Launch(exec_->GetCommandQueue(), desc.AsNDRange()));
  return Status::OK();
}

Status OpenCLDataTransfer::CopyBufferNCHWcToImage2D(
    const Buffer& src,
    const TensorShape shape,
    const Image2D& dst,
    const Image2DDesc& desc) const {
  VLOGF_DEFAULT(0, "[CL] copy  Buffer(%p) --> Image2D(%p)", src(), dst());
  ORT_NOT_IMPLEMENTED("CopyBufferNCHWcToImage2D");
}

Status OpenCLDataTransfer::CopyImage2DToBuffer1D(
    const Image2D& src,
    const Image2DDesc& desc,
    const Buffer& dst,
    const TensorShape shape) const {
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) ---> Buffer(%p)", src(), dst());
  ORT_RETURN_IF_ERROR(
      KernelLauncher{kernels_->GetKernel("CopyImage2DToBuffer1D")}
          .setArg<cl_int>(desc.Width())
          .setArg<cl_int>(desc.Height())
          .setImage2D(src)
          .setBuffer(dst)
          .setArg<cl_int>(shape.Size())  // nelem
          .Launch(exec_->GetCommandQueue(), desc.AsNDRange()));
  return Status::OK();
}

Status OpenCLDataTransfer::CopyImage2DToBuffer2D(
    const Image2D& src,
    const Image2DDesc& desc,
    const Buffer& dst,
    const TensorShape shape) const {
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) ---> Buffer(%p)", src(), dst());
  ORT_NOT_IMPLEMENTED("CopyImage2DToBuffer2D");
}

Status OpenCLDataTransfer::CopyImage2DToBufferNCHW(
    const Image2D& src,
    const Image2DDesc& desc,
    const Buffer& dst,
    const TensorShape shape) const {
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) ---> Buffer(%p)", src(), dst());
  ORT_RETURN_IF_ERROR(
      KernelLauncher{kernels_->GetKernel("CopyImage2DToBufferNCHW")}
          .setArg<cl_int>(desc.Width())
          .setArg<cl_int>(desc.Height())
          .setImage2D(src)
          .setBuffer(dst)
          .setArg<cl_int>(shape[1])  // C
          .setArg<cl_int>(shape[2])  // H
          .setArg<cl_int>(shape[3])  // W
          .Launch(exec_->GetCommandQueue(), desc.AsNDRange()));
  return Status::OK();
}

Status OpenCLDataTransfer::CopyImage2DToBufferNCHWc(
    const Image2D& src,
    const Image2DDesc& desc,
    const Buffer& dst,
    const TensorShape shape) const {
  VLOGF_DEFAULT(0, "[CL] copy Image2D(%p) ---> Buffer(%p)", src(), dst());
  ORT_NOT_IMPLEMENTED("CopyImage2DToBufferNCHWc");
}

Status OpenCLDataTransfer::UnimplementedCopy() const {
  ORT_NOT_IMPLEMENTED("Not Implemented Copy");
};

Status OpenCLDataTransfer::CopyConvWeight(const Tensor& src, Tensor& dst) const {
  auto dst_image2d = CL_IMAGE2D_FROM_TENSOR(dst);
  auto desc = Image2DDesc::PackFromConv2DWeight(dst.Shape());
  VLOGF_DEFAULT(0, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), dst_image2d());

  auto shape = src.Shape();
  auto tmp = exec_->GetScratchBuffer(src.SizeInBytes());
  ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueWriteBuffer(*tmp, /*blocking=*/CL_FALSE, 0, src.SizeInBytes(), src.DataRaw()));
  ORT_RETURN_IF_ERROR(KernelLauncher{kernels_->GetKernel("Conv2DWeightBufferToImage")}
                          .setArg<cl_int>(desc.Width())
                          .setArg<cl_int>(desc.Height())
                          .setBuffer(*tmp)
                          .setInt4(shape[0], shape[1], shape[2], shape[3])
                          .setArg<cl_int>(shape[2] * shape[3])
                          .setImage2D(dst_image2d)
                          .Launch(exec_->GetCommandQueue(), desc.AsNDRange()));
  exec_->GetCommandQueue().finish();  // do sync copy, since we cannot extend the lifetime of src or tmp
  return Status::OK();
}

Status OpenCLDataTransfer::CopyDepthwiseConvWeight(const Tensor& src, Tensor& dst) const {
  auto dst_image2d = CL_IMAGE2D_FROM_TENSOR(dst);
  auto desc = Image2DDesc::PackFromDepthwiseConv2DWeight(dst.Shape());
  VLOGF_DEFAULT(0, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), dst_image2d());

  auto shape = src.Shape();
  ORT_ENFORCE(dst.Shape()[0] == 1);  // TODO: maybe generalize it
  auto tmp = exec_->GetScratchBuffer(src.SizeInBytes());
  ORT_RETURN_IF_CL_ERROR(exec_->GetCommandQueue().enqueueWriteBuffer(*tmp, /*blocking=*/CL_FALSE, 0, src.SizeInBytes(), src.DataRaw()));
  ORT_RETURN_IF_ERROR(KernelLauncher{kernels_->GetKernel("CopyDepthwiseConvWeightBufferToImage")}
                          .setArg<cl_int>(desc.Width())
                          .setArg<cl_int>(desc.Height())
                          .setBuffer(*tmp)
                          .setInt4(shape[0], shape[1], shape[2], shape[3])
                          .setArg<cl_int>(desc.Width() * desc.Height())
                          .setImage2D(dst_image2d)
                          .Launch(exec_->GetCommandQueue(), desc.AsNDRange()));
  return Status::OK();
}

}  // namespace opencl
}  // namespace onnxruntime
