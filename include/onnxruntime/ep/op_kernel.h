// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/pch.h instead."
#endif

#include <memory>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/tensor.h"

#include "api.h"
#include "node.h"
#include "op_kernel_info.h"

namespace onnxruntime {
struct PrePackedWeights;
struct TensorShape;
}  // namespace onnxruntime

namespace onnxruntime {
namespace ep {
namespace detail {

struct OpKernelContext {
  explicit OpKernelContext(OrtKernelContext* context) : context_{context} {
    input_tensors_.resize(InputCount());
    output_tensors_.resize(OutputCount());
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
  const T* Input(int index) const {
    if (input_tensors_[index] != nullptr) {
      return static_cast<const T*>(input_tensors_[index].get());
    }

    auto input = context_.GetInput(index);
    if (input == nullptr || !input.IsTensor()) {
      return nullptr;
    }

    auto type_and_shape_info = input.GetTypeInfo().GetTensorTypeAndShapeInfo();
    auto type = type_and_shape_info.GetElementType();
    auto shape_vec = type_and_shape_info.GetShape();

    auto memory_info = input.GetTensorMemoryInfo();
    MLDataType data_type = DataTypeImpl::TensorTypeFromONNXEnum(type);

    input_tensors_[index] = std::make_unique<Tensor>(data_type,
                                                     TensorShape{shape_vec},
                                                     const_cast<void*>(input.GetTensorRawData()),
                                                     OrtMemoryInfo{
                                                         memory_info.GetAllocatorName(),
                                                         memory_info.GetAllocatorType(),
                                                         OrtDevice{
                                                             static_cast<OrtDevice::DeviceType>(memory_info.GetDeviceType()),
                                                             static_cast<OrtDevice::MemoryType>(memory_info.GetMemoryType()),
                                                             static_cast<OrtDevice::VendorId>(memory_info.GetVendorId()),
                                                             static_cast<OrtDevice::DeviceId>(memory_info.GetDeviceId()),

                                                         },
                                                         memory_info.GetMemoryType()});
    return static_cast<const T*>(input_tensors_[index].get());
  }
  Tensor* Output(int index, const TensorShape& shape) {
    if (output_tensors_[index] != nullptr) {
      return output_tensors_[index].get();
    }
    auto output = context_.GetOutput(index, shape.GetDims().data(), shape.GetDims().size());
    auto type_and_shape_info = output.GetTypeInfo().GetTensorTypeAndShapeInfo();
    auto type = type_and_shape_info.GetElementType();
    auto shape_vec = type_and_shape_info.GetShape();
    auto memory_info = output.GetTensorMemoryInfo();
    MLDataType data_type = DataTypeImpl::TensorTypeFromONNXEnum(type);

    output_tensors_[index] = std::make_unique<Tensor>(data_type,
                                                      TensorShape{shape_vec},
                                                      const_cast<void*>(output.GetTensorRawData()),
                                                      OrtMemoryInfo{
                                                          memory_info.GetAllocatorName(),
                                                          memory_info.GetAllocatorType(),
                                                          OrtDevice{
                                                              static_cast<OrtDevice::DeviceType>(memory_info.GetDeviceType()),
                                                              static_cast<OrtDevice::MemoryType>(memory_info.GetMemoryType()),
                                                              static_cast<OrtDevice::VendorId>(memory_info.GetVendorId()),
                                                              static_cast<OrtDevice::DeviceId>(memory_info.GetDeviceId()),
                                                          },
                                                          memory_info.GetMemoryType()});
    return output_tensors_[index].get();
  }
  Tensor* Output(int index, const std::vector<int64_t>& shape) {
    return Output(index, TensorShape{shape});
  }
  Tensor* Output(int index, const std::initializer_list<int64_t>& shape) {
    return Output(index, TensorShape{shape});
  }
  [[nodiscard]] Status GetTempSpaceCPUAllocator(AllocatorPtr* output) const {
    return Status::OK();  // TODO(fs-eire): Implement GetTempSpaceCPUAllocator
  }
  [[nodiscard]] Status GetTempSpaceAllocator(AllocatorPtr* output) const {
    return Status::OK();  // TODO(fs-eire): Implement GetTempSpaceAllocator
  }
  size_t InputCount() const {
    return context_.GetInputCount();
  }
  size_t OutputCount() const {
    return context_.GetOutputCount();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpKernelContext);
  Ort::KernelContext context_;
  mutable std::vector<std::unique_ptr<Tensor>> input_tensors_;
  std::vector<std::unique_ptr<Tensor>> output_tensors_;
};

struct OpKernel {
  explicit OpKernel(const OpKernelInfo& info) : cache_{info.GetKernelInfo()},
                                                op_kernel_info_{info.GetKernelInfo(), &cache_} {}
  virtual ~OpKernel() {}

  const Node& Node() const {
    return cache_.node;
  }
  const OpKernelInfo& Info() const {
    return op_kernel_info_;
  }

  virtual Status Compute(OpKernelContext* p_op_kernel_context) const = 0;
  virtual Status PrePack(const Tensor& tensor,
                         int input_idx,
                         AllocatorPtr alloc,
                         /*out*/ bool& is_packed,
                         /*out*/ PrePackedWeights* prepacked_weights) {
    // TODO(fs-eire): implement PrePack
    is_packed = false;
    return Status::OK();
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpKernel);
  KernelInfoCache cache_;
  OpKernelInfo op_kernel_info_;
};

struct KernelImpl : OrtKernelImpl {
  explicit KernelImpl(std::unique_ptr<OpKernel> impl)
      : OrtKernelImpl{}, impl_(std::move(impl)) {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL ComputeImpl(_In_ OrtKernelImpl* this_ptr,
                                             _In_ OrtKernelContext* context) noexcept {
    // Implement the compute logic here, possibly delegating to impl_
    OpKernelContext ctx{context};
    Status status = static_cast<KernelImpl*>(this_ptr)->impl_->Compute(&ctx);
    if (status.IsOK()) {
      return nullptr;
    } else {
      return Ort::Status{status.ErrorMessage().c_str(), static_cast<OrtErrorCode>(status.Code())}.release();
    }
  }

  static void ORT_API_CALL ReleaseImpl(_In_ OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<KernelImpl*>(this_ptr);
  }

  ~KernelImpl() = default;

 private:
  std::unique_ptr<OpKernel> impl_;
};

}  // namespace detail
}  // namespace ep
}  // namespace onnxruntime
