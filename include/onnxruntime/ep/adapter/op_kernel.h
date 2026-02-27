// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/tensor.h"

#include "node.h"
#include "op_kernel_info.h"
#include "tensor_helper.h"

namespace onnxruntime {
struct PrePackedWeights;
struct TensorShape;
}  // namespace onnxruntime

namespace onnxruntime {
namespace ep {
namespace adapter {

struct OpKernelContext;

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::OpKernel`.
/// </summary>
struct OpKernel {
  explicit OpKernel(const OpKernelInfo& info) : op_kernel_info_{info} {}
  virtual ~OpKernel() {}

  Node Node() const {
    return op_kernel_info_.node();
  }
  const OpKernelInfo& Info() const {
    return op_kernel_info_;
  }

  virtual Status CreateControlFlowKernelImpl(const OrtKernelInfo* /*info*/, OrtKernelImpl** /*impl*/) {
    return Status::OK();
  }

  virtual Status Compute(OpKernelContext* p_op_kernel_context) const = 0;
  virtual Status PrePack(const Tensor& /*tensor*/,
                         int /*input_idx*/,
                         AllocatorPtr /*alloc*/,
                         /*out*/ bool& is_packed,
                         /*out*/ PrePackedWeights* /*prepacked_weights*/) {
    is_packed = false;
    return Status::OK();
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpKernel);
  OpKernelInfo op_kernel_info_;
};

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::OpKernelContext`.
/// </summary>
struct OpKernelContext {
  explicit OpKernelContext(OrtKernelContext* context, const OpKernel& op_kernel) : context_{context},
                                                                                   op_kernel_{op_kernel},
                                                                                   constant_input_tensors_{op_kernel.Info().GetConstantInputTensors()} {
    input_tensors_.resize(context_.GetInputCount());
    output_tensors_.resize(context_.GetOutputCount());
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
  const T* Input(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= input_tensors_.size()) {
      return nullptr;
    }
    if (input_tensors_[index].DataRaw() != nullptr) {
      return &input_tensors_[index];
    }

    if (static_cast<size_t>(index) < constant_input_tensors_.size() && constant_input_tensors_[index].DataRaw() != nullptr) {
      return &constant_input_tensors_[index];
    }

    auto input = context_.GetInput(index);
    if (input == nullptr || !input.IsTensor()) {
      return nullptr;
    }

    input_tensors_[index] = CreateTensorFromApiValue(const_cast<OrtValue*>(static_cast<const OrtValue*>(input)));
    return &input_tensors_[index];
  }
  Tensor* Output(int index, const TensorShape& shape) {
    if (index < 0 || static_cast<size_t>(index) >= output_tensors_.size()) {
      return nullptr;
    }
    if (output_tensors_[index].DataRaw() != nullptr) {
      return &output_tensors_[index];
    }

    auto output = context_.GetOutput(index, shape.GetDims().data(), shape.GetDims().size());
    if (output == nullptr) {
      return nullptr;
    }

    output_tensors_[index] = CreateTensorFromApiValue(output);
    return &output_tensors_[index];
  }
  Tensor* Output(int index, const std::vector<int64_t>& shape) {
    return Output(index, TensorShape{shape});
  }
  Tensor* Output(int index, const std::initializer_list<int64_t>& shape) {
    return Output(index, TensorShape{shape});
  }
  [[nodiscard]] Status GetTempSpaceCPUAllocator(AllocatorPtr* output) const {
    return static_cast<const Ep*>(op_kernel_.Info().GetKernelInfo().GetEp())->GetTempSpaceCPUAllocator(output);
  }
  [[nodiscard]] Status GetTempSpaceAllocator(AllocatorPtr* output) const {
    return static_cast<const Ep*>(op_kernel_.Info().GetKernelInfo().GetEp())->GetTempSpaceAllocator(output);
  }
  int InputCount() const {
    return static_cast<int>(input_tensors_.size());
  }
  int OutputCount() const {
    return static_cast<int>(output_tensors_.size());
  }
  bool GetUseDeterministicCompute() const {
    // TODO(fs-eire): Implement GetUseDeterministicCompute().
    return false;
  }

  void* GetGPUComputeStream() const {
    return context_.GetGPUComputeStream();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpKernelContext);
  Ort::KernelContext context_;
  const OpKernel& op_kernel_;
  const std::vector<Tensor>& constant_input_tensors_;
  mutable std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
};

/// <summary>
/// A bridge class between `onnxruntime::ep::adapter::OpKernel` and `::OrtKernelImpl`.
/// </summary>
struct KernelImpl : OrtKernelImpl {
  explicit KernelImpl(std::unique_ptr<OpKernel> impl)
      : OrtKernelImpl{}, impl_(std::move(impl)) {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = PrePackWeightImpl;
  }

 private:
  static OrtStatus* ORT_API_CALL ComputeImpl(_In_ OrtKernelImpl* this_ptr,
                                             _In_ OrtKernelContext* context) noexcept {
    Status status;
    ORT_TRY {
      const auto* kernel_impl = static_cast<KernelImpl*>(this_ptr)->impl_.get();
      OpKernelContext ctx{context, *kernel_impl};
      status = kernel_impl->Compute(&ctx);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (status.IsOK()) {
      return nullptr;
    } else {
      return Ort::Status{status.ErrorMessage().c_str(), static_cast<OrtErrorCode>(status.Code())}.release();
    }
  }

  static void ORT_API_CALL ReleaseImpl(_In_ OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<KernelImpl*>(this_ptr);
  }

  static OrtStatus* ORT_API_CALL PrePackWeightImpl(_In_ OrtKernelImpl* this_ptr,
                                                   _In_ const OrtValue* weight,
                                                   int input_index,
                                                   _In_ OrtAllocator* /* allocator */,
                                                   _In_opt_ OrtSharedPrePackedWeightCache* /* prepacked_weight_cache */,
                                                   _Out_ bool* is_packed) noexcept {
    Status status;
    ORT_TRY {
      auto* kernel_impl = static_cast<KernelImpl*>(this_ptr)->impl_.get();
      const auto tensor = CreateTensorFromApiValue(const_cast<OrtValue*>(weight));
      status = kernel_impl->PrePack(tensor, input_index, AllocatorPtr{}, *is_packed, nullptr);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
    if (!status.IsOK()) {
      return Ort::Status{status.ErrorMessage().c_str(), static_cast<OrtErrorCode>(status.Code())}.release();
    }
    return nullptr;
  }

  ~KernelImpl() = default;

 private:
  std::unique_ptr<OpKernel> impl_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
