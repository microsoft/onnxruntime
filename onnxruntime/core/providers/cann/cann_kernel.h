// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_inc.h"
#include "core/providers/cann/cann_call.h"
#include "core/providers/cann/cann_execution_provider.h"
#include "core/providers/cann/cann_fwd.h"
#include "core/providers/cann/cann_utils.h"

namespace onnxruntime {
namespace cann {

class CannKernel : public OpKernel {
 public:
  explicit CannKernel(const OpKernelInfo& info)
      : OpKernel(info),
        provider_(const_cast<CANNExecutionProvider*>(
            static_cast<const CANNExecutionProvider*>(info.GetExecutionProvider()))) {}

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);

    if (s.IsOK()) {
      auto err = aclGetRecentErrMsg();
      if (err != nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CANN error", err);
      }
    }

    return s;
  }

  virtual Status ComputeInternal(OpKernelContext* p_op_kernel_context) const = 0;

  inline aclrtStream Stream() const { return static_cast<aclrtStream>(provider_->GetComputeStream()); }

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    return provider_->GetScratchBuffer<T>(count_or_bytes);
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBufferOnCANNPinned(size_t count_or_bytes) const {
    return provider_->GetScratchBufferOnCANNPinned<T>(count_or_bytes);
  }

  template <typename T>
  inline Status Fill(Tensor* y, void* addr) const {
    return provider_->Fill<T>(y, addr);
  }

  template <typename T>
  inline Status Broadcast(const Tensor* x, Tensor* y, void* addr) const {
    return provider_->Broadcast<T>(x, y, addr);
  }

 protected:
  inline Status CopyTensor(const Tensor& src, Tensor& dst) const {
    return Info().GetDataTransferManager().CopyTensor(src, dst);
  }

 private:
  CANNExecutionProvider* provider_;
};

}  // namespace cann
}  // namespace onnxruntime
