// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::KernelDef`.
/// </summary>
class KernelDef {
 public:
  explicit KernelDef(const OrtKernelInfo* kernel_info) : kernel_info_{kernel_info} {}

  const std::string OpName() const {
    return kernel_info_.GetNodeName();
  }

  const std::string Domain() const {
    return kernel_info_.GetOperatorDomain();
  }

 private:
  const Ort::ConstKernelInfo kernel_info_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
