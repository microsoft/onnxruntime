// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::Node`.
/// </summary>
struct Node {
  explicit Node(const OrtKernelInfo* kernel_info) : kernel_info_{kernel_info} {}
  /** Gets the Node's name. */
  std::string Name() const noexcept {
    return kernel_info_.GetNodeName();
  }

  /** Gets the Node's operator type. */
  std::string OpType() const noexcept {
    return kernel_info_.GetOperatorType();
  }

  /** Gets the since version of the operator. */
  int SinceVersion() const noexcept {
    return kernel_info_.GetOperatorSinceVersion();
  }

 private:
  const Ort::ConstKernelInfo kernel_info_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
