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

  /** Gets the Node's domain. */
  std::string Domain() const {
    return kernel_info_.GetOperatorDomain();
  }

  /** Gets the since version of the operator. */
  int SinceVersion() const noexcept {
    return kernel_info_.GetOperatorSinceVersion();
  }

  /** Gets the number of outputs. */
  size_t OutputCount() const noexcept {
    return kernel_info_.GetOutputCount();
  }

  /** Gets whether an output exists or is an omitted optional output. */
  bool OutputExists(size_t index) const {
    // KernelInfo_GetOutputName returns an empty string for omitted optional
    // outputs, which lets adapter consumers mirror NodeArg::Exists() without
    // pulling in full NodeArg metadata.
    return index < OutputCount() && !kernel_info_.GetOutputName(index).empty();
  }

 private:
  const Ort::ConstKernelInfo kernel_info_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
