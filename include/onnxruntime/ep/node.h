// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/pch.h instead."
#endif

#include "api.h"

namespace onnxruntime {
namespace ep {
namespace detail {

/// <summary>
/// </summary>
struct Node {
  explicit Node(const OrtKernelInfo* kernel_info) : kernel_info_{kernel_info} {}
  /** Gets the Node's name. */
  const std::string Name() const noexcept {
    return kernel_info_.GetNodeName();
  }

  /** Gets the Node's operator type. */
  const std::string OpType() const noexcept {
    return kernel_info_.GetOperatorType();
  }

  /** Gets the since version of the operator. */
  int SinceVersion() const noexcept {
    return kernel_info_.GetSinceVersion();
  }

 private:
  const Ort::ConstKernelInfo kernel_info_;
};

}  // namespace detail
}  // namespace ep
}  // namespace onnxruntime
