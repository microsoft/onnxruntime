// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/pch.h instead."
#endif

#include "api.h"

namespace onnxruntime {
class IExecutionProvider;
}

namespace onnxruntime {
namespace ep {
namespace detail {

class Ep : public OrtEp {
 protected:
  explicit Ep(IExecutionProvider* impl) : OrtEp{}, impl_(impl) {}

 public:
  IExecutionProvider* impl_;
};

}  // namespace detail
}  // namespace ep
}  // namespace onnxruntime
