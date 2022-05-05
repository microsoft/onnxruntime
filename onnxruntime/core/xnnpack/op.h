// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <xnnpack.h>

#include <memory>

namespace onnxruntime {
namespace xnnpack {
struct XNNPackOperatorDeleter {
  void operator()(struct xnn_operator* p) const {
    if (p != nullptr) {
      // Ignore returned value because it fails only when xnn pack wasn't initialized
      xnn_delete_operator(p);
    }
  }
};
using XNNPackOperator = std::unique_ptr<struct xnn_operator, XNNPackOperatorDeleter>;
}  // namespace xnnpack
}  // namespace onnxruntime