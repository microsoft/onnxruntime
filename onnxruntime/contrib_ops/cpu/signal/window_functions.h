// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

#include "core/providers/cpu/signal/window_functions.h"

namespace onnxruntime {
namespace contrib {

class HannWindow final : public VariableOutputDataTypeBase {
 public:
  explicit HannWindow(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class HammingWindow final : public VariableOutputDataTypeBase {
 public:
  explicit HammingWindow(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class BlackmanWindow final : public VariableOutputDataTypeBase {
 public:
  explicit BlackmanWindow(const OpKernelInfo& info) : VariableOutputDataTypeBase(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif
