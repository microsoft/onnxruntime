// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>

#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {

namespace contrib {

class LinalgInv : public OpKernel {
 public:
   LinalgInv(const OpKernelInfo& info)
      : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

private:
  bool left_ = true;
};

}  // namespace contrib
}  // namespace onnxruntime
