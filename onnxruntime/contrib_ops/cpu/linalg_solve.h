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

template <typename T>
class LinalgSolve : public OpKernel {
 public:
   LinalgSolve(const OpKernelInfo& info)
      : OpKernel(info) {
     int64_t left;
     ORT_ENFORCE(info.GetAttr<int64_t>("left", &left).IsOK());
     left_ = left != 0;
   }

  Status Compute(OpKernelContext* context) const override;

private:
  bool left_ = true;
};

}  // namespace contrib
}  // namespace onnxruntime
