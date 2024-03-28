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
class LinalgCholesky : public OpKernel {
 public:
   LinalgCholesky(const OpKernelInfo& info)
      : OpKernel(info) {
     int64_t upper;
     ORT_ENFORCE(info.GetAttr<int64_t>("upper", &upper).IsOK());
     upper_ = upper != 0;
  }

  Status Compute(OpKernelContext* context) const override;

private:
  bool upper_ = false;
};

}  // namespace contrib
}  // namespace onnxruntime
