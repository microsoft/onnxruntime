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
class LinalgSVD : public OpKernel {
 public:
  LinalgSVD(const OpKernelInfo& info)
      : OpKernel(info) {
    int64_t full_matrices;
    ORT_ENFORCE(info.GetAttr<int64_t>("full_matrices", &full_matrices).IsOK());
    full_matrices_ = full_matrices != 0;
     //int64_t compute_uv;
     //ORT_ENFORCE(info.GetAttr<int64_t>("compute_uv", &compute_uv).IsOK());
     //compute_uv_ = compute_uv != 0;
  }

  Status Compute(OpKernelContext* context) const override;

private:
  bool full_matrices_ = true;
};

}  // namespace contrib
}  // namespace onnxruntime
