// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace contrib {

class QLinearGlobalAveragePool final : public OpKernel {
 public:
  QLinearGlobalAveragePool(const OpKernelInfo& info) : OpKernel(info) {
    int64_t nchw_layout;
    ORT_ENFORCE(info.GetAttr("nchw", &nchw_layout).IsOK());
    storage_order_ = nchw_layout ? StorageOrder::NCHW : StorageOrder::NHWC;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  StorageOrder storage_order_;
};

}  // namespace contrib
}  // namespace onnxruntime
