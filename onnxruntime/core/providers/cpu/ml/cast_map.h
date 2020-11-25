// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/ml/ml_common.h"

namespace onnxruntime {
namespace ml {

class CastMap final : public OpKernel {
 public:
  CastMap(const OpKernelInfo& info) : OpKernel(info) {
    std::string attr;

    ORT_ENFORCE(info.GetAttr<std::string>("cast_to", &attr).IsOK());
    cast_to_ = MakeCast(attr);

    ORT_ENFORCE(info.GetAttr<std::string>("map_form", &attr).IsOK());
    map_form_ = MakePack(attr);

    // ignore if not found as we fall back to the default of 1
    ORT_ENFORCE(info.GetAttr<int64_t>("max_map", &max_map_).IsOK());

    ORT_ENFORCE(map_form_ != PACK_MAP::SPARSE || max_map_ > 0, "max_map must be > 0 if map_form is SPARSE");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename TFrom, typename TTo>
  Status ComputeImpl(OpKernelContext& ctx, TTo pad_value) const;

  CAST_TO cast_to_;
  PACK_MAP map_form_;

  int64_t max_map_;
};

}  // namespace ml
}  // namespace onnxruntime
