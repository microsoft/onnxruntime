// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <unordered_set>

inline bool axes_has_dupes(const std::vector<int64_t>& axes) {
    if (axes.size() == 0)
       return false; 
   
    std::unordered_set<int64_t> elements;
    for(const auto& axis : axes) {
       if (elements.find(axis) != elements.end())
           return true;       
    }
    
    return false;
}

namespace onnxruntime {
class Reverse final : public OpKernel {
 public:
  explicit Reverse(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
     auto has_axes = op_kernel_info.GetAttrs("axes", attr_axes_).IsOK();
     ORT_ENFORCE(!has_axes || !axes_has_dupes(attr_axes_) , "axes attribute has duplicates, this is not accordance with Reverse op spec");
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  std::vector<int64_t> attr_axes_;
};
}  // namespace onnxruntime
