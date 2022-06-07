// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <vector>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {
template <typename AttrType, typename TargetType>
class DictVectorizerOp final : public OpKernel {
 public:
  DictVectorizerOp(const OpKernelInfo& info) : OpKernel(info) {
    //In some stupid models, the vocabulary could have duplicated elements.
    //We must support that, otherwise some tests will be break.
    ORT_ENFORCE(info.GetAttrs(std::is_same<AttrType, std::string>::value ? "string_vocabulary" : "int64_vocabulary", vocabulary_).IsOK());
  }
  common::Status Compute(OpKernelContext* ctx) const override {
    const auto* map = ctx->Input<std::map<AttrType, TargetType> >(0);
    auto* Y = ctx->Output(0, {1, static_cast<int64_t>(vocabulary_.size())});
    auto* y_data = Y->template MutableData<TargetType>();
    for (size_t i = 0, end = vocabulary_.size(); i < end; ++i) {
      auto index = map->find(vocabulary_[i]);
      if (index != map->end()) {
        *y_data++ = index->second;
      } else {
        //Any keys not present in the input dictionary, will be zero in the output array
        *y_data++ = TargetType();
      }
    }
    return Status::OK();
  }

  std::vector<AttrType> vocabulary_;
};

}  // namespace ml

}  // namespace onnxruntime
