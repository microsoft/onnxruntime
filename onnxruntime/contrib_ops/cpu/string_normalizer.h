// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

#include <string>
#include <vector>

namespace onnxruntime {
namespace contrib {

class StringNormalizer : public OpKernel {
 public:
  enum CaseAction {
    NONE = 0,
    LOWER = 1,
    UPPER = 2,
  };

  explicit StringNormalizer(const OpKernelInfo& info);
  ~StringNormalizer() = default;

  Status Compute(OpKernelContext* ctx) const override;

 private:
  CaseAction casechangeaction_;
  bool iscasesensitive_;
  std::vector<std::string> stopwords_;
  std::string locale_;  // needed for upper/lowercasing actions and case insensitive compare
};

}  // namespace contrib
}  // namespace onnxruntime
