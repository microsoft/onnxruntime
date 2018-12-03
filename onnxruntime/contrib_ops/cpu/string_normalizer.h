// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

#include <locale>
#include <string>
#include <unordered_set>

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
  bool is_case_sensitive_;
  CaseAction casechangeaction_;
  CaseAction compare_caseaction_;  // used for case-insensitive compare
  std::locale locale_;             // needed for upper/lowercasing actions and case insensitive compare
  // Either if these are populated but not both
  std::unordered_set<std::string> stopwords_;
  std::unordered_set<std::wstring> wstopwords_;
};

}  // namespace contrib
}  // namespace onnxruntime
