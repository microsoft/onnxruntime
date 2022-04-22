// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/op_kernel.h"

#include <locale>
#include <string>

namespace onnxruntime {

class StringNormalizer : public OpKernel {
 public:
  enum CaseAction {
    NONE = 0,
    LOWER = 1,
    UPPER = 2,
  };

  explicit StringNormalizer(const OpKernelInfo& info);
  ~StringNormalizer() override = default;

  Status Compute(OpKernelContext* ctx) const override;

 private:
  bool is_case_sensitive_;
  CaseAction case_change_action_;
  CaseAction compare_caseaction_;  // used for case-insensitive compare
  std::string locale_name_;
  // Either if these are populated but not both
  InlinedHashSet<std::string> stopwords_;
  InlinedHashSet<std::wstring> wstopwords_;
};

}  // namespace onnxruntime
