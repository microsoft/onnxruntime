// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_STRING_TYPE)

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
  bool is_case_sensitive_{true};
  CaseAction case_change_action_{NONE};
  // Hardcoded to LOWER for case-insensitive stopword comparison.
  // Lowercase is used here as a practical fit for the current per-character
  // std::transform-based implementation:
  // - Some characters have no uppercase form or uppercase to multiple characters
  //   (e.g., ß -> SS), which this implementation cannot handle because it
  //   transforms one wchar_t at a time.
  // - Unicode casing can be locale-, context-, and length-dependent, so this
  //   should not be interpreted as full Unicode case folding.
  // The ideal approach would be Unicode case folding (ICU), but that's not
  // warranted for this operator.
  CaseAction compare_caseaction_{LOWER};
  std::string locale_name_;
  // Either if these are populated but not both
  InlinedHashSet<std::string> stopwords_;
  InlinedHashSet<std::wstring> wstopwords_;
};

}  // namespace onnxruntime

#endif  // !defined(DISABLE_STRING_TYPE)
