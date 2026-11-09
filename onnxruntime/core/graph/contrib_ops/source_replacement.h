// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/contrib_ops/source_replacement_policy.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
#include "core/optimizer/pattern_rewriter.h"

namespace onnxruntime::contrib {

enum class NumericalGuarantee {
  NonNaNBitsAndNaNClassification,
  FiniteInputAnalyticalBudgetAndExactState,
};

struct NumericalProfile {
  std::string name;
  std::string scope;
  NumericalGuarantee guarantee;
};

enum class InitializerTransformation { None,
                                       OwnedFlattenedBiasCopy };

// Function-body-like ownership, using the existing source/replacement DAG and port mappings.
// No kernel lookup or registration requirement: compiled EPs can consume the representation,
// but must separately prove backend target support before offering a replacement.
struct SourceBody {
  rewrite::Rule rule;
  NumericalProfile numerical;
  InitializerTransformation initializer_transformation = InitializerTransformation::None;
  std::string transformation_description;
};

}  // namespace onnxruntime::contrib
#endif
