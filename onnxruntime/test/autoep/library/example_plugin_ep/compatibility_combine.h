// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace example_ep {

// Maps an OrtCompiledModelCompatibility value to an ordinal where a lower rank is a "worse" verdict
// (EP_UNSUPPORTED < EP_SUPPORTED_PREFER_RECOMPILATION < EP_SUPPORTED_OPTIMAL). EP_NOT_APPLICABLE is the identity
// element of the fold and is handled by CombineCompatibility before this is called, so it falls into the
// conservative default below.
inline int RankCompatibility(OrtCompiledModelCompatibility c) {
  switch (c) {
    case OrtCompiledModelCompatibility_EP_UNSUPPORTED:
      return 0;
    case OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION:
      return 1;
    case OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL:
      return 2;
    default:
      // Conservative: treat any unknown/unhandled value as the worst rank so an unexpected value can never be
      // reported as compatible. Update this switch if new OrtCompiledModelCompatibility values are added.
      return 0;
  }
}

// Combines two per-device verdicts following the rule documented for
// OrtEpFactory::ValidateCompiledModelCompatibilityInfo in onnxruntime_ep_c_api.h: EP_NOT_APPLICABLE is a neutral
// identity (skipped), and otherwise the worst verdict wins. This is the reduction an EP folds over its per-device
// verdicts to produce the single verdict the API must return.
inline OrtCompiledModelCompatibility CombineCompatibility(OrtCompiledModelCompatibility acc,
                                                          OrtCompiledModelCompatibility next) {
  if (next == OrtCompiledModelCompatibility_EP_NOT_APPLICABLE) {
    return acc;
  }
  if (acc == OrtCompiledModelCompatibility_EP_NOT_APPLICABLE) {
    return next;
  }

  // Take the verdict with the lower rank, i.e. the worse of the two.
  return RankCompatibility(next) < RankCompatibility(acc) ? next : acc;
}

}  // namespace example_ep
