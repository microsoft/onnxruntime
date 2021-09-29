// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_selector_action_transformer.h"

namespace onnxruntime {

class NNAPIQDQSelectorActionTransformer : public NNAPISelectorActionTransformer {
 public:
  NNAPIQDQSelectorActionTransformer();

};

}  // namespace onnxruntime
