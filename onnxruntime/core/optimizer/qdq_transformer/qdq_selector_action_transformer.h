// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {
/**
Transformer that fuses QDQ and fp32 ops into quantized ops. 
*/
class QDQSelectorActionTransformer : public SelectorActionTransformer {
 public:
  QDQSelectorActionTransformer();
};

}  // namespace onnxruntime
