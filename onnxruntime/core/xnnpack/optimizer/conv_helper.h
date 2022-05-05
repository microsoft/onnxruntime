// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/common.h"

namespace onnxruntime {
// In ONNX, Conv operators and LpPool/MaxPool operators have an auto_pad attribute. And there are 4 different padding
// types:NOTSET, SAME_UPPER, SAME_LOWER and VALID. SAME_LOWER doesn't exist in TF2ONNX converter. In the first version
// of this XNNPack integeration, we focuses on TF/TFLite models only. So SAME_LOWER is not handled.
inline bool IsPaddingTypeSupportedByXNNPack(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET || auto_pad == AutoPadType::VALID || auto_pad == AutoPadType::SAME_UPPER;
}
}  // namespace onnxruntime