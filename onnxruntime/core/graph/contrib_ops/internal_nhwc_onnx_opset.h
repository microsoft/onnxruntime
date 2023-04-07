// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnx/defs/schema.h"

namespace onnxruntime {
namespace internal_nhwc_onnx {

// Schemas for ops that are NHWC versions of ONNX operators. They are created by the layout transformer by converting
// the relevant input/outputs of a node between NCHW and NHWC, and moving the node to the kMSInternalNHWCDomain domain.
// The schemas are a copy of the ONNX versions, but input 0 and output 0 will be in NHWC format.
class OpSet_Internal_NHWC_ONNX {
 public:
  static void ForEachSchema(const std::function<void(ONNX_NAMESPACE::OpSchema&&)>& fn);
};

}  // namespace internal_nhwc_onnx
}  // namespace onnxruntime
