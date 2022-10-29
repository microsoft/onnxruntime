#pragma once
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx/onnx_pb.h"

namespace onnxruntime {

struct FunctionTemplate {
  // The generated schema for the function proto
  // ORT rely on it to run type/shape inference
  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
  // reference to the function proto in local function
  const ONNX_NAMESPACE::FunctionProto* onnx_func_proto_;
};

}  // namespace onnxruntime
