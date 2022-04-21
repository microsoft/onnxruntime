// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnx/defs/schema.h"
#include "xnnpack_onnx_schema.h"

namespace onnxruntime {
namespace xnnpack {
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackConvolution2d);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackDepthwiseConvolution2d);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackMaxPooling2d);

static std::vector<ONNX_NAMESPACE::OpSchema> GetSchemas() {
  return std::vector<ONNX_NAMESPACE::OpSchema>{
      GetOpSchema<class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackConvolution2d)>(),
      GetOpSchema<class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackDepthwiseConvolution2d)>(),
      GetOpSchema<class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackMaxPooling2d)>()};
}

}  // namespace xnnpack
}  // namespace onnxruntime