// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/contrib_ops/ms_schema.h"

// This file contains deprecated ONNX operators that have been removed from ONNX spec, but we still need to keep them
// to maintain backward compatibility. Strictly speaking, this file doesn't define an opset. It only contains a group
// of operators.

namespace onnxruntime {
namespace contrib {
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Affine);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ParametricSoftplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ImageScaler);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Crop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ThresholdedRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DynamicSlice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GivenTensorFill);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Scale);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRUUnit);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GivenTensorFill);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Scale);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GRUUnit);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MeanVarianceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ScaledTanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Affine);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ParametricSoftplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ImageScaler);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Crop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DynamicSlice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ScaledTanh);

class OpSet_ONNX_Deprecated {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Affine)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ParametricSoftplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ImageScaler)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Crop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ThresholdedRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DynamicSlice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GivenTensorFill)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Scale)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRUUnit)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GivenTensorFill)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Scale)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GRUUnit)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MeanVarianceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ScaledTanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Affine)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ParametricSoftplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ImageScaler)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Crop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DynamicSlice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ScaledTanh)>());
  }
};
}  // namespace contrib
}  // namespace onnxruntime
