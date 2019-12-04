// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/imputer.h"
#include <cmath>
/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(Imputer)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
Replace imputs that equal replaceValue/s  with  imputeValue/s.
All other inputs are copied to the output unchanged.
This op is used to replace missing values where we know what a missing value looks like.
Only one of imputed_value_floats or imputed_value_int64s should be used.
The size can be 1 element, which will be reused, or the size of the feature set F in input N,F
)DOC")
.Input(0, "X", "Data to be imputed", "T")
.Output(0, "Y", "Imputed output data", "T")
.TypeConstraint(
"T",
{"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
" allowed types.")
.Attr(
"imputed_value_floats",
"value to change to",
AttributeProto::FLOATS,
OPTIONAL)
.Attr(
"replaced_value_float",
"value that needs replacing",
AttributeProto::FLOAT,
0.f)
.Attr(
"imputed_value_int64s",
"value to change to",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"replaced_value_int64",
"value that needs replacing",
AttributeProto::INT,
static_cast<int64_t>(0));
*/
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    Imputer,
    1,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<int64_t>()}),
    ImputerOp);

ImputerOp::ImputerOp(const OpKernelInfo& info) : OpKernel(info),
                                                 imputed_values_float_(info.GetAttrsOrDefault<float>("imputed_value_floats")),
                                                 imputed_values_int64_(info.GetAttrsOrDefault<int64_t>("imputed_value_int64s")) {
  if (!imputed_values_float_.empty() && !info.GetAttr<float>("replaced_value_float", &replaced_value_float_).IsOK())
    ORT_THROW("Expected 'replaced_value_float' attribute since 'imputed_value_floats' is specified");
  if (!imputed_values_int64_.empty() && !info.GetAttr<int64_t>("replaced_value_int64", &replaced_value_int64_).IsOK())
    ORT_THROW("Expected 'replace_value_int64' attribute since 'imputed_values_int64' is specified");
  ORT_ENFORCE(imputed_values_float_.empty() ^ imputed_values_int64_.empty(),
              "Must provide imputed_values_float_ or imputed_values_int64_ but not both.");
}

template <typename T>
common::Status ComputeByType(OpKernelContext* context,
                             T replaced_value,
                             const std::vector<T>& imputed_values) {
  if (imputed_values.empty()) {
    return Status(ONNXRUNTIME, FAIL, "Empty value of imputed values.");
  }

  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  const TensorShape& x_shape = X.Shape();
  auto& dims = x_shape.GetDims();
  if (dims.empty()) {
    return Status(ONNXRUNTIME, FAIL, "Empty input dimensions.");
  }

  const T* x_data = X.template Data<T>();
  size_t x_size = x_shape.Size();
  int64_t stride = dims.size() == 1 ? dims[0] : dims[1];

  Tensor* Y = context->Output(0, x_shape);
  T* y_data = Y->template MutableData<T>();
  if (imputed_values.size() == static_cast<size_t>(stride)) {
    for (size_t i = 0; i < x_size; i++) {
      if (std::isnan(static_cast<float>(x_data[i])) && std::isnan(static_cast<float>(replaced_value))) {
        y_data[i] = imputed_values[i % stride];
      } else if (x_data[i] == replaced_value) {
        y_data[i] = imputed_values[i % stride];
      } else {
        y_data[i] = x_data[i];
      }
    }
  } else {
    for (size_t i = 0; i < x_size; i++) {
      if (std::isnan(static_cast<float>(x_data[i])) && std::isnan(static_cast<float>(replaced_value))) {
        y_data[i] = imputed_values[0];
      } else if (x_data[i] == replaced_value) {
        y_data[i] = imputed_values[0];
      } else {
        y_data[i] = x_data[i];
      }
    }
  }

  return Status::OK();
}

common::Status ImputerOp::Compute(OpKernelContext* context) const {
  const auto* input_tensor_ptr = context->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor_ptr != nullptr);
  auto input_type = input_tensor_ptr->DataType();
  if (utils::IsPrimitiveDataType<float>(input_type)) {
    return ComputeByType<float>(context, replaced_value_float_, imputed_values_float_);
  }
  if (utils::IsPrimitiveDataType<int64_t>(input_type)) {
    return ComputeByType<int64_t>(context, replaced_value_int64_, imputed_values_int64_);
  } else {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid type");
  }
}
}  // namespace ml
}  // namespace onnxruntime
