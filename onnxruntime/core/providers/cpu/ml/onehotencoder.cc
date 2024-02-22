// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/onehotencoder.h"
/**
https://github.com/onnx/onnx/blob/main/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(OneHotEncoder)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
Replace the inputs with an array of ones and zeros, where the only
one is the zero-based category that was passed in.  The total category count
will determine the length of the vector. For example if we pass a
tensor with a single value of 4, and a category count of 8, the
output will be a tensor with 0,0,0,0,1,0,0,0 .

This operator assumes every input in X is of the same category set
(meaning there is only one category count).

If the input is a tensor of float, int32, or double, the data will be cast
to int64s and the cats_int64s category list will be used for the lookups.
)DOC")
.Input(0, "X", "Data to be encoded", "T")
.Output(0, "Y", "encoded output data", "tensor(float)")
.TypeConstraint("T", { "tensor(string)", "tensor(int64)","tensor(int32)", "tensor(float)","tensor(double)" }, " allowed types.")
.Attr("cats_int64s", "list of categories, ints", AttributeProto::INTS, OPTIONAL)
.Attr("cats_strings", "list of categories, strings", AttributeProto::STRINGS, OPTIONAL)
.Attr(
"zeros",
"if true and category is not present, will return all zeros, if false and missing category, operator will return false",
AttributeProto::INT,
OPTIONAL);
*/
using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

#define REG_KERNEL(TYPE)                                                           \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                               \
      OneHotEncoder,                                                               \
      1,                                                                           \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      OneHotEncoderOp<TYPE>);

REG_KERNEL(int64_t);
REG_KERNEL(float);
REG_KERNEL(double);
REG_KERNEL(string);

template <typename T>
OneHotEncoderOp<T>::OneHotEncoderOp(const OpKernelInfo& info) : OpKernel(info), zeros_(info.GetAttrOrDefault<int64_t>("zeros", 1)), num_categories_(0) {
  std::vector<int64_t> tmp_cats_int64s = info.GetAttrsOrDefault<int64_t>("cats_int64s");
  std::vector<std::string> tmp_cats_strings = info.GetAttrsOrDefault<string>("cats_strings");
  ORT_ENFORCE(tmp_cats_int64s.empty() || tmp_cats_strings.empty(),
              "One and only one of the 'cats_*' attributes must be defined");
  if (!tmp_cats_int64s.empty()) {
    num_categories_ = tmp_cats_int64s.size();
    for (size_t idx = 0, end = tmp_cats_int64s.size(); idx < end; ++idx) {
      cats_int64s_[tmp_cats_int64s[idx]] = idx;
    }
  } else {
    num_categories_ = tmp_cats_strings.size();
    for (size_t idx = 0, end = tmp_cats_strings.size(); idx < end; ++idx) {
      cats_strings_[tmp_cats_strings[idx]] = idx;
    }
  }
  ORT_ENFORCE(num_categories_ > 0);
}

template <typename T>
common::Status OneHotEncoderOp<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();

  auto output_shape = input_shape.AsShapeVector();
  output_shape.push_back(num_categories_);

  Tensor* Y = context->Output(0, TensorShape(output_shape));
  auto* y_data = Y->MutableData<float>();
  std::fill_n(y_data, Y->Shape().Size(), 0.0f);

  const auto* x_data = X->Data<T>();
  const auto x_size = input_shape.Size();
  std::unordered_map<int64_t, size_t>::const_iterator idx;
  for (int64_t i = 0; i < x_size; ++i) {
    auto int_idx = cats_int64s_.find(static_cast<int64_t>(x_data[i]));
    if (int_idx != cats_int64s_.cend())
      y_data[i * num_categories_ + int_idx->second] = 1.0f;
    else if (!zeros_)
      return Status(ONNXRUNTIME, FAIL, "Unknown Category and zeros = 0.");
  }
  return Status::OK();
}

template <>
common::Status OneHotEncoderOp<std::string>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();

  std::vector<int64_t> output_shape(input_shape.GetDims().begin(), input_shape.GetDims().end());
  output_shape.push_back(num_categories_);

  Tensor* Y = context->Output(0, TensorShape(output_shape));
  auto* y_data = Y->MutableData<float>();
  std::fill_n(y_data, Y->Shape().Size(), 0.0f);

  const auto* x_data = X->Data<std::string>();
  const auto x_size = input_shape.Size();
  for (int64_t i = 0; i < x_size; ++i) {
    auto str_idx = cats_strings_.find(x_data[i]);
    if (str_idx != cats_strings_.cend())
      y_data[i * num_categories_ + str_idx->second] = 1.0f;
    else if (!zeros_)
      return Status(ONNXRUNTIME, FAIL, "Unknown Category and zeros = 0.");
  }
  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
