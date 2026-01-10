// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/array_feature_extractor.h"

/**
https://github.com/onnx/onnx/blob/main/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(ArrayFeatureExtractor)
    .SetDomain("ai.onnx.ml")
    .SetDoc(R"DOC(
    Select a subset of the data X based on the indices provided Y.
)DOC")
    .Input(0, "X", "Data to be selected", "T")
    .Input(
        1,
        "Y",
        "The index values to select as a int64 tensor",
        "tensor(int64)")
    .Output(0, "Z", "Selected output data as an array", "T")
    .TypeConstraint(
        "T",
        {"tensor(float)",
         "tensor(double)",
         "tensor(int64)",
         "tensor(int32)",
         "tensor(string)"},
        "allowed types.");
*/
using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {
#define REG_ARRAYFEATUREEXTRACTOR(in_type)                                            \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                  \
      ArrayFeatureExtractor,                                                          \
      1,                                                                              \
      in_type,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<in_type>()), \
      ArrayFeatureExtractorOp<in_type>);

REG_ARRAYFEATUREEXTRACTOR(float);
REG_ARRAYFEATUREEXTRACTOR(double);
REG_ARRAYFEATUREEXTRACTOR(int32_t);
REG_ARRAYFEATUREEXTRACTOR(int64_t);
REG_ARRAYFEATUREEXTRACTOR(string);

template <typename T>
ArrayFeatureExtractorOp<T>::ArrayFeatureExtractorOp(const OpKernelInfo& info)
    : OpKernel(info) {
}

template <typename T>
common::Status ArrayFeatureExtractorOp<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  const size_t x_num_dims = x_shape.NumDimensions();
  const T* x_data = X.Data<T>();

  if (x_num_dims == 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid argument: X input has empty dimensions.");
  }

  const int64_t stride = x_shape[x_num_dims - 1];

  const Tensor& Y = *context->Input<Tensor>(1);
  const TensorShape& y_shape = Y.Shape();
  const auto* y_data = Y.Data<int64_t>();
  const int64_t num_indices = y_shape.Size();

  // validate Y
  if (num_indices == 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid Y argument: num_indices = 0");
  }

  for (int64_t i = 0; i < num_indices; ++i) {
    if (y_data[i] >= stride) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Invalid Y argument: index is out of range: Y[", i, "] (", y_data[i], ") >=", stride);
    }
  }

  const TensorShape z_shape = [num_indices, x_num_dims, &x_shape]() {
    if (x_num_dims == 1) {
      // special case: for 1D input, return {1, num_indices} for backwards compatibility
      return TensorShape{1, num_indices};
    }
    TensorShape shape{x_shape};
    shape[x_num_dims - 1] = num_indices;
    return shape;
  }();
  Tensor* Z = context->Output(0, z_shape);
  T* z_data = Z->MutableData<T>();

  const int64_t x_size_until_last_dim = x_shape.SizeToDimension(x_num_dims - 1);
  for (int64_t i = 0; i < x_size_until_last_dim; ++i) {
    for (int64_t j = 0; j < num_indices; ++j) {
      *z_data++ = x_data[y_data[j]];
    }
    x_data += stride;
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
