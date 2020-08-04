// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "onnx/shape_inference/implementation.h"
#include <onnx/onnx_pb.h>
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {
inline void createValueInfo(
    ValueInfoProto& value_info,
    const std::string& name,
    std::vector<int64_t> dim_values) {
  value_info.set_name(name);
  TypeProto* type = value_info.mutable_type();
  TypeProto_Tensor* tensor_type = type->mutable_tensor_type();
  tensor_type->set_elem_type(TensorProto_DataType_FLOAT);
  TensorShapeProto* shape = tensor_type->mutable_shape();
  for (int64_t& value : dim_values)
    shape->add_dim()->set_dim_value(value);
}

void checkShapeEquality(TensorShapeProto* shape1, TensorShapeProto* shape2) {
  EXPECT_NE(shape1, nullptr);
  EXPECT_NE(shape2, nullptr);
  if ((shape1 != nullptr) && (shape2 != nullptr)) {
    EXPECT_EQ(shape1->dim_size(), shape2->dim_size()) << "Shapes do not have same rank";
    auto min_dims = std::min(shape1->dim_size(), shape2->dim_size());
    for (int i = 0; i < min_dims; ++i) {
      auto dim1 = shape1->dim(i);
      auto dim2 = shape2->dim(i);
      EXPECT_EQ(dim1.has_dim_value(), dim2.has_dim_value());
      if (dim1.has_dim_value()) {
        EXPECT_EQ(dim1.dim_value(), dim2.dim_value());
      }
      EXPECT_EQ(dim1.has_dim_param(), dim2.has_dim_param());
      if (dim1.has_dim_param()) {
        EXPECT_EQ(dim1.dim_param(), dim2.dim_param());
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime