// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_protobuf_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace openvino_ep {
float get_float_initializer_data(const void* initializer) {
  const auto* tp = reinterpret_cast<const ONNX_NAMESPACE::TensorProto*>(initializer);
  ORT_ENFORCE((tp->has_data_type() && (tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));
  // ORT_ENFORCE(initializer.dims_size() == 1);
  return tp->float_data(0);
}
void set_float_initializer_data(const void* initializer, float data) {
  auto* tp = (ONNX_NAMESPACE::TensorProto*)(initializer);
  ORT_ENFORCE((tp->has_data_type() && (tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));
  // ORT_ENFORCE(initializer.dims_size() == 1);
  tp->set_float_data(0, data);
}
}  // namespace openvino_ep
}  // namespace onnxruntime
