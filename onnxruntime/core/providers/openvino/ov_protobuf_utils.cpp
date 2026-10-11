// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_protobuf_utils.h"

#include <cstring>

#include "core/graph/onnx_protobuf.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace openvino_ep {
float get_float_initializer_data(const void* initializer) {
  const auto* tp = reinterpret_cast<const ONNX_NAMESPACE::TensorProto*>(initializer);
  ORT_ENFORCE((tp->has_data_type() && (tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));

  // A FLOAT scalar/tensor may store its value either in the typed float_data
  // field or in raw_data. Indexing float_data(0) when it is empty is undefined
  // behavior, so pick the field that actually holds the data.
  if (tp->float_data_size() > 0) {
    return tp->float_data(0);
  }

  ORT_ENFORCE(tp->has_raw_data() && tp->raw_data().size() >= sizeof(float),
              "FLOAT initializer has neither float_data nor sufficient raw_data to read a value");
  float value;
  std::memcpy(&value, tp->raw_data().data(), sizeof(float));
  return value;
}
void set_float_initializer_data(const void* initializer, float data) {
  auto* tp = (ONNX_NAMESPACE::TensorProto*)(initializer);
  ORT_ENFORCE((tp->has_data_type() && (tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));

  // Mirror get_float_initializer_data: write back into whichever storage the
  // initializer actually uses. set_float_data(0, data) on an empty float_data
  // field is an out-of-bounds write.
  if (tp->float_data_size() > 0) {
    tp->set_float_data(0, data);
    return;
  }

  ORT_ENFORCE(tp->has_raw_data() && tp->raw_data().size() >= sizeof(float),
              "FLOAT initializer has neither float_data nor sufficient raw_data to write a value");
  tp->set_raw_data(&data, sizeof(float));
}
}  // namespace openvino_ep
}  // namespace onnxruntime
