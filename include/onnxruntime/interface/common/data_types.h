// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

namespace interface {

enum TensorDataType {
  float_tp = 0,
  double_tp,
  int8_tp,
  uint8_tp,
  int16_tp,
  uint16_tp,
  int32_tp,
  uint32_tp,
  int64_tp,
  uint64_tp,
  bool_tp,
  uknownn_tp,
};

}

}  // namespace onnxruntime