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

enum TensorSeqDataType {
  float_seq = 0,
  double_seq,
  int8_seq,
  uint8_seq,
  int16_seq,
  uint16_seq,
  int32_seq,
  uint32_seq,
  int64_seq,
  uint64_seq,
  bool_seq,
  uknownn_seq,
};

}

}  // namespace onnxruntime