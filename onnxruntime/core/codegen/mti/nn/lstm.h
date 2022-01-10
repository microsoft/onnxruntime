// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

// A bubble now. But don't remove it
// TODO: refactor the LSTMcell building to a tvm function
//       and move it here

namespace onnxruntime {
namespace tvm_codegen {

struct LSTMAttributes {
  LSTMAttributes(int64_t hidden_size_p) : hidden_size(hidden_size_p) {}
  int64_t hidden_size;
};

void LSTM_cell(
    const LSTMAttributes& lstm_attrs,
    const tvm::te::Tensor& X,
    const tvm::te::Tensor& W,
    const tvm::te::Tensor& R,
    const tvm::te::Tensor& B,
    bool has_B,
    const tvm::te::Tensor& prev_H,
    const tvm::te::Tensor& prev_C,
    const tvm::te::Tensor& P,
    bool has_P,
    tvm::te::Tensor& Y_h,
    tvm::te::Tensor& Y_c);

}  // namespace tvm_codegen
}  // namespace onnxruntime
