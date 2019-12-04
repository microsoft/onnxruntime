// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

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
    const tvm::Tensor& X,
    const tvm::Tensor& W,
    const tvm::Tensor& R,
    const tvm::Tensor& B,
    bool has_B,
    const tvm::Tensor& prev_H,
    const tvm::Tensor& prev_C,
    const tvm::Tensor& P,
    bool has_P,
    tvm::Tensor& Y_h,
    tvm::Tensor& Y_c);

}  // namespace tvm_codegen
}  // namespace onnxruntime
