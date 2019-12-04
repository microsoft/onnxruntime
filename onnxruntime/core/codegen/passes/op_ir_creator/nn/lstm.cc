// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/nn/lstm.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// In the cell computation, we don't have the "direction" dimention and sequence dimension,
// which have been processed outside of the cell.
// Here we implement an LTSM cell.
// For those args (inputs/outputs) of hidden states we put AFTER regular args (inputs/outputs)
// with a pre-defined order
// In a LSTM, the order is H and then C.
// Ouputs of LSTM is Y_h and then Y_c
Status GENERIC_OP_IR_CREATOR_CLASS(LSTM)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  std::string direction_attr;
  ORT_RETURN_IF_ERROR(attrs.GetAttr("direction", &direction_attr));
  int64_t hidden_size;
  ORT_RETURN_IF_ERROR(attrs.GetAttr("hidden_size", &hidden_size));

  // input tensor with shape [seq_length, batch_size, input_size]
  const tvm::Tensor& X = inputs[0];  // input tensor with shape [seq_length, batch_size, input_size]
  const tvm::Tensor& W = inputs[1];  // weights tensor with shape [4*hidden_size, input_size]
  const tvm::Tensor& R = inputs[2];  // recurrence tensor with shape [4*hidden_size, hidden_size]
  const tvm::Tensor& B = inputs[3];  // optional bias tensor with shape [8*hidden_size]
  bool has_B = node.InputDefs()[3]->Exists();

  // Unsupported the 4th inputs
  // optional tensor specifying sequence lengths in a batch, shape: [batch_size]
  // const tvm::Tensor* seq_len = inputs[4] ? &inputs[4]->tensor : nullptr;

  const tvm::Tensor& prev_H = inputs[5];  // optional initial H, shape: [batch_size, hidden_size]
  const tvm::Tensor& prev_C = inputs[6];  // optional initial C, shape: [batch_size, hidden_size]

  const tvm::Tensor& P = inputs[7];  // optional peepholes tensor with shape [3*hidde_size]
  bool has_P = node.InputDefs()[7]->Exists();

  tvm::Tensor Y_h;  // shape: [batch_size, hidden_size]
  tvm::Tensor Y_c;  // shape: [batch_size, hidden_size]
  LSTMAttributes lstm_attrs(hidden_size);
  LSTM_cell(lstm_attrs, X, W, R, B, has_B, prev_H, prev_C, P, has_P, Y_h, Y_c);

  // Since we only generate lstm cell, lstm's states need to be always outputs,
  // regardless whethere they are skipped or not.
  // The skipped trailing outputs need to be handled by Execution
  outputs.push_back(Y_h);
  outputs.push_back(Y_c);

  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
