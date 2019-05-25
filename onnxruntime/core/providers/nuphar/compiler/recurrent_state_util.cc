// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "recurrent_state_util.h"
#include "core/framework/op_kernel_info.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/target/ort_tvm_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

static std::pair<tvm::Tensor, tvm::Tensor> CreateOneState(const std::vector<int64_t>& dims,
                                                          HalideIR::Type halide_type,
                                                          const std::string& name) {
  return std::make_pair(tvm::placeholder(ToTvmArray(dims), halide_type, name + "in"),
                        tvm::placeholder(ToTvmArray(dims), halide_type, name + "out"));
}

static void CreateLSTMState(const Node& node,
                            LoopState& loop_states) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
  int64_t hidden_size;
  bool attr_is_ok = attrs.GetAttr("hidden_size", &hidden_size).IsOK();
  ORT_UNUSED_PARAMETER(attr_is_ok);
  ORT_ENFORCE_DEBUG(attr_is_ok);

  // For LSTM, the type of H and C states is the same as X, which is the first input.
  const NodeArg* def = node.InputDefs()[0];
  MLDataType ONNXRUNTIME_data_type = DataTypeImpl::TypeFromProto(*def->TypeAsProto());
  DLDataType dtype = ToTvmDLDataType(ONNXRUNTIME_data_type);
  HalideIR::Type halide_type((halideir_type_code_t)dtype.code, dtype.bits, dtype.lanes);

  int64_t batch_size = ShapeValue(def, 1);
  // FIXME batch to symbolic

  std::string name_prefix = node.Name() + "_" + node.OpType() + "_";

  // Add H state
  loop_states.push_back(CreateOneState({batch_size, hidden_size}, halide_type, name_prefix + "H_"));
  // Add C State
  loop_states.push_back(CreateOneState({batch_size, hidden_size}, halide_type, name_prefix + "C_"));
}

void CreateRecurrentStates(const Node& node,
                           LoopState& loop_states) {
  if (node.OpType() == "LSTM") {
    CreateLSTMState(node, loop_states);
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
