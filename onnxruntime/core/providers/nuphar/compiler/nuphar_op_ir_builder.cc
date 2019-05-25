// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/nuphar_op_ir_builder.h"

#include "core/codegen/common/op_macro.h"
#include "core/providers/nuphar/compiler/recurrent_state_util.h"
#include "core/providers/nuphar/compiler/tvm_initializer.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/target/generic/op_ir_creator/all_ops.h"
#include "core/codegen/target/ort_tvm_utils.h"
#include "core/codegen/target/tvm_ir_builder.h"
#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// Declaration of GetOrCreateInitializer
// GetOrCreateInitializer create tvm::placeholder for a marshalled weight
// with correpsonding data layout transfomration for a weight,
// Note the weight is fed during build
static const tvm::Tensor& GetOrCreateInitializer(const NodeArg* def,
                                                 const Tensor* tensor,
                                                 bool is_sliced,
                                                 NupharCodeGenCtx& ctx_codegen);

// CreateInputPlaceholder create tvm input placeholder (tvm::Tensor)
// NOTE: here we assume axis 0 is sequence
// TODO: add support for sequence not axis 0
static tvm::Tensor CreateInputPlaceholder(const tvm::Array<tvm::Expr>& shape,
                                          HalideIR::Type halide_type,
                                          const std::string& name,
                                          bool is_sliced) {
  return tvm::placeholder(is_sliced && shape.size() > 1 ? SliceShapeFromDimension(shape, 1) : shape, halide_type, name);
}

// CreateInput creats tvm::Tensor of corresponding ORT input
// Inputs are either initializer or regular input placeholder
static bool CreateInput(
    const NodeArg* def,
    tvm::Tensor& input,
    bool initializer_only,
    bool is_sliced,
    NupharCodeGenCtx& ctx_codegen) {
  const Tensor* initialized_tensor = ctx_codegen.GetOrtInitializedTensor(def);
  if (nullptr == initialized_tensor && initializer_only)
    return false;

  ORT_ENFORCE(def->Shape());
  if (nullptr != initialized_tensor) {
    input = GetOrCreateInitializer(def, initialized_tensor, is_sliced, ctx_codegen);
  } else {
    // Handle inputs without initializer
    std::string name = NormalizeNodeArgName(def);
    MLDataType ONNXRUNTIME_data_type = DataTypeImpl::TypeFromProto(*def->TypeAsProto());
    DLDataType dtype = ToTvmDLDataType(ONNXRUNTIME_data_type);
    HalideIR::Type halide_type((halideir_type_code_t)dtype.code, dtype.bits, dtype.lanes);
    tvm::Array<tvm::Expr> shape = ShapeToTvmArray(def, ctx_codegen);

    // Create InputPlaceholder
    // Slice InputPlaceholder if it is asked for.
    input = CreateInputPlaceholder(shape, halide_type, name, is_sliced);
  }
  return true;
}

// GetOrCreateInitializer create tvm::placeholder for a marshalled weight
// with correpsonding data layout transfomration for a weight,
// Note the weight is fed during build
const tvm::Tensor& GetOrCreateInitializer(const NodeArg* def,
                                          const Tensor* tensor,
                                          bool is_sliced,
                                          NupharCodeGenCtx& ctx_codegen) {
  auto info = ctx_codegen.GetInitializerInfo(def->Name());
  ORT_ENFORCE(nullptr != info);

  if (nullptr != info->layout_info) {
    return info->layout_info->marshalled_tensor;
  }

  auto ONNXRUNTIME_data_type = tensor->DataType();
  DLDataType dtype = tvm_codegen::ToTvmDLDataType(ONNXRUNTIME_data_type);
  HalideIR::Type halide_type((halideir_type_code_t)dtype.code, dtype.bits, dtype.lanes);
  std::string name = NormalizeNodeArgName(def);
  auto tvm_shape = ToTvmArray(tensor->Shape().GetDims());
  auto tvm_tensor = CreateInputPlaceholder(tvm_shape, halide_type, name, is_sliced);
  // create the layout info
  info->layout_info = std::make_unique<WeightLayoutInfo>(tvm_tensor);
  return info->layout_info->marshalled_tensor;
}

// CreateOutputs constructs tvm::Tensor with corresponding computation
static Status CreateOutputs(const Node* node,
                            const tvm::Array<tvm::Tensor>& inputs,
                            tvm::Array<tvm::Tensor>& outputs,
                            NupharCodeGenCtx& ctx_codegen) {
  ORT_RETURN_IF_ERROR(ctx_codegen.GetCodeGenHandle()
                          ->op_ir_builder
                          ->Evaluate(inputs, *node, ctx_codegen, outputs));

  // Collect constructed tvm::Node to onnxruntime::Node mapping
  // Both states and outputs
  // TODO remove GetTVMTensorCtx and LookupLoopStates
  for (const auto& l_state : ctx_codegen.GetTVMTensorCtx().LookupLoopStates(node)) {
    ctx_codegen.RecordTensorToNode(l_state.second, node);
  }

  for (const auto& t : outputs) {
    ctx_codegen.RecordTensorToNode(t, node);
  }

  return Status::OK();
}

// CreateTVMIR is the entry function for building TVM IR
// It will call TVMIRBuilder (in CreateOutputs) from CodeGenContext
Status CreateTVMIR(
    const GraphViewer& graph,
    NupharCodeGenCtx& ctx_codegen,
    bool use_placeholder_for_input) {
  TVMTensorCtx& ctx_shape_and_tensor = ctx_codegen.GetTVMTensorCtx();

  if (use_placeholder_for_input) {
    // build graph inputs
    const auto& graph_inputs = graph.GetInputs();
    for (size_t i = 0; i < graph_inputs.size(); ++i) {
      tvm::Tensor value;
      if (CreateInput(graph_inputs[i], value,
                      /*initializer_only*/ false, /*is_sliced*/ false,
                      ctx_codegen))
        ctx_shape_and_tensor.inputs.emplace(graph_inputs[i]->Name(), std::move(value));
    }
  }

  for (const auto& node : graph.Nodes()) {
    // initializers
    node.ForEachWithIndex(
        node.InputDefs(),
        [&ctx_codegen, &ctx_shape_and_tensor](const NodeArg& def, size_t) {
          tvm::Tensor value;
          if (CreateInput(&def, value, /*initializer_only*/ true, /*is_sliced*/ false,
                          ctx_codegen))
            ctx_shape_and_tensor.inputs.emplace(def.Name(), std::move(value));
          return Status::OK();
        });
  }

  // iterate though the graph and create op (outputs)
  for (auto node_index : graph.GetNodesInTopologicalOrder()) {
    const auto& node = *graph.GetNode(node_index);
    tvm::Array<tvm::Tensor> inputs;
    for (const NodeArg* def : node.InputDefs()) {
      inputs.push_back(def->Exists() ? ctx_shape_and_tensor.Lookup(def) : tvm::Tensor());
    }

    // TODO: remove this
    // TODO remove CreateRecurrentStates
    std::vector<std::pair<tvm::Tensor, tvm::Tensor>> l_states;
    CreateRecurrentStates(node, l_states);
    ctx_shape_and_tensor.loop_states.emplace(&node, std::move(l_states));

    auto subgraph = GetSubgraph(node);
    if (subgraph) {
      GraphViewer subgraph_viewer(*subgraph);
      ORT_RETURN_IF_ERROR(CreateTVMIR(subgraph_viewer, ctx_codegen, /*use_placeholder_for_input*/ false));
    } else {
      tvm::Array<tvm::Tensor> op_outputs;
      ORT_RETURN_IF_ERROR(CreateOutputs(&node, inputs, op_outputs, ctx_codegen));
      ctx_shape_and_tensor.ops.emplace(&node, std::move(op_outputs));

      // input_from_
      node.ForEachWithIndex(
          node.OutputDefs(),
          [&node, &ctx_shape_and_tensor](const NodeArg& def, size_t index) {
            ORT_ENFORCE(ctx_shape_and_tensor.input_from.count(def.Name()) == 0);
            ctx_shape_and_tensor.input_from.emplace(def.Name(), std::make_pair(&node, index));
            return Status::OK();
          });
    }
  }

  return Status::OK();
}

// CreateTVMIR is the entry function for building TVM IR
// It will call TVMIRBuilder (in CreateOutputs) from CodeGenContext
Status CreateTVMIR(
    const Node& node,
    NupharCodeGenCtx& ctx_codegen) {
  // wrapper
  TVMTensorCtx& ctx_shape_and_tensor = ctx_codegen.GetTVMTensorCtx();
  bool has_loop = HasLoop(node);

  // create real Inputs
  node.ForEachWithIndex(
      node.InputDefs(),
      [&has_loop, &ctx_codegen, &ctx_shape_and_tensor](const NodeArg& def, size_t) {
        tvm::Tensor value;
        if (CreateInput(&def, value, /*initializer_only*/ false, /*is_sliced*/ has_loop,
                        ctx_codegen))
          ctx_shape_and_tensor.inputs.emplace(def.Name(), std::move(value));
        return Status::OK();
      });

  // input_from_
  node.ForEachWithIndex(
      node.OutputDefs(),
      [&node, &ctx_shape_and_tensor](const NodeArg& def, size_t index) {
        ctx_shape_and_tensor.input_from.emplace(def.Name(), std::make_pair(&node, index));
        return Status::OK();
      });

  tvm::Array<tvm::Tensor> inputs;
  for (const NodeArg* def : node.InputDefs()) {
    inputs.push_back(def->Exists() ? ctx_shape_and_tensor.Lookup(def) : tvm::Tensor());
  }

  // TODO remove this
  // TODO remove CreateRecurrentStates
  // create loop states.
  std::vector<std::pair<tvm::Tensor, tvm::Tensor>> l_states;
  CreateRecurrentStates(node, l_states);
  ctx_shape_and_tensor.loop_states.emplace(&node, std::move(l_states));

  // create ops (outputs)
  tvm::Array<tvm::Tensor> op_outputs;
  ORT_RETURN_IF_ERROR(CreateOutputs(&node, inputs, op_outputs, ctx_codegen));
  ctx_shape_and_tensor.ops.emplace(&node, std::move(op_outputs));

  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
