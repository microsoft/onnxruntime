// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/nuphar_op_ir_builder.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/passes/op_ir_creator/all_ops.h"
#include "core/codegen/passes/op_ir_creator/tvm_ir_builder.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/common/common.h"
#include "core/providers/nuphar/common/nuphar_tvm_utils.h"
#include "core/providers/nuphar/compiler/initializer_info.h"
#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

namespace onnxruntime {
namespace nuphar {

// Declaration of GetOrCreateInitializer
// GetOrCreateInitializer create tvm::placeholder for a marshalled weight
// with correpsonding data layout transfomration for a weight,
// Note the weight is fed during build
static const tvm::Tensor& GetOrCreateInitializer(const std::string& name,
                                                 const Tensor* tensor,
                                                 bool is_sliced,
                                                 NupharCodeGenCtx& ctx_codegen);

static const tvm::Tensor& GetOrCreateInitializer(const NodeArg* def,
                                                 const Tensor* tensor,
                                                 bool is_sliced,
                                                 NupharCodeGenCtx& ctx_codegen);

static bool CreateScalarTensorFromInitializer(const Tensor* tensor,
                                              const std::string& name,
                                              NupharCodeGenCtx& ctx_codegen);

// CreateInputPlaceholder create tvm input placeholder (tvm::Tensor)
// NOTE: here we assume axis 0 is sequence
// TODO: add support for sequence not axis 0
static tvm::Tensor CreateInputPlaceholder(const tvm::Array<tvm::Expr>& shape,
                                          HalideIR::Type halide_type,
                                          const std::string& name,
                                          bool is_sliced) {
  return tvm::placeholder(is_sliced && shape.size() > 1 ? tvm_codegen::SliceShapeFromDimension(shape, 1) : shape, halide_type, name);
}

// CreateInput creats tvm::Tensor of corresponding ORT input
// Inputs are either initializer or regular input placeholder
static bool CreateInput(
    const NodeArg* def,
    tvm::Tensor& input,
    bool initializer_only,
    bool is_sliced,
    NupharCodeGenCtx& ctx_codegen) {
  const Tensor* initialized_tensor = ctx_codegen.GetOrtInitializerTensor(def->Name());
  if (nullptr == initialized_tensor && initializer_only)
    return false;

  ORT_ENFORCE(def->Shape());

  if (nullptr != initialized_tensor &&
      CreateScalarTensorFromInitializer(initialized_tensor, def->Name(), ctx_codegen)) {
    return false;  // constant scalar tensor do not need to be in input
  }

  if (nullptr != initialized_tensor) {
    input = GetOrCreateInitializer(def, initialized_tensor, is_sliced, ctx_codegen);
  } else {
    // Handle inputs without initializer
    std::string name = NormalizeNodeArgName(def);
    MLDataType ONNXRUNTIME_data_type = DataTypeImpl::TypeFromProto(*def->TypeAsProto());
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(ONNXRUNTIME_data_type);
    HalideIR::Type halide_type((halideir_type_code_t)dtype.code, dtype.bits, dtype.lanes);
    tvm::Array<tvm::Expr> shape = ShapeToTvmArray(def, ctx_codegen);

    // Create InputPlaceholder
    // Slice InputPlaceholder if it is asked for.
    input = CreateInputPlaceholder(shape, halide_type, name, is_sliced);
  }
  return true;
}

bool CreateScalarTensorFromInitializer(const Tensor* tensor,
                                       const std::string& name,
                                       NupharCodeGenCtx& ctx_codegen) {
  TVMTensorCtx& ctx_tensor = ctx_codegen.GetTVMTensorCtx();
  ORT_ENFORCE(tensor != nullptr);

  tvm::Expr constant_scalar;
  if (!TryCreateConstantScalar(constant_scalar, tensor))
    return false;

  std::string normalized_name = NormalizeCppName(name);
  auto tvm_tensor = tvm::compute(
      tvm_codegen::ToTvmArray(tensor->Shape().GetDims()),
      [&](const tvm::Array<tvm::Var>&) {
        return constant_scalar;
      },
      normalized_name);

  ctx_codegen.InsertLiteral(normalized_name);
  ctx_tensor.inputs.emplace(name, std::move(tvm_tensor));
  return true;
}

// GetOrCreateInitializer create tvm::placeholder for a marshalled weight
// with correpsonding data layout transfomration for a weight,
// Note the weight is fed during build
const tvm::Tensor& GetOrCreateInitializer(const std::string& name,
                                          const Tensor* tensor,
                                          bool is_sliced,
                                          NupharCodeGenCtx& ctx_codegen) {
  ORT_ENFORCE(ctx_codegen.IsInitializer(name));

  auto layout_info = ctx_codegen.GetWeightLayoutInfo(name);
  if (nullptr != layout_info) {
    return layout_info->marshalled_tensor;
  }

  auto ONNXRUNTIME_data_type = tensor->DataType();
  DLDataType dtype = tvm_codegen::ToTvmDLDataType(ONNXRUNTIME_data_type);
  HalideIR::Type halide_type((halideir_type_code_t)dtype.code, dtype.bits, dtype.lanes);
  std::string normalized_name = NormalizeCppName(name);
  auto tvm_shape = tvm_codegen::ToTvmArray(tensor->Shape().GetDims());
  auto tvm_tensor = CreateInputPlaceholder(tvm_shape, halide_type, normalized_name, is_sliced);
  // create the layout info
  ctx_codegen.CreateWeightLayoutInfo(name, tvm_tensor);
  return ctx_codegen.GetWeightLayoutInfo(name)->marshalled_tensor;
}

const tvm::Tensor& GetOrCreateInitializer(const NodeArg* def,
                                          const Tensor* tensor,
                                          bool is_sliced,
                                          NupharCodeGenCtx& ctx_codegen) {
  return GetOrCreateInitializer(def->Name(), tensor, is_sliced, ctx_codegen);
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
  TVMTensorCtx& ctx_tensor = ctx_codegen.GetTVMTensorCtx();

  if (use_placeholder_for_input) {
    // build graph inputs
    const auto& graph_inputs = graph.GetInputs();
    for (size_t i = 0; i < graph_inputs.size(); ++i) {
      tvm::Tensor value;
      if (CreateInput(graph_inputs[i], value,
                      /*initializer_only*/ false, /*is_sliced*/ false,
                      ctx_codegen)) {
        ctx_tensor.inputs.emplace(graph_inputs[i]->Name(), std::move(value));
      }
    }
  }

  for (const auto& node : graph.Nodes()) {
    // initializers
    ORT_RETURN_IF_ERROR(node.ForEachWithIndex(
        node.InputDefs(),
        [&ctx_codegen, &ctx_tensor](const NodeArg& def, size_t) {
          tvm::Tensor value;
          if (CreateInput(&def, value, /*initializer_only*/ true, /*is_sliced*/ false,
                          ctx_codegen)) {
            ctx_tensor.inputs.emplace(def.Name(), std::move(value));
          }
          return Status::OK();
        }));
  }

  // iterate through the graph and create op (outputs)
  for (auto node_index : graph.GetNodesInTopologicalOrder()) {
    const auto& node = *graph.GetNode(node_index);
    tvm::Array<tvm::Tensor> inputs;
    for (const NodeArg* def : node.InputDefs()) {
      tvm::Tensor input;
      if (def->Exists()) {
        bool exist = ctx_tensor.Lookup(def, input);
        if (!exist) {
          tvm::Tensor value;
          if (CreateInput(def, value,
                          /*initializer_only*/ false, /*is_sliced*/ false,
                          ctx_codegen)) {
            ctx_tensor.inputs.emplace(def->Name(), std::move(value));
          }
          input = ctx_tensor.Lookup(def);
        }
      }
      inputs.push_back(input);
    }

    auto subgraph = GetSubgraph(node);
    if (nullptr != subgraph) {
      // unboxing
      GraphViewer subgraph_viewer(*subgraph);
      ORT_RETURN_IF_ERROR(CreateTVMIR(subgraph_viewer, ctx_codegen, /*use_placeholder_for_input*/ false));
    } else {
      tvm::Array<tvm::Tensor> op_outputs;
      ORT_RETURN_IF_ERROR(CreateOutputs(&node, inputs, op_outputs, ctx_codegen));
      ctx_tensor.ops.emplace(&node, std::move(op_outputs));

      // input_from_
      ORT_RETURN_IF_ERROR(node.ForEachWithIndex(
          node.OutputDefs(),
          [&node, &ctx_tensor](const NodeArg& def, size_t index) {
            ORT_ENFORCE(ctx_tensor.input_from.count(def.Name()) == 0);
            ctx_tensor.input_from.emplace(def.Name(), std::make_pair(&node, index));
            return Status::OK();
          }));
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
  TVMTensorCtx& ctx_tensor = ctx_codegen.GetTVMTensorCtx();
  bool has_loop = HasLoop(node);

  // create real Inputs
  ORT_RETURN_IF_ERROR(node.ForEachWithIndex(
      node.InputDefs(),
      [&has_loop, &ctx_codegen, &ctx_tensor](const NodeArg& def, size_t) {
        tvm::Tensor value;
        if (CreateInput(&def, value, /*initializer_only*/ false, /*is_sliced*/ has_loop,
                        ctx_codegen)) {
          ctx_tensor.inputs.emplace(def.Name(), std::move(value));
        }
        return Status::OK();
      }));

  // input_from_
  ORT_RETURN_IF_ERROR(node.ForEachWithIndex(
      node.OutputDefs(),
      [&node, &ctx_tensor](const NodeArg& def, size_t index) {
        ctx_tensor.input_from.emplace(def.Name(), std::make_pair(&node, index));
        return Status::OK();
      }));

  tvm::Array<tvm::Tensor> inputs;
  for (const NodeArg* def : node.InputDefs()) {
    inputs.push_back(def->Exists() ? ctx_tensor.Lookup(def) : tvm::Tensor());
  }

  // create ops (outputs)
  tvm::Array<tvm::Tensor> op_outputs;
  ORT_RETURN_IF_ERROR(CreateOutputs(&node, inputs, op_outputs, ctx_codegen));
  ctx_tensor.ops.emplace(&node, std::move(op_outputs));

  return Status::OK();
}

// CreateTVMIR is the entry function for building TVM IR
// It will call TVMIRBuilder (in CreateOutputs) from CodeGenContext
Status CreateTVMIR(
    const nuphar::NupharSubgraphUnit& subgraph,
    NupharCodeGenCtx& ctx_codegen) {
  ////////////////////////////////////////
  // handle a special case for a single node
  ////////////////////////////////////////
  if (subgraph.IsSingleNode()) {
    const Node* node = subgraph.nodes.front();

    const Graph* onnx_graph = GetSubgraph(*node);

    if (nullptr != onnx_graph) {
      return CreateTVMIR(GraphViewer(*onnx_graph), ctx_codegen, true);
    }
    return CreateTVMIR(*node, ctx_codegen);
  }

  //////////////////////////////
  // handle a generic subgraph below
  //////////////////////////////
  TVMTensorCtx& ctx_tensor = ctx_codegen.GetTVMTensorCtx();

  // build subgraph inputs
  for (const NodeArg* def : subgraph.inputs) {
    tvm::Tensor value;

    if (CreateInput(def, value, /*initializer_only*/ false, /*is_sliced*/ false,
                    ctx_codegen)) {
      ctx_tensor.inputs.emplace(def->Name(), std::move(value));
    }
  }

  // build subgraph initializers
  for (auto& p : subgraph.initializers) {
    tvm::Tensor value = GetOrCreateInitializer(p.first, p.second, false, ctx_codegen);
    ctx_tensor.inputs.emplace(p.first, std::move(value));
  }

  // iterate through the subgraph nodes and create op (outputs)
  for (auto& node : subgraph.nodes) {
    tvm::Array<tvm::Tensor> inputs;

    // collects local inputs
    for (const NodeArg* def : node->InputDefs()) {
      inputs.push_back(def->Exists() ? ctx_tensor.Lookup(def) : tvm::Tensor());
    }

    tvm::Array<tvm::Tensor> op_outputs;
    ORT_RETURN_IF_ERROR(CreateOutputs(node, inputs, op_outputs, ctx_codegen));
    ctx_tensor.ops.emplace(node, std::move(op_outputs));

    // input_from_
    ORT_RETURN_IF_ERROR(node->ForEachWithIndex(
        node->OutputDefs(),
        [&node, &ctx_tensor](const NodeArg& def, size_t index) {
          ORT_ENFORCE(ctx_tensor.input_from.count(def.Name()) == 0);
          ctx_tensor.input_from.emplace(def.Name(), std::make_pair(node, index));
          return Status::OK();
        }));
  }

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
