// Copyright (c) Shukant Pal.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

namespace onnxruntime {
namespace coreml {

class ReductionOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool SupportsMLProgram() const override { return true; }
};

namespace {
template <typename T>
void AddReductionParams(T* params, const std::vector<int64_t>& axes, bool keepdims, bool noop_with_empty_axes) {
  params->set_keepdims(keepdims);

  for (auto& axis : axes)
    params->add_axes(axis);

  if (axes.size() == 0 && !noop_with_empty_axes)
    params->set_reduceall(true);
}
}  // namespace

void ReductionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs(node.InputDefs());

  // We have already embedded the axes into the CoreML layer.
  // No need to copy them later to reduce memory consumption.
  if (input_defs.size() > 1)
    model_builder.AddInitializerToSkip(input_defs[1]->Name());
}

Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());

  std::vector<int64_t> axes;

  NodeAttrHelper helper(node);
  if (input_defs.size() > 1 && input_defs[1]->Exists()) {
    auto& axes_tensor = *model_builder.GetConstantInitializer(input_defs[1]->Name());
    Initializer axes_initializer(axes_tensor);
    int64_t* data = axes_initializer.data<int64_t>();
    int64_t size = axes_initializer.size();

    axes = std::vector<int64_t>(data, data + size);
  } else if (helper.HasAttr("axes")) {
    axes = helper.Get("axes", std::vector<int64_t>{});
  }

  const bool keepdims = helper.Get("keepdims", 1) != 0;
  const bool noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0) != 0;
#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    std::string_view coreml_op_type;
    if (noop_with_empty_axes && axes.size() == 0) {
      coreml_op_type = "identity";
    } else if (op_type == "ReduceSum") {
      coreml_op_type = "reduce_sum";
    } else if (op_type == "ReduceMean") {
      coreml_op_type = "reduce_mean";
    } else if (op_type == "ReduceMax") {
      coreml_op_type = "reduce_max";
    } else if (op_type == "ReduceMin") {
      coreml_op_type = "reduce_min";
    } else if (op_type == "ReduceProd") {
      coreml_op_type = "reduce_prod";
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "ReductionOpBuilder::AddToModelBuilderImpl, unexpected op: ", op_type);
    }
    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, coreml_op_type);
    AddOperationInput(*op, "x", input_defs[0]->Name());
    if (coreml_op_type != "identity") {
      if (axes.size() > 0) {
        AddOperationInput(*op, "axes", model_builder.AddConstant(op->type(), "axes", axes));
      }
      AddOperationInput(*op, "keep_dims", model_builder.AddScalarConstant(op->type(), "keep_dims", keepdims));
    }
    AddOperationOutput(*op, *node.OutputDefs()[0]);

    model_builder.AddOperation(std::move(op));
  } else
#endif  // (COREML_ENABLE_MLPROGRAM)
  {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

    if (op_type == "ReduceSum") {
      AddReductionParams(layer->mutable_reducesum(), axes, keepdims, noop_with_empty_axes);
    } else if (op_type == "ReduceMean") {
      AddReductionParams(layer->mutable_reducemean(), axes, keepdims, noop_with_empty_axes);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "ReductionOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
    }

    *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool ReductionOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  if (!input_params.create_mlprogram &&
      (node.OpType() == "ReduceMax" || node.OpType() == "ReduceMin" || node.OpType() == "ReduceProd")) {
    return false;
  }

#if defined(TARGET_OS_IOS) && defined(TARGET_CPU_X86_64) && TARGET_OS_IOS && TARGET_CPU_X86_64
  // skip ReductionOpTest.ReduceSum_half_bert because reduce_sum will output all zeros
  int32_t input_type;
  GetType(*input_defs[0], input_type, logger);
  if (node.OpType() == "ReduceSum" && input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return false;
  }
#endif

  NodeAttrHelper helper(node);

  // noop_with_empty_axes defaults to false and is only available in newer opsets where 'axes' is an optional input
  // so we don't need to check the 'axes' attribute used in older opsets here.
  const bool noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0) != 0;
  bool empty_axes = true;

  if (input_defs.size() > 1 && input_defs[1]->Exists()) {
    // 'axes' is optional input in new opsets
    const auto& axes_name = input_defs[1]->Name();
    const auto* axes = input_params.graph_viewer.GetConstantInitializer(axes_name);
    if (!axes) {
      LOGS(logger, VERBOSE) << "Axes of reduction must be a constant initializer";
      return false;
    }

    empty_axes = axes->int64_data_size() == 0;
  }
  if (empty_axes && noop_with_empty_axes && !input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "NeuralNetwork doesn't support noop on empty axes for reduction layers";
    return false;
  }

  return true;
}

void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReductionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
