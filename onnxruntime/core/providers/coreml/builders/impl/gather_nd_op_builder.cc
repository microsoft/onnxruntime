// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <optional>
#include <vector>

#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// ONNX GatherND(data, indices) maps to the CoreML ML Program 'gather_nd' op.
class GatherNDOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status GatherNDOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                const logging::Logger& logger) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();

  // CoreML's gather_nd does not accept a bool 'x'. Transformer attention-mask
  // graphs gather from bool tensors, so for that case the op is composed as
  // cast(bool -> int32) -> gather_nd -> cast(int32 -> bool). int32 represents
  // 0/1 exactly, so the round-trip is lossless.
  int32_t data_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  GetType(*input_defs[0], data_type, logger);
  const bool data_is_bool = data_type == ONNX_NAMESPACE::TensorProto_DataType_BOOL;

  std::string_view gather_x_name = input_defs[0]->Name();
  if (data_is_bool) {
    std::vector<int64_t> x_shape;
    const bool has_x_shape = GetShape(*input_defs[0], x_shape, logger);
    const std::string& cast_x_name = model_builder.GetUniqueName(node, "gather_nd_x_int32");
    std::unique_ptr<Operation> cast_in = model_builder.CreateOperation(node, "cast");
    AddOperationInput(*cast_in, "x", input_defs[0]->Name());
    AddOperationInput(*cast_in, "dtype",
                      model_builder.AddScalarConstant(cast_in->type(), "dtype", std::string("int32")));
    AddIntermediateOperationOutput(*cast_in, cast_x_name, ONNX_NAMESPACE::TensorProto_DataType_INT32,
                                   has_x_shape ? std::optional<gsl::span<const int64_t>>(x_shape)
                                               : std::nullopt);
    model_builder.AddOperation(std::move(cast_in));
    gather_x_name = cast_x_name;
  }

  // ONNX GatherND permits negative indices (wrapped by the corresponding data dim); CoreML's gather_nd
  // does not. The indices are a constant and the indexed data dims are static (both gated in
  // IsOpSupportedImpl), so wrap any negatives now and re-emit them as an int32 'indices' constant. The
  // original initializer is skipped (see AddInitializersToSkip).
  std::string indices_name;
  {
    std::vector<int64_t> data_shape, indices_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], data_shape, logger) &&
                          GetShape(*input_defs[1], indices_shape, logger) && !indices_shape.empty(),
                      "GatherND: failed to get data/indices shape");
    const size_t depth = static_cast<size_t>(indices_shape.back());
    const Initializer unpacked(model_builder.GetGraphViewer().GetGraph(),
                               *model_builder.GetConstantInitializer(input_defs[1]->Name()),
                               model_builder.GetGraphViewer().ModelPath());
    int32_t indices_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    GetType(*input_defs[1], indices_type, logger);

    std::vector<int64_t> normalized;
    const auto wrap = [&](auto src) {
      normalized.reserve(src.size());
      for (size_t i = 0; i < src.size(); ++i) {
        int64_t v = static_cast<int64_t>(src[i]);
        if (v < 0) v += data_shape[i % depth];
        normalized.push_back(v);
      }
    };
    if (indices_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      wrap(unpacked.DataAsSpan<int32_t>());
    } else {
      wrap(unpacked.DataAsSpan<int64_t>());
    }
    // AddConstant with int64 values emits an int32 'const' (CoreML uses int32 indices).
    indices_name = model_builder.AddConstant(node.OpType(), "indices", normalized,
                                             gsl::span<const int64_t>(indices_shape));
  }

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.scatter_gather.gather_nd
  // The iOS15 gather_nd has no batch_dims parameter and is equivalent to ONNX
  // GatherND with batch_dims == 0 (other values are gated in IsOpSupportedImpl).
  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "gather_nd");
  AddOperationInput(*op, "x", gather_x_name);
  AddOperationInput(*op, "indices", indices_name);
  // CoreML docs mark validate_indices as optional, but the ML Program parser
  // rejects gather_nd without it (same as the 'gather' op builder).
  AddOperationInput(*op, "validate_indices",
                    model_builder.AddScalarConstant(op->type(), "validate_indices", false));

  if (!data_is_bool) {
    AddOperationOutput(*op, *output_defs[0]);
    model_builder.AddOperation(std::move(op));
    return Status::OK();
  }

  // Cast the int32 gather_nd result back to bool to match the ONNX output type.
  std::vector<int64_t> out_shape;
  const bool has_out_shape = GetShape(*output_defs[0], out_shape, logger);
  const std::string& gather_out_name = model_builder.GetUniqueName(node, "gather_nd_out_int32");
  AddIntermediateOperationOutput(*op, gather_out_name, ONNX_NAMESPACE::TensorProto_DataType_INT32,
                                 has_out_shape ? std::optional<gsl::span<const int64_t>>(out_shape)
                                               : std::nullopt);
  model_builder.AddOperation(std::move(op));

  std::unique_ptr<Operation> cast_out = model_builder.CreateOperation(node, "cast");
  AddOperationInput(*cast_out, "x", gather_out_name);
  AddOperationInput(*cast_out, "dtype",
                    model_builder.AddScalarConstant(cast_out->type(), "dtype", std::string("bool")));
  AddOperationOutput(*cast_out, *output_defs[0]);
  model_builder.AddOperation(std::move(cast_out));
  return Status::OK();
}

bool GatherNDOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                          const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "GatherND is only supported for the ML Program format.";
    return false;
  }

  // The iOS15 gather_nd op has no batch_dims parameter, so only batch_dims == 0
  // (the ONNX default) maps directly.
  NodeAttrHelper helper(node);
  const auto batch_dims = helper.Get("batch_dims", int64_t{0});
  if (batch_dims != 0) {
    LOGS(logger, VERBOSE) << "GatherND only supports batch_dims == 0. Got: " << batch_dims;
    return false;
  }

  // CoreML's gather_nd miscomputes the result for some data/indices shape combinations when 'indices'
  // is a non-constant (runtime) input -- it returns slice 0 regardless of the actual index value. With
  // a constant 'indices' the op is correct (verified on-device), and constant indices is the common case
  // (e.g. transformer attention-mask gathers). Require a constant 'indices' so we never silently emit
  // wrong results; non-constant cases fall back to CPU.
  if (!input_params.graph_viewer.IsConstantInitializer(node.InputDefs()[1]->Name(), /*check_outer_scope*/ true)) {
    LOGS(logger, VERBOSE) << "GatherND: 'indices' must be a constant initializer for the CoreML EP.";
    return false;
  }

  // Negative indices are normalized to positive at build time (AddToModelBuilderImpl), which needs the
  // indexed data dims -- the first indices.shape[-1] dims -- to be statically known.
  std::vector<int64_t> data_shape, indices_shape;
  if (!GetShape(*node.InputDefs()[0], data_shape, logger) ||
      !GetShape(*node.InputDefs()[1], indices_shape, logger) || indices_shape.empty()) {
    LOGS(logger, VERBOSE) << "GatherND: data or indices shape is unknown.";
    return false;
  }
  const size_t depth = static_cast<size_t>(indices_shape.back());
  if (depth > data_shape.size()) {
    LOGS(logger, VERBOSE) << "GatherND: index tuple depth " << depth << " exceeds data rank " << data_shape.size();
    return false;
  }
  for (size_t k = 0; k < depth; ++k) {
    if (data_shape[k] < 0) {
      LOGS(logger, VERBOSE) << "GatherND: indexed data dims must be static.";
      return false;
    }
  }

  return true;
}

void GatherNDOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // 'indices' is re-emitted as a normalized int32 constant in AddToModelBuilderImpl, so skip the original.
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

bool GatherNDOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t data_type = 0, indices_type = 0;
  if (!GetType(*input_defs[0], data_type, logger) || !GetType(*input_defs[1], indices_type, logger)) {
    return false;
  }

  // gather_nd itself is type-agnostic over 'x' but rejects bool; bool 'data'
  // (transformer mask graphs) is supported via a cast round-trip in
  // AddToModelBuilderImpl. INT64 'data' is accepted because the CoreML EP
  // implicitly narrows int64 to int32 at the model boundary (the int64->int32
  // input conversion in model.mm and the matching INT32 feature/output handling
  // in ModelBuilder::RegisterModelInputOutput), so CoreML never sees int64.
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
    LOGS(logger, VERBOSE) << "GatherND: 'data' input type not supported. Got type: " << data_type;
    return false;
  }

  // ONNX GatherND indices are int64; the CoreML EP converts int64 <-> int32.
  if (indices_type != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
      indices_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    LOGS(logger, VERBOSE) << "GatherND: 'indices' input must be int32 or int64. Got type: " << indices_type;
    return false;
  }
  return true;
}

void CreateGatherNDOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherNDOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
