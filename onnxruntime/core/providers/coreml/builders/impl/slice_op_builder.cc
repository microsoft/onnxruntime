// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"

#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/shared/utils/utils.h"

#if defined(__APPLE__)
#include "core/providers/coreml/builders/model_builder.h"
#endif

namespace onnxruntime::coreml {

class SliceOpBuilder : public BaseOpBuilder {
  // Add operator related
#ifdef __APPLE__
 private:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif

  // Operator support related
 private:
  int GetMinSupportedOpSet(const Node& /* node */) const override {
    // Before Slice-10, some inputs were attributes instead. We don't support that for now.
    return 10;
  }

  bool HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const override;
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& builder_params,
                         const logging::Logger& logger) const override;
};

namespace {
Status PrepareSliceComputeMetadataFromConstantInitializers(const Node& slice_node,
                                                           const GraphViewer& graph_viewer,
                                                           SliceOp::PrepareForComputeMetadata& compute_metadata) {
  // TODO largely copied from nnapi::SliceOpBuilder::AddToModelBuilderImpl. put it somewhere where it can be reused?

  const auto input_defs = slice_node.InputDefs();

  // We need to copy the data from the starts/ends/axes/steps initializers to int64 vectors
  // to be used in shared PrepareForCompute function to calculate the output shape
  // and normalize inputs, for example, input can be starts/ends/steps for certain axes,
  // PrepareForCompute can generate standard starts/ends/steps/axes for each axes
  TensorShapeVector input_starts;
  TensorShapeVector input_ends;
  TensorShapeVector input_axes;
  TensorShapeVector input_steps;

  const auto CopyInputData = [&input_defs, &graph_viewer](size_t input_idx, TensorShapeVector& data) {
    // This is an optional input, return empty vector
    if (input_idx >= input_defs.size() || !input_defs[input_idx]->Exists()) {
      data = {};
      return Status::OK();
    }

    const auto* tensor_proto = graph_viewer.GetConstantInitializer(input_defs[input_idx]->Name(), true);
    ORT_RETURN_IF_NOT(tensor_proto, "Failed to get constant initializer.");
    Initializer unpacked_tensor(*tensor_proto, graph_viewer.ModelPath());
    const auto data_type = unpacked_tensor.data_type();
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      auto tensor_data = unpacked_tensor.DataAsSpan<int64_t>();
      data.insert(data.end(), tensor_data.begin(), tensor_data.end());
    } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      auto tensor_data = unpacked_tensor.DataAsSpan<int32_t>();
      data.insert(data.end(), tensor_data.begin(), tensor_data.end());
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Data type for starts and ends inputs' is not supported in this build. Got ",
                             data_type);
    }

    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(CopyInputData(1, input_starts));
  ORT_RETURN_IF_ERROR(CopyInputData(2, input_ends));
  ORT_RETURN_IF_ERROR(CopyInputData(3, input_axes));
  ORT_RETURN_IF_ERROR(CopyInputData(4, input_steps));
  ORT_RETURN_IF_ERROR(
      SliceOp::PrepareForComputeHelper(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  return Status::OK();
}

// check things that CoreML is more particular about to avoid CoreML model compilation errors
bool ValidateSliceComputeMetadataForCoreML(const SliceOp::PrepareForComputeMetadata& compute_metadata,
                                           const logging::Logger& logger) {
  for (size_t i = 0; i < compute_metadata.starts_.size(); ++i) {
    const auto step = compute_metadata.steps_[i],
               start = compute_metadata.starts_[i],
               end = compute_metadata.ends_[i];
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
      LOGS(logger, VERBOSE) << "Empty range is not supported: [" << start << ", " << end << ") with step " << step;
      return false;
    }
  }
  return true;
}
}  // namespace

// Add operator related
#if defined(__APPLE__)

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();

  model_builder.AddInitializerToSkip(input_defs[1]->Name());
  model_builder.AddInitializerToSkip(input_defs[2]->Name());
  if (input_defs.size() > 3 && input_defs[3]->Exists()) {
    model_builder.AddInitializerToSkip(input_defs[3]->Name());
  }
  if (input_defs.size() > 4 && input_defs[4]->Exists()) {
    model_builder.AddInitializerToSkip(input_defs[4]->Name());
  }
}

Status SliceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& logger) const {
  std::vector<int64_t> data_shape;
  ORT_RETURN_IF_NOT(GetStaticShape(*node.InputDefs()[0], data_shape, logger), "Failed to get input shape.");

  SliceOp::PrepareForComputeMetadata compute_metadata{data_shape};
  ORT_RETURN_IF_ERROR(PrepareSliceComputeMetadataFromConstantInitializers(node, model_builder.GetGraphViewer(),
                                                                          compute_metadata));

  auto layer = CreateNNLayer(model_builder, node);
  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();
  auto* slice_static = layer->mutable_slicestatic();

  for (size_t i = 0; i < compute_metadata.starts_.size(); ++i) {
    const auto step = compute_metadata.steps_[i],
               start = compute_metadata.starts_[i],
               end = compute_metadata.ends_[i];

    slice_static->add_beginids(start);
    slice_static->add_beginmasks(false);

    if (step < 0 && end == -1) {
      // Special case - stepping backwards up to and including the first index in the dimension.
      // In ONNX Slice, we use end <= -(rank + 1) to represent this. In CoreML, setting endids like that doesn't work,
      // so use endmasks to specify the rest of the dimension instead.
      slice_static->add_endids(-1);  // ignored
      slice_static->add_endmasks(true);
    } else {
      slice_static->add_endids(end);
      slice_static->add_endmasks(false);
    }

    slice_static->add_strides(step);
  }

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

#endif  // defined(__APPLE__)

// Operator support related
bool SliceOpBuilder::HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool SliceOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& builder_params,
                                       const logging::Logger& logger) const {
  const auto input_defs = node.InputDefs();

  std::vector<int64_t> data_shape;
  if (!GetStaticShape(*input_defs[0], data_shape, logger)) {
    return false;
  }

  if (!CheckIsConstantInitializer(*input_defs[1], builder_params.graph_viewer, logger, "'starts'")) {
    return false;
  }

  if (!CheckIsConstantInitializer(*input_defs[2], builder_params.graph_viewer, logger, "'ends'")) {
    return false;
  }

  if (input_defs.size() > 3 && input_defs[3]->Exists() &&
      !CheckIsConstantInitializer(*input_defs[3], builder_params.graph_viewer, logger, "'axes'")) {
    return false;
  }

  if (input_defs.size() > 4 && input_defs[4]->Exists() &&
      !CheckIsConstantInitializer(*input_defs[4], builder_params.graph_viewer, logger, "'steps'")) {
    return false;
  }

  SliceOp::PrepareForComputeMetadata compute_metadata{data_shape};
  ORT_THROW_IF_ERROR(PrepareSliceComputeMetadataFromConstantInitializers(node, builder_params.graph_viewer,
                                                                         compute_metadata));
  if (!ValidateSliceComputeMetadataForCoreML(compute_metadata, logger)) {
    return false;
  }

  return true;
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::coreml
