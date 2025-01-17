// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime::coreml {

class SliceOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  int GetMinSupportedOpSet(const Node& /* node */) const override {
    // Before Slice-10, some inputs were attributes instead. We don't support that for now.
    return 10;
  }

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& builder_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

namespace {
Status PrepareSliceComputeMetadata(const Node& slice_node,
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

    const auto* tensor_proto = graph_viewer.GetConstantInitializer(input_defs[input_idx]->Name());
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
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();

  std::vector<int64_t> data_shape;
  ORT_RETURN_IF_NOT(GetStaticShape(*node.InputDefs()[0], data_shape, logger), "Failed to get input shape.");
  auto rank = data_shape.size();

  SliceOp::PrepareForComputeMetadata compute_metadata{data_shape};
  ORT_RETURN_IF_ERROR(PrepareSliceComputeMetadata(node, model_builder.GetGraphViewer(), compute_metadata));

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;  // NOLINT
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation.slice_by_index

    const InlinedVector<bool> begin_mask_values(rank, false);
    InlinedVector<bool> end_mask_values(rank, false);

    // Special case - stepping backwards up to and including the first index in the dimension.
    // In ONNX Slice, we use end <= -(rank + 1) to represent this. In CoreML, setting endids like that doesn't work,
    // so use endmasks to specify the rest of the dimension instead.
    for (size_t i = 0; i < rank; ++i) {
      if (compute_metadata.steps_[i] < 0 && compute_metadata.ends_[i] == -1) {
        end_mask_values[i] = true;
      }
    }

    // Int32, float and float16 are supported by CoreML slice_by_index.
    // We convert any int64 model input to int32 when running the CoreML model for the partition.
    // Any other integer data created at runtime is the output from CoreML operations, and should int32 not int64.
    // Based on that, we assume that the actual input when running will be int32, so we override the output data
    // type to reflect this.
    // If we were to leave it as TensorProto_DataType_INT64 the CoreML model would be invalid.
    std::optional<int32_t> output_datatype;

    int32_t input_type;
    ORT_RETURN_IF_NOT(GetType(*node.InputDefs()[0], input_type, logger), "Failed to get input type");

    if (input_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      output_datatype = ONNX_NAMESPACE::TensorProto_DataType_INT32;
    }

    auto op = model_builder.CreateOperation(node, "slice_by_index");

    auto begin = model_builder.AddConstant(op->type(), "begin", AsSpan(compute_metadata.starts_));
    auto end = model_builder.AddConstant(op->type(), "end", AsSpan(compute_metadata.ends_));
    auto stride = model_builder.AddConstant(op->type(), "stride", AsSpan(compute_metadata.steps_));
    auto begin_mask = model_builder.AddConstant(op->type(), "begin_mask", AsSpan(begin_mask_values));
    auto end_mask = model_builder.AddConstant(op->type(), "end_mask", AsSpan(end_mask_values));

    AddOperationInput(*op, "x", input_defs[0]->Name());
    AddOperationInput(*op, "begin", begin);
    AddOperationInput(*op, "end", end);
    AddOperationInput(*op, "stride", stride);
    AddOperationInput(*op, "begin_mask", begin_mask);
    AddOperationInput(*op, "end_mask", end_mask);

    AddOperationOutput(*op, *output_defs[0], output_datatype);

    model_builder.AddOperation(std::move(op));

  } else  // NOLINT
#endif    // defined(COREML_ENABLE_MLPROGRAM)
  {
    auto layer = model_builder.CreateNNLayer(node);
    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = output_defs[0]->Name();
    auto* slice_static = layer->mutable_slicestatic();

    for (size_t i = 0; i < rank; ++i) {
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
  }

  return Status::OK();
}

bool SliceOpBuilder::HasSupportedInputsImpl(const Node& node,
                                            [[maybe_unused]] const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger)) {
    return false;
  }

#ifdef COREML_ENABLE_MLPROGRAM
  // The [Doc](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation.slice_by_index)
  // says ML Program slice_by_index supports fp16 in CoreML 5 (iOS 15).
  // It's incorrect and CoreML 6+ (iOS16, CoreML spec version >= 7) is required otherwise only float is supported.
  // CoreML 5:https://github.com/apple/coremltools/blob/89d058ffdcb0b39a03031782d8a448b6889ac425/coremltools/converters/mil/mil/ops/defs/tensor_transformation.py#L515
  // CoreML 6:https://github.com/apple/coremltools/blob/c3ea4cf56fef1176417246c1b85363417f3e713d/coremltools/converters/mil/mil/ops/defs/iOS15/tensor_transformation.py#L495
  if (input_params.create_mlprogram && input_params.coreml_version >= 6 &&
      input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
  } else
#endif  // nolint
    if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      LOGS(logger, VERBOSE) << "[" << node.OpType() << "] Input type: [" << input_type << "] is not supported";
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
  auto status = PrepareSliceComputeMetadata(node, builder_params.graph_viewer, compute_metadata);
  if (status != Status::OK()) {
    LOGS(logger, VERBOSE) << "PrepareSliceComputeMetadata failed:" << status.ErrorMessage();
    return false;
  }

  if (!ValidateSliceComputeMetadataForCoreML(compute_metadata, logger)) {
    // error logged in ValidateSliceComputeMetadataForCoreML
    return false;
  }

  return true;
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::coreml
