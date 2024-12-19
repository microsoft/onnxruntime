// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class DepthToSpaceOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status DepthToSpaceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                    const Node& node,
                                                    [[maybe_unused]] const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  const auto& input_name = input_defs[0]->Name();

  NodeAttrHelper helper(node);
  int64_t blocksize = *helper.GetInt64("blocksize");  // required attribute

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;  // NOLINT

    const auto mode = helper.Get("mode", "DCR");

    if (mode == "DCR") {
      // DCR is directly supported
      // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation.depth_to_space
      // Validated with depth_to_space.py.
      auto op = model_builder.CreateOperation(node, "depth_to_space");
      AddOperationInput(*op, "x", input_name);
      AddOperationInput(*op, "block_size", model_builder.AddScalarConstant(op->type(), "blocksize", blocksize));
      AddOperationOutput(*op, *output_defs[0]);
      model_builder.AddOperation(std::move(op));
    } else {
      // CRD is manual. there may be a perf cost from the Reshape's (typically that happens on CPU) but if the input
      // is a fixed size hopefully CoreML is smart enough to handle that aspect during model compilation instead
      // of execution.

      // https://github.com/onnx/onnx/blob/main/docs/Operators.md#depthtospace
      // b, c, h, w = x.shape
      // tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
      // tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
      // y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
      //
      // CoreML has a 5D limit, so we merge the batch dim into the channel dim as that doesn't change the data
      // movement.
      // First reshape is to [b * c // (blocksize ** 2), blocksize, blocksize, h, w]
      // Transpose is to [0, 3, 1, 4, 2]

      // we checked shape was static in IsOpSupportedImpl so this should never fail
      std::vector<int64_t> input_shape;
      ORT_RETURN_IF_NOT(GetStaticShape(*input_defs[0], input_shape, logger), "Failed to get input shape");
      auto input_dtype = input_defs[0]->TypeAsProto()->tensor_type().elem_type();

      const int32_t elem_type = static_cast<int32_t>(input_dtype);

      // reshape to [b * c // (blocksize ** 2), blocksize, blocksize, h, w]
      auto reshape1 = model_builder.CreateOperation(node, "reshape", "pre");
      std::vector<int64_t> shape1 = {input_shape[0] * input_shape[1] / (blocksize * blocksize),
                                     blocksize, blocksize, input_shape[2], input_shape[3]};
      AddOperationInput(*reshape1, "x", input_name);
      AddOperationInput(*reshape1, "shape", model_builder.AddConstant(reshape1->type(), "shape", shape1));
      const auto& reshape1_output = model_builder.GetUniqueName(node, "reshape1");
      AddIntermediateOperationOutput(*reshape1, reshape1_output, elem_type, shape1);

      // transpose to [0, 3, 1, 4, 2]
      auto transpose = model_builder.CreateOperation(node, "transpose");
      std::vector<int64_t> perm = {0, 3, 1, 4, 2};
      std::vector<int64_t> shape2 = {shape1[0], shape1[3], shape1[1], shape1[4], shape1[2]};
      AddOperationInput(*transpose, "x", reshape1_output);
      AddOperationInput(*transpose, "perm", model_builder.AddConstant(transpose->type(), "perm", perm));
      const auto& transpose_output = model_builder.GetUniqueName(node, "transpose");
      AddIntermediateOperationOutput(*transpose, transpose_output, elem_type, shape2);

      // reshape to [b, c // (blocksize ** 2), h * blocksize, w * blocksize]
      auto reshape2 = model_builder.CreateOperation(node, "reshape", "post");
      std::vector<int64_t> shape3 = {input_shape[0],
                                     input_shape[1] / (blocksize * blocksize),
                                     input_shape[2] * blocksize,
                                     input_shape[3] * blocksize};
      AddOperationInput(*reshape2, "x", transpose_output);
      AddOperationInput(*reshape2, "shape", model_builder.AddConstant(reshape2->type(), "shape", shape3));

      AddOperationOutput(*reshape2, *output_defs[0]);

      model_builder.AddOperation(std::move(reshape1));
      model_builder.AddOperation(std::move(transpose));
      model_builder.AddOperation(std::move(reshape2));
    }
  } else  // NOLINT
#endif    // if defined(COREML_ENABLE_MLPROGRAM)
  {
    const auto& output_name = output_defs[0]->Name();
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

    auto* coreml_depthtospace = layer->mutable_reorganizedata();
    coreml_depthtospace->set_blocksize(static_cast<uint64_t>(blocksize));
    coreml_depthtospace->set_mode(CoreML::Specification::ReorganizeDataLayerParams_ReorganizationType::
                                      ReorganizeDataLayerParams_ReorganizationType_DEPTH_TO_SPACE);

    *layer->mutable_input()->Add() = input_name;
    *layer->mutable_output()->Add() = output_name;

    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

bool DepthToSpaceOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "DepthToSpace: no input shape";
    return false;
  }

  // ONNX and CoreML both require 4D input so no need to check the shape here.

  NodeAttrHelper helper(node);
  const auto mode = helper.Get("mode", "DCR");

  if (input_params.create_mlprogram) {
    if (mode == "CRD" && !IsStaticShape(input_shape)) {
      // we need to manually implement the logic with a Reshape, so we need to know the shape to do that
      LOGS(logger, VERBOSE) << "DepthToSpace: CRD mode requires static shape";
      return false;
    }

    if (mode == "DCR" && input_params.coreml_version < 7) {
      int32_t input_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
      GetType(*input_defs[0], input_type, logger);

      if (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        // In CoreML version 6 (e.g., on an iOS 16 simulator) with DCR mode and float16 input, the output is all zeros
        // in this unit test: TensorOpTest/1.DepthToSpaceTest_4.
        // However, CoreML version 7 is fine.
        // Don't support CoreML version < 7, DCR mode, and float16 input.
        LOGS(logger, VERBOSE) << "DepthToSpace: DCR mode with float16 input requires at least CoreML version 7.";
        return false;
      }
    }
  } else {
    if (mode != "DCR") {
      LOGS(logger, VERBOSE) << "DepthToSpace: " << mode << " mode is not supported";
      return false;
    }
  }

  return true;
}

void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DepthToSpaceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
