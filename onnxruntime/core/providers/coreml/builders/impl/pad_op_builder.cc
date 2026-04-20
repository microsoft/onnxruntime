// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class PadOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }

  int GetMinSupportedOpSet(const Node& /* node */) const override {
    // Note: before Pad-11, inputs `pads` and `constant_value` were attributes
    return 11;
  }
};

// Helper function
// Use axes initializer data if `axes` input provided or create default axes vector.
static InlinedVector<int64_t> GetPaddingAxesData(const InitializedTensorSet& initializers,
                                                 const Node& node, int64_t input_rank) {
  InlinedVector<int64_t> axes_tensor_data;
  const auto& input_defs = node.InputDefs();

  if (input_defs.size() > 3) {
    // optional input axes is provided, use axes initializer data
    const ONNX_NAMESPACE::TensorProto& axes_tensor = *initializers.at(input_defs[3]->Name());
    Initializer axes_initializer(axes_tensor);
    const auto axes_data_span = axes_initializer.DataAsSpan<int64_t>();
    std::transform(
        axes_data_span.begin(), axes_data_span.end(), std::back_inserter(axes_tensor_data),
        [input_rank](int64_t axis) { return HandleNegativeAxis(axis, input_rank); });
  } else {
    // if not provided, make a default axes as [0, 1, ..., input_rank - 1]
    InlinedVector<int64_t> default_axes(input_rank);
    std::iota(std::begin(default_axes), std::end(default_axes), 0);
    axes_tensor_data = std::move(default_axes);
  }
  return axes_tensor_data;
}

void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();
  model_builder.AddInitializerToSkip(input_defs[1]->Name());  // pads
  if (input_defs.size() > 2 && input_defs[2]->Exists()) {
    model_builder.AddInitializerToSkip(input_defs[2]->Name());  // constant_value
  }
  if (input_defs.size() > 3 && input_defs[3]->Exists()) {
    model_builder.AddInitializerToSkip(input_defs[3]->Name());  // axes
  }
}

Status PadOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const Node& node,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  GetShape(*input_defs[0], input_shape, logger);
  const auto input_rank = onnxruntime::narrow<int64_t>(input_shape.size());

  const auto& pads_tensor = *model_builder.GetInitializerTensors().at(input_defs[1]->Name());
  Initializer pads_initializer(pads_tensor);
  auto pads_span = pads_initializer.DataAsSpan<int64_t>();

  InlinedVector<int64_t> axes_tensor_data = GetPaddingAxesData(model_builder.GetInitializerTensors(), node, input_rank);
  int64_t num_axes = axes_tensor_data.size();

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;  // NOLINT
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation.pad

    NodeAttrHelper helper(node);
    const auto mode = helper.Get("mode", "constant");

    auto op = model_builder.CreateOperation(node, "pad");
    AddOperationInput(*op, "x", input_defs[0]->Name());

    // Convert ONNX pads format to MIL format.
    // ONNX: [x1_start, x2_start, ..., xN_start, x1_end, x2_end, ..., xN_end] for N axes
    // MIL:  [x1_start, x1_end, x2_start, x2_end, ...] interleaved, for last N padded dims
    // MIL pads the last N dimensions where N = len(pad) / 2.
    // Find the first dimension that has non-zero padding to minimize the pad vector size.
    int64_t first_padded_dim = input_rank;
    for (int64_t dim = 0; dim < input_rank; ++dim) {
      for (int64_t i = 0; i < num_axes; ++i) {
        if (axes_tensor_data[i] == dim && (pads_span[i] != 0 || pads_span[i + num_axes] != 0)) {
          first_padded_dim = dim;
          break;
        }
      }
      if (first_padded_dim < input_rank) break;
    }

    // MIL requires at least 1 pair. If no padding, default to last dim. Pad op is meaningless in this case though.
    if (first_padded_dim == input_rank) {
      first_padded_dim = input_rank - 1;
    }

    std::vector<int64_t> mil_pads;
    for (int64_t dim = first_padded_dim; dim < input_rank; ++dim) {
      int64_t pad_start = 0;
      int64_t pad_end = 0;
      for (int64_t i = 0; i < num_axes; ++i) {
        if (axes_tensor_data[i] == dim) {
          pad_start = pads_span[i];
          pad_end = pads_span[i + num_axes];
          break;
        }
      }
      mil_pads.push_back(pad_start);
      mil_pads.push_back(pad_end);
    }

    AddOperationInput(*op, "pad", model_builder.AddConstant(op->type(), "pad", mil_pads));
    AddOperationInput(*op, "mode", model_builder.AddScalarConstant(op->type(), "mode", std::string(mode)));

    // CoreML runtime requires constant_val even for non-constant modes (despite docs saying optional).
    auto input_dtype = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
    if (mode == "constant" && input_defs.size() > 2 &&
        Contains(model_builder.GetInitializerTensors(), input_defs[2]->Name())) {
      const auto& constant_value_tensor = *model_builder.GetInitializerTensors().at(input_defs[2]->Name());
      Initializer constant_value_initializer(constant_value_tensor);
      if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        AddOperationInput(*op, "constant_val",
                          model_builder.AddScalarConstant(op->type(), "constant_val",
                                                          constant_value_initializer.DataAsSpan<MLFloat16>()[0]));
      } else {
        AddOperationInput(*op, "constant_val",
                          model_builder.AddScalarConstant(op->type(), "constant_val",
                                                          constant_value_initializer.DataAsSpan<float>()[0]));
      }
    } else {
      // Provide default 0.0 for constant mode without explicit value, and for non-constant modes.
      if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        AddOperationInput(*op, "constant_val",
                          model_builder.AddScalarConstant(op->type(), "constant_val", MLFloat16(0.0f)));
      } else {
        AddOperationInput(*op, "constant_val",
                          model_builder.AddScalarConstant(op->type(), "constant_val", 0.0f));
      }
    }

    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  } else {
    // NeuralNetwork path — constant mode only
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);
    auto* coreml_pad = layer->mutable_padding();
    auto* constant_padding_type = coreml_pad->mutable_constant();

    const auto& constant_value_tensor = *model_builder.GetInitializerTensors().at(input_defs[2]->Name());
    Initializer constant_value_initializer(constant_value_tensor);
    float constant_value = constant_value_initializer.DataAsSpan<float>()[0];
    constant_padding_type->set_value(constant_value);

    auto* height_border = coreml_pad->mutable_paddingamounts()->add_borderamounts();
    auto* width_border = coreml_pad->mutable_paddingamounts()->add_borderamounts();
    for (int64_t i = 0; i < num_axes; i++) {
      if (axes_tensor_data[i] == input_rank - 2) {
        height_border->set_startedgesize(pads_span[i]);
        height_border->set_endedgesize(pads_span[i + num_axes]);
      }
      if (axes_tensor_data[i] == input_rank - 1) {
        width_border->set_startedgesize(pads_span[i]);
        width_border->set_endedgesize(pads_span[i + num_axes]);
      }
    }

    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();
    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

bool PadOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                          const logging::Logger& logger) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger))
    return false;

  // NeuralNetwork supports float only. ML Program supports float and float16.
  if (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return true;
  }

  if (input_params.create_mlprogram && input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return true;
  }

  LOGS(logger, VERBOSE) << "[" << node.OpType()
                        << "] Input type: [" << input_type
                        << "] is not supported";
  return false;
}

bool PadOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_shape.empty()) {
    LOGS(logger, VERBOSE) << "Pad requires input to have a shape.";
    return false;
  }

  // NeuralNetwork PaddingLayerParams requires at least 2D input (H,W dimensions).
  // ML Program's MIL pad op supports any rank >= 1.
  if (!input_params.create_mlprogram && input_shape.size() < 2) {
    LOGS(logger, VERBOSE) << "NeuralNetwork Pad requires input shape to be at least 2d, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  const TensorShape shape(input_shape);
  if (shape.Size() == 0) {
    LOGS(logger, VERBOSE) << "Cases that input data being empty due to a dimension with value of 0 is not supported";
    return false;
  }

  NodeAttrHelper helper(node);
  const auto mode = helper.Get("mode", "constant");

  // ML Program supports constant and reflect modes via the MIL pad op.
  // NeuralNetwork only supports constant mode.
  if (input_params.create_mlprogram) {
    if (mode != "constant" && mode != "reflect") {
      LOGS(logger, VERBOSE) << "For ML Program, only `constant` and `reflect` modes are supported, mode: " << mode;
      return false;
    }
  } else {
    if (mode != "constant") {
      LOGS(logger, VERBOSE) << "Only `constant` mode Pad is supported for NeuralNetwork, mode: " << mode;
      return false;
    }
  }

  // For constant mode, ML Program allows omitted `constant_value` (defaults to 0 per ONNX Pad semantics).
  // NeuralNetwork path requires explicit `constant_value`.
  if (mode == "constant") {
    const bool has_constant_value = input_defs.size() > 2 && input_defs[2]->Exists();

    if (!has_constant_value) {
      if (!input_params.create_mlprogram) {
        LOGS(logger, VERBOSE) << "`constant_value` input is required for constant mode Pad op in NeuralNetwork mode.";
        return false;
      }
    } else {
      if (!Contains(initializers, input_defs[2]->Name())) {
        LOGS(logger, VERBOSE) << "constant_value must be a constant initializer.";
        return false;
      }
    }

    if (!input_params.create_mlprogram) {
      // NeuralNetwork only supports float constant_value
      int32_t constant_value_type;
      GetType(*input_defs[2], constant_value_type, logger);
      if (constant_value_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        LOGS(logger, VERBOSE) << "Only float constant_value is supported for NeuralNetwork, got type: "
                              << constant_value_type;
        return false;
      }
    }
  }

  {
    // only support if `pads` input is known and does not contain negative values and only applies padding values
    // for last two dimensions.
    const auto pads_initializer_it = initializers.find(input_defs[1]->Name());
    if (pads_initializer_it == initializers.end()) {
      LOGS(logger, VERBOSE) << "pads must be a constant initializer.";
      return false;
    }

    const ONNX_NAMESPACE::TensorProto& pads_initializer = *pads_initializer_it->second;
    Initializer unpacked_tensor(pads_initializer);

    auto pads_tensor_data = unpacked_tensor.DataAsSpan<int64_t>();
    for (size_t i = 0; i < unpacked_tensor.size(); i++) {
      if (pads_tensor_data[i] < 0) {
        LOGS(logger, VERBOSE) << "Negative pad value is not supported: pads["
                              << i << "] = " << pads_tensor_data[i];
        return false;
      }
    }

    // Check if provided, `axes` input must be a constant initializer
    if (input_defs.size() > 3 && input_defs[3]->Exists()) {
      const auto axes_initializer_it = initializers.find(input_defs[3]->Name());
      if (axes_initializer_it == initializers.end()) {
        LOGS(logger, VERBOSE) << "if provided, `axes` input is required to a constant initializer";
        return false;
      }
    }

    const auto input_rank = onnxruntime::narrow<int64_t>(input_shape.size());
    InlinedVector<int64_t> axes_tensor_data = GetPaddingAxesData(initializers, node, input_rank);
    int64_t num_axes = axes_tensor_data.size();

    // NeuralNetwork PaddingLayerParams only supports padding on last two dimensions [H,W].
    // ML Program's MIL pad op supports padding on any dimensions for constant mode,
    // but only the last two dimensions for reflect/edge modes.
    // https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html#paddinglayerparams
    if (!input_params.create_mlprogram || mode != "constant") {
      for (int64_t i = 0; i < num_axes; i++) {
        if (axes_tensor_data[i] < input_rank - 2) {
          if (pads_tensor_data[i] != 0 || pads_tensor_data[i + num_axes] != 0) {
            LOGS(logger, VERBOSE) << "Only padding on the last two dimensions is supported for "
                                  << (input_params.create_mlprogram ? "non-constant" : "NeuralNetwork") << " mode.";
            return false;
          }
        }
      }
    }

    // For reflect mode, pad amount must be less than the dimension size on each axis.
    if (mode == "reflect") {
      for (int64_t i = 0; i < num_axes; i++) {
        int64_t dim_size = input_shape[axes_tensor_data[i]];
        if (pads_tensor_data[i] >= dim_size || pads_tensor_data[i + num_axes] >= dim_size) {
          LOGS(logger, VERBOSE) << "Reflect pad amount must be less than dimension size. "
                                << "Axis " << axes_tensor_data[i] << " has size " << dim_size
                                << " but pad amounts are [" << pads_tensor_data[i] << ", "
                                << pads_tensor_data[i + num_axes] << "]";
          return false;
        }
      }
    }
  }

  return true;
}

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
