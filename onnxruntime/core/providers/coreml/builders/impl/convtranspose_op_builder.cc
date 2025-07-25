// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

using namespace CoreML::Specification;

namespace onnxruntime {
namespace coreml {

class ConvTransposeOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& /* node */, const OpBuilderInputParams& /* input_params */,
                         const logging::Logger& /* logger */) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status ConvTransposeOpBuilder::AddToModelBuilderImpl([[maybe_unused]] ModelBuilder& model_builder,
                                                     [[maybe_unused]] const Node& node,
                                                     const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;  // NOLINT
  const auto input_defs = node.InputDefs();
  const auto output_defs = node.OutputDefs();
  const auto& input_name = input_defs[0]->Name();

  NodeAttrHelper helper(node);

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.conv.conv_transpose
  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "conv_transpose");
  const auto& op_type = op->type();

  AddOperationInput(*op, "x", input_name);
  AddOperationInput(*op, "weight", input_defs[1]->Name());

  if (input_defs.size() > 2) {
    AddOperationInput(*op, "bias", input_defs[2]->Name());
  }

  // we know this input has a valid shape due to the check in IsOpSupportedImpl. ignore N and C dims.
  const auto num_spatial_dims = input_defs[1]->Shape()->dim_size() - 2;

  // Spec says strides/dilations/pads are optional but reality is they're required for at least the iOS15 target
  // which is CoreML5. Due to that we just add everything for simplicity.
  const auto strides = helper.Get("strides", std::vector<int64_t>(num_spatial_dims, 1));
  const auto dilations = helper.Get("dilations", std::vector<int64_t>(num_spatial_dims, 1));

  AddOperationInput(*op, "strides", model_builder.AddConstant(op_type, "strides", strides));
  AddOperationInput(*op, "dilations", model_builder.AddConstant(op_type, "dilations", dilations));

  const std::optional<int64_t> groups = helper.GetInt64("group");
  if (groups) {
    AddOperationInput(*op, "groups", model_builder.AddScalarConstant(op_type, "groups", *groups));
  }

  // if we can enable output_shape, this code works. see IsOpSupportedImpl for the reason it's disabled.
  // const auto output_shape = helper.GetInt64s("output_shape");
  // if (output_shape) {
  //  AddOperationInput(*op, "output_shape", model_builder.AddConstant(op_type, "output_shape", *output_shape));
  //  // these are required despite the spec saying otherwise
  //  AddOperationInput(*op, "pad_type", model_builder.AddScalarConstant(op_type, "pad_type", std::string("valid")));
  //  std::vector<int64_t> pads(num_spatial_dims * 2, 0);
  //  AddOperationInput(*op, "pad", model_builder.AddConstant(op_type, "pad", pads));
  //} else {
  //  AddPadTypeAndPads(*op, model_builder, op_type, helper, num_spatial_dims);
  //}

  AddPadTypeAndPads(*op, model_builder, op_type, helper, num_spatial_dims);

  AddOperationOutput(*op, *output_defs[0]);

  model_builder.AddOperation(std::move(op));

  return Status::OK();
}

bool ConvTransposeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                               const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "ConvTranspose: ML Program required";
    return false;
  }

  // ML Program
  // - const weight until CoreML7 (iOS17)
  //   - require constant for now as non-const would be unusual and we rely on the shape of W to be known to validate
  //     the kernel_shape can be used
  // - const bias
  // - const pad
  //   - if auto_pad is same_upper or same_lower the output[i] - (input[i] * strides[i]) must be divisible by 2
  //     as the pads must be equally split as there's no upper/lower option in CoreML
  //     - punting on supporting this for now
  //   - must be symmetric for CoreML to do the right thing
  // - const strides/dilations/groups
  // - output_shape CoreML output is inconsistent so disabled for now
  //
  // NOTE: need to test with/without the COREML_FLAG_USE_CPU_ONLY flag being set to get an idea of how flaky the CoreML
  // behavior is.
  // Update /onnxruntime/test/util/default_providers.cc:DefaultCoreMLExecutionProvider to do so

  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    // requires the rank at least to be known
    LOGS(logger, VERBOSE) << "ConvTranspose: failed to get input shape";
    return false;
  }

  // for simplicity require weight to be constant
  const auto& weight_arg = *input_defs[1];
  const auto& weight_name = input_defs[1]->Name();
  const auto* weight = input_params.graph_viewer.GetConstantInitializer(weight_name);
  if (!weight) {
    LOGS(logger, VERBOSE) << "ConvTranspose: weight must be constant";
    return false;
  }

  if (input_defs.size() > 2 && !input_params.graph_viewer.GetConstantInitializer(input_defs[2]->Name())) {
    LOGS(logger, VERBOSE) << "ConvTranspose: bias must be constant";
    return false;
  }

  std::vector<int64_t> weight_shape;
  if (!GetShape(weight_arg, weight_shape, logger)) {
    // impossible as it's a constant initializer
    LOGS(logger, VERBOSE) << "ConvTranspose: failed to get weight shape";
    return false;
  }

  if (!CheckShapeForConvMemoryLimit(weight_shape, logger) || !CheckShapeForConvMemoryLimit(input_shape, logger)) {
    return false;
  }

  int64_t num_spatial_dims = narrow<int64_t>(weight_shape.size()) - 2;

  NodeAttrHelper helper(node);

  // Punt on SAME_UPPER/SAME_LOWER for now.
  // We could infer that 'same' -> 'same_upper' based on the CoreML conv spec having 'same' and 'same_lower' but
  // need to validate that assertion.
  // Additionally, if the pads size is equal, there's no difference between same_upper and same_lower.
  // To do that we'd need the 'output_shape' attribute to check against.
  // Can add this handling if/when needed.
  auto autopad = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
  if (autopad == AutoPadType::SAME_LOWER || autopad == AutoPadType::SAME_UPPER) {
    LOGS(logger, VERBOSE) << "ConvTranspose: support for SAME_LOWER/SAME_UPPER is not implemented yet";
    return false;
  } else if (autopad == AutoPadType::NOTSET) {
    // CoreML output is inconsistent between CPU_ONLY and ALL if the pads aren't all the same value.
    // CPU matches the expected output, but other devices don't seem to (at least on macOS).
    auto onnx_pads = *helper.GetInt64s("pads");  // 'pads' are required if auto_pad is NOTSET
    const auto pad_value = onnx_pads[0];
    if (!std::all_of(onnx_pads.begin() + 1, onnx_pads.end(),
                     [pad_value](auto value) { return value == pad_value; })) {
      LOGS(logger, VERBOSE) << "ConvTranspose: all pad values must be the same for CoreML to return "
                               "consistent results";
      return false;
    }
  }

  // there's no input to specify a kernel shape in CoreML.
  // it's OK if a specified kernel_shape matches kH and kW dims of the weight input.
  auto kernel_shape = helper.GetInt64s("kernel_shape");
  if (kernel_shape) {
    bool valid = true;

    if (static_cast<int64_t>(kernel_shape->size()) == num_spatial_dims) {
      for (int i = 0; i < num_spatial_dims; ++i) {
        // check the specified kernel shape matches the weight shape. skip the initial N and C dims in the latter.
        if ((*kernel_shape)[i] != weight_shape[i + 2]) {
          valid = false;
          break;
        }
      }
    } else {
      valid = false;
    }

    if (!valid) {
      LOGS(logger, VERBOSE) << "ConvTranspose: kernel_shape attribute does not match the weight shape";
      return false;
    }
  }

  // In theory this can be supported, but running with COREML_FLAG_USE_CPU_ONLY produces output that doesn't match
  // ONNX. Running without that flag produces the expected output. Madness...
  auto output_shape = helper.GetInt64s("output_shape");
  if (output_shape) {
    LOGS(logger, VERBOSE) << "ConvTranspose: output_shape is not supported as the CoreML output is inconsistent";
    return false;
  }

  // output_padding, if specified, must be the default value of all zeros as there's no equivalent in CoreML.
  auto output_padding = helper.GetInt64s("output_padding");
  if (output_padding &&
      std::any_of(output_padding->begin(), output_padding->end(), [](auto value) { return value != 0; })) {
    LOGS(logger, VERBOSE) << "ConvTranspose: output_padding is not supported";
    return false;
  }

  return true;
}

void CreateConvTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConvTransposeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
