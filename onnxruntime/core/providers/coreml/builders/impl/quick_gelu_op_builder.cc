// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

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

// com.microsoft:QuickGelu is produced by ORT's QuickGeluFusion pass
// (onnxruntime/core/optimizer/quick_gelu_fusion.cc) at optimization level
// ORT_ENABLE_EXTENDED and above. The schema in contrib_defs.cc defines it as
//     Y = X * Sigmoid(alpha * X)    default alpha = 1.702
// CoreML has no native equivalent, so we decompose to three MIL ops — all
// primitives are already CoreML-supported. Same approach the QNN EP uses
// in qnn/builder/opbuilder/quick_gelu_op_builder.cc.
class QuickGeluOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status QuickGeluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  // IsOpSupportedImpl gates this, but fail fast rather than silently produce an
  // invalid model if the path is ever reached without MLProgram.
  ORT_RETURN_IF_NOT(model_builder.CreateMLProgram(),
                    "QuickGelu is only supported by the CoreML EP in MLProgram format");

  NodeAttrHelper helper(node);
  const float alpha = helper.Get("alpha", 1.702f);

  const auto input_dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t elem_type = static_cast<int32_t>(input_dtype);
  const std::string& x_name = node.InputDefs()[0]->Name();

  std::vector<int64_t> x_shape;
  ORT_RETURN_IF_NOT(GetShape(*node.InputDefs()[0], x_shape, logger), "Failed to get QuickGelu input shape");

  {
    using namespace CoreML::Specification::MILSpec;

    // When alpha ≈ 1.0 (e.g. CLIP's approximate GELU, `x * sigmoid(x)`), skip
    // the leading mul and feed x straight into sigmoid. Saves one op and
    // avoids the rounding it would introduce. Mirrors QNN's builder at
    // qnn/builder/opbuilder/quick_gelu_op_builder.cc:42-49.
    constexpr float kAlphaEpsilon = 1e-6f;
    const bool skip_alpha_mul = std::abs(alpha - 1.0f) < kAlphaEpsilon;

    std::string sigmoid_input_name = x_name;
    std::unique_ptr<Operation> mul_alpha;
    if (!skip_alpha_mul) {
      // alpha_x = mul(x, alpha)
      mul_alpha = model_builder.CreateOperation(node, "mul", "alpha");
      AddOperationInput(*mul_alpha, "x", x_name);
      if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        AddOperationInput(*mul_alpha, "y", model_builder.AddScalarConstant(mul_alpha->type(), "alpha", alpha));
      } else {
        AddOperationInput(*mul_alpha, "y",
                          model_builder.AddScalarConstant(mul_alpha->type(), "alpha", MLFloat16(alpha)));
      }
      sigmoid_input_name = model_builder.GetUniqueName(node, "quick_gelu_alpha_x");
      AddIntermediateOperationOutput(*mul_alpha, sigmoid_input_name, elem_type, x_shape);
    }

    // sig = sigmoid(sigmoid_input)
    auto sig = model_builder.CreateOperation(node, "sigmoid");
    AddOperationInput(*sig, "x", sigmoid_input_name);
    const std::string& sig_name = model_builder.GetUniqueName(node, "quick_gelu_sigmoid");
    AddIntermediateOperationOutput(*sig, sig_name, elem_type, x_shape);

    // y = mul(x, sig)
    auto mul_final = model_builder.CreateOperation(node, "mul", "final");
    AddOperationInput(*mul_final, "x", x_name);
    AddOperationInput(*mul_final, "y", sig_name);
    AddOperationOutput(*mul_final, *node.OutputDefs()[0]);

    if (mul_alpha) {
      model_builder.AddOperation(std::move(mul_alpha));
    }
    model_builder.AddOperation(std::move(sig));
    model_builder.AddOperation(std::move(mul_final));
  }

  return Status::OK();
}

bool QuickGeluOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  // Only the MLProgram path is implemented. NeuralNetwork format is deprecated
  // on Apple Silicon and not worth carrying a second implementation for.
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "QuickGelu: only MLProgram format is supported by the CoreML EP";
    return false;
  }

  // AddToModelBuilderImpl requires the input shape to size intermediate MIL
  // outputs, so check here and fall back to CPU if shape inference was
  // incomplete — don't claim the node and then fail at model-build time.
  std::vector<int64_t> x_shape;
  if (!GetShape(*node.InputDefs()[0], x_shape, logger)) {
    LOGS(logger, VERBOSE) << "QuickGelu: failed to get input shape";
    return false;
  }

  return true;
}

void CreateQuickGeluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<QuickGeluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
