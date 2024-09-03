// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class GemmOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

// Add operator related.
Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
  emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());
  emscripten::val output = emscripten::val::object();
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  if (op_type == "MatMul") {
    std::vector<int64_t> a_shape;
    if (!GetShape(*input_defs[a_idx], a_shape, logger)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Can not get shape of A.");
    }
    std::vector<int64_t> b_shape;
    if (!GetShape(*input_defs[b_idx], b_shape, logger)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Can not get shape of B.");
    }
    // If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
    bool extended_a_shape = false;
    if (a_shape.size() == 1) {
      extended_a_shape = true;
      a_shape.insert(a_shape.begin(), 1);
      emscripten::val reshape_a_options = emscripten::val::object();
      reshape_a_options.set("label", node.Name() + "_reshape_a");
      a = model_builder.GetBuilder().call<emscripten::val>("reshape", a,
                                                           emscripten::val::array(GetVecUint32FromVecInt64(a_shape)),
                                                           reshape_a_options);
    }
    // If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions.
    bool extended_b_shape = false;
    if (b_shape.size() == 1) {
      extended_b_shape = true;
      b_shape.push_back(1);
      emscripten::val reshape_b_options = emscripten::val::object();
      reshape_b_options.set("label", node.Name() + "_reshape_b");
      b = model_builder.GetBuilder().call<emscripten::val>("reshape", b,
                                                           emscripten::val::array(GetVecUint32FromVecInt64(b_shape)),
                                                           reshape_b_options);
    }

    output = model_builder.GetBuilder().call<emscripten::val>("matmul", a, b, options);

    emscripten::val reshape_output_options = emscripten::val::object();
    reshape_output_options.set("label", node.Name() + "_reshape_output");
    // If the inputs are both 1D， reduce the output to a scalar.
    if (extended_a_shape && extended_b_shape) {
      output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                output,
                                                                emscripten::val::array(),
                                                                reshape_output_options);
    }
    // After matrix multiplication the prepended 1 is removed.
    else if (extended_a_shape) {
      std::vector<uint32_t> new_shape;
      for (size_t i = 0; i < b_shape.size() - 2; i++) {
        new_shape.push_back(narrow<uint32_t>(b_shape[i]));
      }
      new_shape.push_back(narrow<uint32_t>(b_shape.back()));
      output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                output,
                                                                emscripten::val::array(new_shape),
                                                                reshape_output_options);
    }
    // After matrix multiplication the appended 1 is removed.
    else if (extended_b_shape) {
      std::vector<uint32_t> new_shape;
      for (size_t i = 0; i < a_shape.size() - 1; i++) {
        new_shape.push_back(narrow<uint32_t>(a_shape[i]));
      }
      output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                output,
                                                                emscripten::val::array(new_shape),
                                                                reshape_output_options);
    }
  } else if (op_type == "MatMulInteger") {
    emscripten::val a_zero_point = emscripten::val::null();
    emscripten::val b_zero_point = emscripten::val::null();
    if (input_defs.size() >= 3) {
      a_zero_point = model_builder.GetOperand(node.InputDefs()[2]->Name());
    } else {
      a_zero_point = model_builder.GetZeroConstant("uint8");
    }
    if (input_defs.size() >= 4) {
      b_zero_point = model_builder.GetOperand(node.InputDefs()[3]->Name());
    } else {
      b_zero_point = model_builder.GetZeroConstant("uint8");
    }
    output = model_builder.GetBuilder().call<emscripten::val>("matmulInteger",
                                                              a,
                                                              a_zero_point,
                                                              b,
                                                              b_zero_point,
                                                              options);
  } else {  // Gemm
    NodeAttrHelper helper(node);
    const auto transA = helper.Get("transA", 0);
    options.set("aTranspose", emscripten::val(transA == 1));
    const auto transB = helper.Get("transB", 0);
    options.set("bTranspose", emscripten::val(transB == 1));
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);
    options.set("alpha", alpha);
    options.set("beta", beta);

    // Add bias if present.
    if (input_defs.size() > 2) {
      options.set("c", model_builder.GetOperand(node.InputDefs()[c_idx]->Name()));
    }

    output = model_builder.GetBuilder().call<emscripten::val>("gemm", a, b, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool GemmOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                      const Node& node,
                                      const WebnnDeviceType /* device_type */,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs(node.InputDefs());
  const size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  std::vector<int64_t> a_shape;
  if (!GetShape(*input_defs[a_idx], a_shape, logger))
    return false;
  if (Product(a_shape) == 0) {
    LOGS(logger, VERBOSE) << "A must be non-empty";
    return false;
  }

  std::vector<int64_t> b_shape;
  if (!GetShape(*input_defs[b_idx], b_shape, logger))
    return false;
  if (Product(b_shape) == 0) {
    LOGS(logger, VERBOSE) << "B must be non-empty";
    return false;
  }

  if (op_type == "Gemm") {
    if (a_shape.size() != 2 || b_shape.size() != 2) {
      LOGS(logger, VERBOSE) << "A and B must be 2D for Gemm";
      return false;
    }

    // C of Gemm.
    if (input_defs.size() == 3) {
      std::vector<int64_t> c_shape;
      if (!GetShape(*input_defs[c_idx], c_shape, logger))
        return false;

      size_t c_dim = c_shape.size();

      if (c_dim > 1) {
        // TODO: Supports other shape of C.
        // Currently WebNN implementation in Chromium only supports 1-D C.
        return false;
      }
      if (c_dim == 0) {
        LOGS(logger, VERBOSE) << "C of Gemm is a scalar";
      } else {
        auto c_size = c_shape[c_dim - 1];
        NodeAttrHelper helper(node);
        const auto transB = helper.Get("transB", 0);
        if (c_size != (transB == 0 ? b_shape[1] : b_shape[0])) {
          LOGS(logger, VERBOSE) << "C of Gemm must be a vector of b_shape["
                                << (transB == 0 ? "1" : "0") << "]"
                                << " b_shape: [" << b_shape[0] << ", " << b_shape[1] << "]"
                                << " c_size: " << c_size;

          return false;
        }
      }
    }
  }

  return true;
}

bool GemmOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input0_type;  // A data type
  int32_t input1_type;  // B data type
  int32_t input2_type;  // C or a_zero_point data type
  int32_t input3_type;  // b_zero_point data type
  bool has_input2 = input_defs.size() > 2 && input_defs[2]->Exists();
  bool has_input3 = input_defs.size() > 3 && input_defs[3]->Exists();

  if (!GetType(*input_defs[0], input0_type, logger) ||
      !GetType(*input_defs[1], input1_type, logger) ||
      (has_input2 && !GetType(*input_defs[2], input2_type, logger)) ||
      (has_input3 && !GetType(*input_defs[3], input3_type, logger))) {
    return false;
  }

  std::string webnn_op_type;
  if (!GetWebNNOpType(op_type, webnn_op_type))
    return false;

  if (!IsSupportedDataType(input0_type, wnn_limits[webnn_op_type]["a"]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input type: [" << input0_type
                          << "] is not supported for now";
    return false;
  }

  if (input0_type != input1_type ||
      (has_input2 && input0_type != input2_type) ||
      (has_input3 && input0_type != input3_type)) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input data types should be the same.";
    return false;
  }

  return true;
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Gemm",
          "MatMul",
          "MatMulInteger",
      };

  op_registrations.builders.push_back(std::make_unique<GemmOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}
}  // namespace webnn
}  // namespace onnxruntime
