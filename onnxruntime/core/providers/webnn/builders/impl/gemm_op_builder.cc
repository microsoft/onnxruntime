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
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.
Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& /* logger */) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
  emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());
  emscripten::val output = emscripten::val::object();
  if (op_type == "MatMul") {
    output = model_builder.GetBuilder().call<emscripten::val>("matmul", a, b);
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
    output = model_builder.GetBuilder().call<emscripten::val>("matmulInteger", a, a_zero_point, b, b_zero_point);
  } else {  // Gemm
    emscripten::val options = emscripten::val::object();
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

bool GemmOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                      const Node& node,
                                      const WebnnDeviceType /* device_type */,
                                      const logging::Logger& logger) const {
  (void)initializers;
  const auto& op_type = node.OpType();
  const auto& input_defs(node.InputDefs());
  const size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  if (op_type == "Gemm") {
    std::vector<int64_t> a_shape;
    {
      if (!GetShape(*input_defs[a_idx], a_shape, logger))
        return false;

      if (a_shape.size() != 2) {
        LOGS(logger, VERBOSE) << "A must be 2D";
        return false;
      }

      if (Product(a_shape) == 0) {
        LOGS(logger, VERBOSE) << "A must be non-empty";
        return false;
      }
    }

    std::vector<int64_t> b_shape;
    {
      if (!GetShape(*input_defs[b_idx], b_shape, logger))
        return false;

      if (b_shape.size() != 2) {
        LOGS(logger, VERBOSE) << "B must be 2D";
        return false;
      }

      if (Product(b_shape) == 0) {
        LOGS(logger, VERBOSE) << "B must be non-empty";
        return false;
      }
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
