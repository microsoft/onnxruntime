// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
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

class GemmOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& op = node.OpType();
  const auto& input_defs(node.InputDefs());
  const bool is_gemm = op == "Gemm";

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    // we have to transpose the weight input of Gemm if transB is false, and potentially override the bias shape
    if (is_gemm) {
      NodeAttrHelper helper(node);
      const auto transB = helper.Get("transB", 0);
      if (transB == 0) {
        model_builder.AddInitializerToSkip(input_defs[1]->Name());
      }

      if (input_defs.size() > 2) {
        // ONNX spec requires B to be 2D and we required it to be a constant initializer so reading N this way is safe
        // B is {K, N] by default. or {N, K} if transB is true
        int N_dim = transB ? 0 : 1;
        int64_t N = input_defs[1]->Shape()->dim().at(N_dim).dim_value();

        const auto& bias_name = input_defs[2]->Name();
        const auto& bias = *model_builder.GetConstantInitializer(bias_name);
        if (bias.dims_size() != 1 || bias.dims(0) != N) {
          // we have to override the shape/duplicate data to convert {}, {1} or {1, N} to 1D {N}
          // when adding the Gemm operation so skip adding the original initializer
          model_builder.AddInitializerToSkip(bias_name);
        }
      }
    }
  } else
#endif  // defined(COREML_ENABLE_MLPROGRAM)
  {
    // We have already embedded the weights (matrix B and C(if any)) into the coreml layer
    // No need to copy them later to reduce memory consumption
    model_builder.AddInitializerToSkip(input_defs[1]->Name());
    if (is_gemm && input_defs.size() > 2) {
      model_builder.AddInitializerToSkip(input_defs[2]->Name());
    }
  }
}

// This is an internal function, requires input tensor to be 2d float tensor
// TODO, add support of other data types
static Status GetTensorFloatDataTransposed(const ONNX_NAMESPACE::TensorProto& tensor,
                                           std::vector<float>& transposed_data) {
  Initializer unpacked_tensor(tensor);
  auto src_data = unpacked_tensor.DataAsSpan<float>();
  const auto& tensor_shape = tensor.dims();
  auto x_t = SafeInt<size_t>(tensor_shape[0]);
  auto y_t = SafeInt<size_t>(tensor_shape[1]);
  transposed_data.resize(x_t * y_t);
  for (size_t x = 0; x < x_t; x++) {
    for (size_t y = 0; y < y_t; y++) {
      transposed_data[y * x_t + x] = src_data[x * y_t + y];
    }
  }

  return Status::OK();
}

Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& a = *input_defs[0];
  const auto& b = *input_defs[1];
  const auto* b_initializer = model_builder.GetConstantInitializer(b.Name());  // MLProgram MatMul may not be constant

  const bool is_matmul = op_type == "MatMul";
  const bool is_gemm = op_type == "Gemm";

  NodeAttrHelper helper(node);
  const auto transB = is_gemm ? helper.Get("transB", 0) : 0;

  std::vector<int64_t> b_shape;
  ORT_IGNORE_RETURN_VALUE(GetShape(b, b_shape, logger));
  int64_t b0 = -1, b1 = -1;

  // ML Program MatMul supports N-D input
  if (model_builder.CreateMLProgram() && is_matmul) {
    if (b_shape.size() == 1) {
      // B is treated as {b_shape[0], 1} according to the numpy rules.
      b0 = b_shape[0];
      b1 = 1;
    } else {
      // last 2 dims are used
      b0 = b_shape[b_shape.size() - 2];
      b1 = b_shape[b_shape.size() - 1];
    }
  } else {
    // we only support 2D input
    b0 = b_shape[0];
    b1 = b_shape[1];
  }

  // B is {K, N} in ONNX spec by default, or {N, K} in Gemm if transB is true
  const auto K = transB ? b1 : b0;
  const auto N = transB ? b0 : b1;

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    if (is_gemm) {
      // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.linear.linear
      auto gemm_op = model_builder.CreateOperation(node, "linear");
      AddOperationInput(*gemm_op, "x", a.Name());

      // CoreML takes weight input as {N, K} which is the reverse of ONNX.
      // if transB is true the input weight is {N, K} so can be added directly.
      if (transB) {
        AddOperationInput(*gemm_op, "weight", b.Name());
      } else {
        // transpose from {K, N} to {N, K}
        std::vector<float> weight_nk;
        std::vector<int64_t> weight_nk_shape = {N, K};
        ORT_RETURN_IF_ERROR(GetTensorFloatDataTransposed(*b_initializer, weight_nk));

        AddOperationInput(*gemm_op, "weight",
                          model_builder.AddConstant(gemm_op->type(), b.Name() + "_t", weight_nk, weight_nk_shape));
      }

      if (input_defs.size() == 3) {
        const auto& bias_arg = *input_defs[2];
        const auto& bias = *model_builder.GetConstantInitializer(bias_arg.Name());

        // CoreML linear op requires bias to be 1D tensor of size N
        if (bias.dims_size() == 1 && bias.dims().at(0) == N) {
          // can use existing initializer
          AddOperationInput(*gemm_op, "bias", bias_arg.Name());
        } else {
          Initializer unpacked_tensor(bias);
          auto bias_data = unpacked_tensor.DataAsSpan<float>();
          std::string_view bias_data_name;
          if (bias_data.size() == 1) {
            // expand scalar to N
            std::vector<float> expanded_bias_data(N, bias_data[0]);
            bias_data_name = model_builder.AddConstant(gemm_op->type(), "bias", expanded_bias_data);
          } else {
            // can use data as-is but need to adjust shape (inferred by AddConstant as {bias_data.size()})
            bias_data_name = model_builder.AddConstant(gemm_op->type(), "bias", bias_data);
          }

          AddOperationInput(*gemm_op, "bias", bias_data_name);
        }
      }

      AddOperationOutput(*gemm_op, *node.OutputDefs()[0]);
      model_builder.AddOperation(std::move(gemm_op));
    } else {
      // CoreML implementation is the same as ONNX MatMul.
      // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.linear.matmul
      auto matmul_op = model_builder.CreateOperation(node, "matmul");
      AddOperationInput(*matmul_op, "x", a.Name());
      AddOperationInput(*matmul_op, "y", b.Name());

      // once again the spec lies and says transpose_y and transpose_x are optional...
      auto false_value_name = model_builder.AddScalarConstant(matmul_op->type(), "false", false);
      AddOperationInput(*matmul_op, "transpose_x", false_value_name);
      AddOperationInput(*matmul_op, "transpose_y", false_value_name);

      AddOperationOutput(*matmul_op, *node.OutputDefs()[0]);
      model_builder.AddOperation(std::move(matmul_op));
    }
  } else
#endif  // defined(COREML_ENABLE_MLPROGRAM)
  {
    auto* coreml_inner_product = layer->mutable_innerproduct();

    *layer->mutable_input()->Add() = a.Name();

    coreml_inner_product->set_inputchannels(K);
    coreml_inner_product->set_outputchannels(N);

    // CoreML takes weight input as {N, K} which is the reverse of ONNX.
    // if Gemm's transB is true the input weight is {N, K} and can be added directly.
    if (transB) {
      ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*coreml_inner_product->mutable_weights(), *b_initializer));
    } else {
      std::vector<float> b_transposed;
      ORT_RETURN_IF_ERROR(GetTensorFloatDataTransposed(*b_initializer, b_transposed));
      CreateCoreMLWeight(*coreml_inner_product->mutable_weights(), b_transposed);
    }

    if (is_gemm && input_defs.size() > 2) {
      // Add bias
      coreml_inner_product->set_hasbias(true);
      const auto& bias_tensor = *model_builder.GetConstantInitializer(input_defs[2]->Name());

      // if scalar, or single value expand to 1D tensor of size N
      // IsOpSupportedImpl enforces it's scalar, {1}, {N}, or {1, N}.
      Initializer unpacked_tensor(bias_tensor);
      auto bias_data = unpacked_tensor.DataAsSpan<float>();
      if (bias_data.size() == 1 && N > 1) {
        std::vector<float> expanded_bias_data(N, bias_data[0]);
        CreateCoreMLWeight(*coreml_inner_product->mutable_bias(), expanded_bias_data);
      } else {
        CreateCoreMLWeight(*coreml_inner_product->mutable_bias(), bias_data);
      }
    }

    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();
    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

bool GemmOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs(node.InputDefs());
  const bool is_matmul = op_type == "MatMul";
  const bool is_gemm = op_type == "Gemm";

  size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  std::vector<int64_t> a_shape;
  if (!GetShape(*input_defs[a_idx], a_shape, logger)) {
    return false;
  }

  std::vector<int64_t> b_shape;
  if (!GetShape(*input_defs[b_idx], b_shape, logger)) {
    return false;
  }

  if (!input_params.graph_viewer.GetConstantInitializer(input_defs[b_idx]->Name())) {
    if (input_params.create_mlprogram && is_matmul) {
      // ML Program MatMul allows non-constant B input
    } else {
      LOGS(logger, VERBOSE) << op_type << " B input must be a constant initializer";
      return false;
    }
  }

  if (is_matmul) {
    if (input_params.create_mlprogram) {
      // ML Program matmul op has numpy semantics the same as the ONNX spec so we can use directly
    } else {
      // we could potentially support 1D and 3D if required. beyond 3D the dims that merge diverge.
      // https://github.com/apple/coremltools/blob/1931758aae383c83daddfc56f11a24a9d2bf4b87/coremltools/converters/onnx/_operators.py#L1607
      // https://github.com/apple/coremltools/blob/1931758aae383c83daddfc56f11a24a9d2bf4b87/coremltools/converters/mil/backend/nn/op_mapping.py#L1374
      // https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html#innerproductlayerparams
      if (a_shape.size() != 2 || b_shape.size() != 2) {
        LOGS(logger, VERBOSE) << "a and b inputs must be 2D. ";
        return false;
      }

      if (input_defs.size() > 2) {
        LOGS(logger, VERBOSE) << "MatMul with C input is not supported";
        return false;
      }
    }
  }

  if (is_gemm) {
    // A and B are 2D due to the ONNX spec
    NodeAttrHelper helper(node);
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);

    // TODO: We can support transA, alpha and beta by using multiple layers/operations if needed.
    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      LOGS(logger, VERBOSE) << "Only support for transA == 0, alpha == 1.0 "
                            << "and beta == 1.0 is currently implemented."
                            << " transA " << transA
                            << " alpha " << alpha
                            << " beta " << beta;
      return false;
    }

    if (input_defs.size() == 3) {
      if (!input_params.graph_viewer.GetConstantInitializer(input_defs[c_idx]->Name())) {
        LOGS(logger, VERBOSE) << "C of Gemm must be a constant initializer";
        return false;
      }

      std::vector<int64_t> c_shape;
      if (!GetShape(*input_defs[c_idx], c_shape, logger)) {
        return false;
      }

      // B is {K, N} in ONNX spec by default, or {N, K} in Gemm if transB is true
      const auto N = transB ? b_shape[0] : b_shape[1];

      size_t c_rank = c_shape.size();

      // allowed: scalar, or 1D where the value is 1 or N, 2D with shape {1, N}
      bool c_valid = false;
      switch (c_rank) {
        case 0:
          c_valid = true;
          break;
        case 1:
          if (c_shape[0] == 1 || c_shape[0] == N) {
            c_valid = true;
          }
          break;
        case 2:
          if (c_shape[0] == 1 && c_shape[1] == N) {
            c_valid = true;
          }
          break;
      }

      if (!c_valid) {
        LOGS(logger, VERBOSE) << "Shape of C Gemm input must be {}, {1}, {N}, or {1, N}. N:" << N << " C shape:"
                              << Shape2String(c_shape);

        return false;
      }
    }
  }

  return true;
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Gemm",
          "MatMul",
      };

  op_registrations.builders.push_back(std::make_unique<GemmOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace coreml
}  // namespace onnxruntime
