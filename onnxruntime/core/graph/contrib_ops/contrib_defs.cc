// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/contrib_ops/contrib_defs.h"

#include <cmath>
#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/attn_lstm_schema_defs.h"
#include "core/graph/contrib_ops/range_schema_defs.h"
#include "core/graph/op.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/signal_ops/signal_defs.h"
#include "core/graph/contrib_ops/onnx_function_util.h"
#include "onnx/defs/function.h"

namespace ONNX_NAMESPACE {
void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
void matmulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx);

void convTransposeWithDynamicPadsShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;  // Input tensor should have at least two dimensions.
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - size_t{2});

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      return;
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      return;
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  } else {
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] =
        (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;

  // Infer output shape if 'pads' tensor is available
  const auto* pads_initializer = ctx.getInputData(2);
  if (nullptr == pads_initializer) {
    return;
  }

  if (pads_initializer->dims_size() != 1 ||
      pads_initializer->data_type() != TensorProto::INT64)
    fail_shape_inference(
        "'pads' input must be a 1D (shape: [2 * n_input_dims]) tensor of type int64");

  pads = ParseData<int64_t>(pads_initializer);

  if (pads.size() != static_cast<size_t>(2 * n_input_dims))
    fail_shape_inference("Pads has incorrect number of values");

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) {  // Added only to one side.
      return;
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto final_output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1) *
      group;  // channels should be the second dim of second input multiply
              // group.

  int size_of_output;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
          // TODO: throw exception?
          return;  // output shape value cannot be smaller than the input shape
                   // value
        }
      }
      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim =
            strides[i] * (input_shape.dim(i + 2).dim_value() - 1) +
            output_padding[i] + effective_kernel_shape[i] - pads[i] -
            pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {
using namespace ONNX_NAMESPACE;
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void FusedMatMulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto transAAttr = ctx.getAttribute("transA");
  bool transa = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
  auto transBAttr = ctx.getAttribute("transB");
  bool transb = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
  auto trans_batch_a_attr = ctx.getAttribute("transBatchA");
  bool trans_batch_a = trans_batch_a_attr ? static_cast<int>(trans_batch_a_attr->i()) != 0 : false;
  auto trans_batch_b_attr = ctx.getAttribute("transBatchB");
  bool trans_batch_b = trans_batch_b_attr ? static_cast<int>(trans_batch_b_attr->i()) != 0 : false;
  int input1Idx = 0;
  int input2Idx = 1;
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0_raw = getInputShape(ctx, input1Idx);
  const auto shape1_raw = getInputShape(ctx, input2Idx);

  if (shape0_raw.dim_size() == 0 || shape1_raw.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  // numpy transpose on a vector does not change anything.
  if (shape0_raw.dim_size() == 1) {
    transa = false;
  }
  if (shape1_raw.dim_size() == 1) {
    transb = false;
  }

  ONNX_NAMESPACE::TensorShapeProto shape0, shape1;
  auto rank0 = shape0_raw.dim_size();
  if (rank0 == 1) {
    // for vector input, transa does not make impact on the dim.
    shape0 = shape0_raw;
  } else {
    int start = trans_batch_a ? 1 : 0;
    int end = trans_batch_a ? rank0 - 1 : rank0 - 2;
    for (int i = start; i < end; ++i) {
      *shape0.add_dim() = shape0_raw.dim(i);
    }
    *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 1 : (trans_batch_a ? 0 : rank0 - 2));
    *shape0.add_dim() = shape0_raw.dim(transa ? (trans_batch_a ? 0 : rank0 - 2) : rank0 - 1);
  }

  auto rank1 = shape1_raw.dim_size();
  if (rank1 == 1) {
    // for vector input, transb does not make impact on the dim.
    shape1 = shape1_raw;
  } else {
    int start = trans_batch_b ? 1 : 0;
    int end = trans_batch_b ? rank1 - 1 : rank1 - 2;
    for (int i = start; i < end; ++i) {
      *shape1.add_dim() = shape1_raw.dim(i);
    }
    *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 1 : (trans_batch_b ? 0 : rank1 - 2));
    *shape1.add_dim() = shape1_raw.dim(transb ? (trans_batch_b ? 0 : rank1 - 2) : rank1 - 1);
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() &&
        dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(
        prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }
  updateOutputShape(ctx, 0, resultShape);
}

// input1Idx - sparse matrix
// input2Idx - dense matrix.
// Output is dense
void sparseCompatibleMatmulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx) {
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0 = getInputShape(ctx, input1Idx);
  const auto shape1 = getInputShape(ctx, input2Idx);

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() &&
        dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(
        prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }

  // if the input 2 type was not previously propagate to output
  // we want to make sure that it is the tensor type of input 2
  auto default_tensor_type = ctx.getInputType(input2Idx)->value_case();
  updateOutputShape(ctx, 0, resultShape, default_tensor_type);
}


bool ParseScalar(const TensorProto* initializer, int& value) {
  std::vector<int32_t> parsed_data;
  if (initializer->data_type() == TensorProto::INT32) {
    const auto& data = ParseData<int32_t>(initializer);
    parsed_data.insert(parsed_data.end(), data.begin(), data.end());

    if (parsed_data.size() == 1) {
      value = parsed_data[0];
      return true;
    }
  }

  return false;
}

void BeamSearchShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (ctx.getNumOutputs() > 1) {
    // Here we assume that the third output exist only if second output exists.
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 5, 1);
    if (ctx.getNumOutputs() > 2) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 5, 2);
    }
  }

  // Shape inference
  // input 0 (input_ids) shape: (batch_size, sequence_length)
  // output 0 (sequences) shape: (batch_size, num_return_sequences, max_length)
  // output 1 (sequences_scores) shape: (batch_size, num_return_sequences)
  // output 2 (scores) shape: (max_length - sequence_length, batch_size, num_beams, vocab_size)
  if (!hasInputShape(ctx, 0)) {
    return;
  }
  auto& input_ids_shape = getInputShape(ctx, 0);
  auto& input_ids_dims = input_ids_shape.dim();
  if (input_ids_dims.size() != 2) {
    fail_shape_inference("Inputs 0 shall be 2 dimensions");
  }
  if (!(input_ids_dims[0].has_dim_value() && input_ids_dims[1].has_dim_value())) {
    return;
  }

  int64_t batch_size = input_ids_dims[0].dim_value();
  int64_t sequence_length = input_ids_dims[1].dim_value();

  const auto max_length = ctx.getInputData(1);
  const auto num_beams = ctx.getInputData(3);
  const auto num_return_sequences = ctx.getInputData(4);
  if (num_beams == nullptr || max_length == nullptr || num_return_sequences == nullptr) {  // not initializer
    return;
  }

  int max_length_value = 0;
  if (!ParseScalar(max_length, max_length_value) || max_length_value <= 0) {
    fail_shape_inference("Failed to parse max_length or it is not positive integer scalar");
  }

  int num_beams_value = 0;
  if (!ParseScalar(num_beams, num_beams_value) || num_beams_value <= 0) {
    fail_shape_inference("Failed to parse num_beams or it is not positive integer scalar");
  }

  int num_return_sequences_value = 0;
  if (!ParseScalar(num_return_sequences, num_return_sequences_value) || num_return_sequences_value <= 0) {
    fail_shape_inference("Failed to parse num_return_sequences or it is not positive integer scalar");
  }

  ONNX_NAMESPACE::TensorShapeProto sequences_shape;
  sequences_shape.add_dim()->set_dim_value(batch_size);
  sequences_shape.add_dim()->set_dim_value(num_return_sequences_value);
  sequences_shape.add_dim()->set_dim_value(max_length_value);
  updateOutputShape(ctx, 0, sequences_shape);

  if (ctx.getNumOutputs() > 1) {
    ONNX_NAMESPACE::TensorShapeProto sequences_scores_shape;
    sequences_shape.add_dim()->set_dim_value(batch_size);
    sequences_shape.add_dim()->set_dim_value(num_return_sequences_value);
    updateOutputShape(ctx, 1, sequences_scores_shape);

    if (ctx.getNumOutputs() > 2) {
      ONNX_NAMESPACE::TensorShapeProto scores_shape;
      scores_shape.add_dim()->set_dim_value(max_length_value - sequence_length);
      scores_shape.add_dim()->set_dim_value(batch_size);
      scores_shape.add_dim()->set_dim_value(num_beams_value);
      scores_shape.add_dim();  // vocab_size is unknown
      updateOutputShape(ctx, 2, scores_shape);
    }
  }
}

constexpr const char* Gelu_ver1_doc =
    R"DOC(Gaussian Error Linear Unit.
A high-performing neural network activation function.The GELU nonlinearity is
the expected transformation of a stochastic regularizer which randomly applies
the identity or zero map to a neuron's input. The GELU nonlinearity weights
inputs by their magnitude, rather than gates inputs by their sign as in ReLUs.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Gelu, 1,
                            OpSchema()
                                .SetDoc(Gelu_ver1_doc)
                                .Input(0, "X", "The input data as Tensor.", "T")
                                .Output(0, "Y", "The output.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
                                .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
                                  // gelu(x) = x * Phi(x) = x * 1/2(1+erf(x/sqrt(2)))
                                  auto* tp = ctx.getInputType(0);
                                  if ((tp == nullptr) || (!tp->has_tensor_type()))
                                    return false;
                                  auto elem_type = (TensorProto_DataType)(tp->tensor_type().elem_type());

                                  FunctionBuilder builder(functionProto);
                                  builder
                                      .AddOpset("", 13)
                                      .Const("Half", ToTensor(0.5, elem_type))
                                      .Const("One", ToTensor(1.0, elem_type))
                                      .Const("C", ToTensor(std::sqrt(0.5), elem_type))
                                      .Add(R"(
                CX = Mul (C, X)
                ERFCX = Erf (CX)
                ERFCXPlus1 = Add (ERFCX, One)
                PhiX = Mul (ERFCXPlus1, Half)
                Y = Mul (X, PhiX)
            )");

                                  schema.BuildFunction(functionProto);
                                  return true;
                                }));

constexpr const char* BiasGelu_ver1_doc =
    R"DOC(Bias Gelu.
It's an extension of Gelu. It takes the sum of input A and bias input B as the input of Gelu activation. )DOC";
ONNX_MS_OPERATOR_SET_SCHEMA(BiasGelu, 1,
                            OpSchema()
                                .SetDomain(kMSDomain)
                                .SinceVersion(1)
                                .SetDoc(BiasGelu_ver1_doc)
                                .Input(0, "A", "The normal input data.", "T")
                                .Input(1, "B", "The bias input data that is a 1D tensor.", "T")
                                .Output(0, "C", "The output.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

// Used to be ONNX 1.7 Inverse(12)
// Comment out docs not to increase the binary size
//
//  constexpr const char* Inverse_ver1_doc = R"DOC(
// Calculates inverse of a square matrix or batches of square matrices.
// Inverse takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
// and the inner-most 2 dimensions form square matrices. These matrices must be invertible (full-rank).
// The behavior where one of the matrices is not invertible is undefined. The implementation can choose
// to throw an error or output (garbage) results as is. The output is a tensor of shape `[*, M, M]`,
// containing the individual inverses of all input submatrices.
//)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Inverse, 1,
                            OpSchema()
                                .Input(0, "X", "Input tensor. Every matrix in the batch must be invertible.", "T")
                                .Output(0, "Y", "Output tensor of the same type and shape as the input tensor.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  using namespace ONNX_NAMESPACE;
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  // Shape inference
                                  if (hasInputShape(ctx, 0)) {
                                    const TensorShapeProto& input_shape =
                                        ctx.getInputType(0)->tensor_type().shape();
                                    const int rank = static_cast<int>(input_shape.dim_size());

                                    if (rank < 2) {
                                      fail_shape_inference("Input rank must be >= 2.")
                                    }

                                    const auto mat_w = input_shape.dim(rank - 1);
                                    const auto mat_h = input_shape.dim(rank - 2);
                                    if (mat_w.has_dim_value() && mat_h.has_dim_value() &&
                                        (mat_w.dim_value() != mat_h.dim_value())) {
                                      fail_shape_inference(
                                          "The inner-most 2 dimensions must have the same size (mat_w:",
                                          mat_w.dim_value(),
                                          " != mat_h:",
                                          mat_h.dim_value(),
                                          ").");
                                    }

                                    // Shape inference
                                    propagateShapeFromInputToOutput(ctx, 0, 0);
                                  }
                                }));

constexpr const char* TorchEmbedding_ver1_doc = R"DOC(
      Based on Torch operator Embedding, creates a lookup table of embedding vectors of fixed size,
       for a dictionary of fixed size.
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(TorchEmbedding, 1,
                            OpSchema()
                                .SetDoc(TorchEmbedding_ver1_doc)
                                .Input(
                                    0,
                                    "weight",
                                    "The embedding matrix of size N x M. 'N' is equal to the maximum possible index + 1, and 'M' is "
                                    "equal to the embedding size",
                                    "T")
                                .Input(
                                    1,
                                    "indices",
                                    "Long tensor containing the indices to extract from embedding matrix.",
                                    "tensor(int64)")
                                .Input(
                                    2,
                                    "padding_idx",
                                    "A 0-D scalar tensor. If specified, the entries at `padding_idx` do not contribute to the gradient; "
                                    "therefore, the embedding vector at `padding_idx` is not updated during training, "
                                    "i.e. it remains as a fixed pad.",
                                    "tensor(int64)",
                                    OpSchema::Optional)
                                .Input(
                                    3,
                                    "scale_grad_by_freq",
                                    "A 0-D bool tensor. If given, this will scale gradients by the inverse of frequency of "
                                    "the indices (words) in the mini-batch. Default  is ``False``",
                                    "tensor(bool)",
                                    OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "Output tensor of the same type as the input tensor. Shape of the output is * x M, where '*' is the shape of "
                                    "input indices, and 'M' is the embedding size.",
                                    "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)",
                                     "tensor(bfloat16)",
                                     "tensor(uint8)",
                                     "tensor(uint16)",
                                     "tensor(uint32)",
                                     "tensor(uint64)",
                                     "tensor(int8)",
                                     "tensor(int16)",
                                     "tensor(int32)",
                                     "tensor(int64)"},
                                    "Constrain input and output types to all numeric tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  using namespace ONNX_NAMESPACE;
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  TensorShapeProto outputs_shape;
                                  Dim input_dim_i;

                                  if (hasInputShape(ctx, 1)) {
                                    auto& input_shape = getInputShape(ctx, 1);
                                    for (int32_t i = 0; i < input_shape.dim_size(); i++) {
                                      input_dim_i = input_shape.dim(i);
                                      *outputs_shape.add_dim() = input_dim_i;
                                    }
                                  }

                                  Dim embedding_dim;
                                  unifyInputDim(ctx, 0, 1, embedding_dim);
                                  *outputs_shape.add_dim() = embedding_dim;
                                  updateOutputShape(ctx, 0, outputs_shape);
                                }));

constexpr const char* Trilu_ver1_doc = R"DOC(
      Returns the upper or lower triangular part of a 2-D matrix, or batches of 2-D matrices. If the attribute "upper" is set to true,
      the upper triangular matrix is retained. Lower triangular matrix is retained otherwise. Default value for upper is true.
      Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
      of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
      All other elements in the matrix are set to zero.
      If k = 0, the triangular part on and above/below the main diagonal is retained.
      If upper is set to true, a positive k retains the upper triangular matrix excluding k diagonals above
      the main diagonal. A negative k value includes as many diagonals below the main diagonal.
      If upper is set to false, a positive k retains the lower triangular matrix including k diagonals above
      the main diagonal. A negative k value excludes as many diagonals below the main diagonal.
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Trilu, 1,
                            OpSchema()
                                .SetDoc(Trilu_ver1_doc)
                                .Attr("upper",
                                      "Boolean. Indicates whether upper or lower part of matrix is retained. Default is true.",
                                      AttributeProto::INT,
                                      static_cast<int64_t>(1))
                                .Input(
                                    0,
                                    "X",
                                    "Input tensor of rank 2 or higher.",
                                    "T")
                                .Input(
                                    1,
                                    "k",
                                    "A 0-D tensor containing a single value corresponding to the number diagonals above or the main diagonal to exclude or include."
                                    "Default value is 0 if it's not specified.",
                                    "tensor(int64)",
                                    OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "Output tensor of the same type and shape as the input tensor.",
                                    "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)",
                                     "tensor(bfloat16)",
                                     "tensor(uint8)",
                                     "tensor(uint16)",
                                     "tensor(uint32)",
                                     "tensor(uint64)",
                                     "tensor(int8)",
                                     "tensor(int16)",
                                     "tensor(int32)",
                                     "tensor(int64)",
                                     "tensor(bool)"},
                                    "Constrain input and output types to all numeric tensors and bool tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  using namespace ONNX_NAMESPACE;
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  if (hasInputShape(ctx, 0)) {
                                    const TensorShapeProto& input_shape =
                                        ctx.getInputType(0)->tensor_type().shape();
                                    const int rank = static_cast<int>(input_shape.dim_size());
                                    if (rank < 2) {
                                      fail_shape_inference("Input rank must be >= 2.")
                                    }
                                    propagateShapeFromInputToOutput(ctx, 0, 0);
                                  }
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(BiasSoftmax, 1,
                            OpSchema()
                                .SetDoc(
                                    "Y = softmax(scores + bias)) with simple broadcast on bias. "
                                    "Intended to specialize softmax(scores + additive_mask) commonly found in transformer models.")
                                .Attr("softmax_axis", "apply softmax to elements for dimensions softmax_axis or higher", AttributeProto::INT, static_cast<int64_t>(1))
                                .Attr("broadcast_axis", "broadcast bias across input for dimensions broadcast_axis to softmax_axis-1", AttributeProto::INT, static_cast<int64_t>(1))
                                .Input(0, "data", "The input data as Tensor.", "T")
                                .Input(1, "bias", "The bias (or mask) as Tensor.", "T")
                                .Output(0, "output", "The output.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_MS_OPERATOR_SET_SCHEMA(BiasDropout, 1,
                            OpSchema()
                                .SetDoc(
                                    "output, dropout_mask = Dropout(data + bias, ratio) + residual, "
                                    "Intended to specialize the dropout pattern commonly found in transformer models.")
                                .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.", AttributeProto::INT, OPTIONAL_VALUE)
                                .AllowUncheckedAttributes()
                                .Input(0, "data", "The input data as Tensor.", "T")
                                .Input(1, "bias", "The bias input, a vector with the same shape as last dim of data OR same shape with data", "T")
                                .Input(2, "residual", "The residual input, must have the same shape as data", "T", OpSchema::Optional)
                                .Input(3, "ratio",
                                       "The ratio of random dropout, with value in [0, 1). If this input was not set, "
                                       "or if it was set to 0, the output would be a simple copy of the input. "
                                       "If it's non-zero, output will be a random dropout of input, which is typically "
                                       "the case during training.",
                                       "T1",
                                       OpSchema::Optional)
                                .Input(4, "training_mode",
                                       "If set to true then it indicates dropout is being used for "
                                       "training. It is an optional value hence unless specified explicitly, it is false. "
                                       "If it is false, ratio is ignored and the operation mimics inference mode where nothing "
                                       "will be dropped from the input data and if mask is requested as output it will contain "
                                       "all ones.",
                                       "T2",
                                       OpSchema::Optional)
                                .Output(0, "output", "The output.", "T")
                                .Output(1, "mask", "The output mask of dropout.", "T2", OpSchema::Optional)
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input 'ratio' types to float tensors.")
                                .TypeConstraint(
                                    "T2",
                                    {"tensor(bool)"},
                                    "Constrain output 'mask' types to boolean tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateShapeAndTypeFromFirstInput(ctx);
                                  if (ctx.getNumOutputs() == 2) {
                                    updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
                                    if (hasNInputShapes(ctx, 1)) {
                                      propagateShapeFromInputToOutput(ctx, 0, 1);
                                    }
                                  }
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(IsAllFinite, 1,
                            OpSchema()
                                .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
                                .SetDoc("IsAllFinite")
                                .SetDomain(kMSDomain)
                                .SinceVersion(1)
                                .Attr("isinf_only",
                                      "If true, check only for Inf, -Inf.",
                                      AttributeProto::INT,
                                      static_cast<int64_t>(0))
                                .Attr("isnan_only",
                                      "If true, check only for NaN.",
                                      AttributeProto::INT,
                                      static_cast<int64_t>(0))
                                .TypeConstraint(
                                    "V",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(bool)"},
                                    "Constrain the output to a boolean tensor.")
                                .Input(0, "input", "Input tensors to check.", "V",
                                       OpSchema::Variadic)
                                .Output(
                                    0,
                                    "output",
                                    "The output scalar. Its value is true if all input "
                                    "tensors are finite. Otherwise, the output value would "
                                    "be false.",
                                    "T")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  bool isinf_only = static_cast<bool>(getAttribute(ctx, "isinf_only", int64_t(0)));
                                  bool isnan_only = static_cast<bool>(getAttribute(ctx, "isnan_only", int64_t(0)));
                                  if (isinf_only && isnan_only) {
                                    fail_shape_inference("Both attributes isinf_only and isnan_only cannot be set. Unset both to check for both conditions.");
                                  }
                                  updateOutputShape(ctx, 0, {});
                                  updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
                                }));

constexpr const char* GridSample_ver1_doc = R"DOC(
      Given an `input` and a flow-field `grid`, computes the `output` using `input` values and pixel locations from `grid`.
      Currently, only spatial (4-D) inputs are supported. For `input` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
      the `output` will have shape (N, C, H_out, W_out).
      For each output location `output[n, :, h, w]`, the size-2 vector `grid[n, h, w]` specifies `input` pixel locations `x` and `y`,
      which are used to interpolate the output value `output[n, :, h, w]`.
      The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
      See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample).
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(GridSample, 1,
                            OpSchema()
                                .SetDoc(GridSample_ver1_doc)
                                .Attr(
                                    "mode",
                                    "Three interpolation modes: bilinear (default), nearest and bicubic.",
                                    AttributeProto::STRING,
                                    std::string("bilinear"))
                                .Attr(
                                    "padding_mode",
                                    "Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. "
                                    "zeros: use 0 for out-of-bound grid locations, "
                                    "border: use border values for out-of-bound grid locations, "
                                    "reflection: use values at locations reflected by the border for out-of-bound grid locations.",
                                    AttributeProto::STRING,
                                    std::string("zeros"))
                                .Attr(
                                    "align_corners",
                                    "If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. "
                                    "If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic.",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Input(
                                    0,
                                    "X",
                                    "4-D tensor of shape (N, C, H, W), "
                                    "where N is the batch size, C is the numbers of channels, "
                                    "H and W are the height and width of the input data.",
                                    "T1")
                                .Input(
                                    1,
                                    "Grid",
                                    "Input offset, 4-D tensor of shape (N, H_out, W_out, 2), "
                                    "where H_out and W_out are the height and width of grid and output, "
                                    "Grid specifies the sampling pixel locations normalized by the input spatial dimensions. "
                                    "Therefore, it should have most values in the range of [-1, 1]. "
                                    "If grid has values outside the range of [-1, 1], the corresponding outputs will be handled as defined by padding_mode.",
                                    "T1")
                                .Output(
                                    0,
                                    "Y",
                                    "4-D tensor of shape (N, C, H_out, W_out).",
                                    "T2")
                                .TypeConstraint(
                                    "T1",
                                    OpSchema::all_tensor_types(),
                                    "Constrain input types to all tensor types.")
                                .TypeConstraint(
                                    "T2",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  size_t input_param = 0, grid_param = 1;

                                  checkInputRank(ctx, input_param, 4);
                                  checkInputRank(ctx, grid_param, 4);

                                  // Output dimensions, initialized to an unknown-dimension-value
                                  Dim N, C, H_out, W_out;

                                  // Get value of N from dim 0 of input_param, if available
                                  unifyInputDim(ctx, input_param, 0, N);
                                  // Get value of C from dim 1 of input_param, if available
                                  unifyInputDim(ctx, input_param, 1, C);

                                  // Get value of H_out from dim 1 of grid_param, if available
                                  unifyInputDim(ctx, grid_param, 1, H_out);
                                  // Get value of W_out from dim 2 of grid_param, if available
                                  unifyInputDim(ctx, grid_param, 2, W_out);

                                  // set output shape:
                                  updateOutputShape(ctx, 0, {N, C, H_out, W_out});
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(BeamSearch, 1,
                            OpSchema()
                                .SetDoc("Beam Search for text generation. Supports GPT-2 decoder.")
                                .Attr("eos_token_id", "The id of the end-of-sequence token", AttributeProto::INT)
                                .Attr("pad_token_id", "The id of the padding token", AttributeProto::INT)
                                .Attr("no_repeat_ngram_size", "no repeat ngrams size", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("early_stopping", "early stop or not", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("model_type", "model type: 0 for GPT-2; 1 for encoder decoder like T5", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("encoder_decoder_init", "subgraph for initialization of encoder and decoder. It will be called once before decoder subgraph.", AttributeProto::GRAPH, OPTIONAL_VALUE)
                                .Attr("decoder", "Decoder subgraph to execute in a loop.", AttributeProto::GRAPH)
                                .Input(0, "input_ids", "The sequence used as a prompt for the generation. Shape is (batch_size, sequence_length)", "I")
                                .Input(1, "max_length", "The maximum length of the sequence to be generated. Shape is (1)", "I")
                                .Input(2, "min_length", "The minimum length below which the score of eos_token_id is set to -Inf. Shape is (1)", "I", OpSchema::Optional)
                                .Input(3, "num_beams", "Number of beams for beam search. 1 means no beam search. Shape is (1)", "I")
                                .Input(4, "num_return_sequences", "The number of returned sequences in the batch. Shape is (1)", "I")
                                .Input(5, "temperature", "The value used to module the next token probabilities. Accepts value > 0.0. Shape is (1)", "T")
                                .Input(6, "length_penalty",
                                       "Exponential penalty to the length. Default value 1.0 means no penalty."
                                       "Value > 1.0 encourages longer sequences, while values < 1.0 produces shorter sequences."
                                       "Shape is (1,)",
                                       "T", OpSchema::Optional)
                                .Input(7, "repetition_penalty", "The parameter for repetition penalty. Default value 1.0 means no penalty. Accepts value > 0.0. Shape is (1)", "T", OpSchema::Optional)
                                .Input(8, "vocab_mask", "Mask of vocabulary. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (vacab_size)", "M", OpSchema::Optional)
                                .Input(9, "prefix_vocab_mask", "Mask of vocabulary for first step. Words that masked with 0 are not allowed to be generated, and 1 is allowed. Shape is (batch_size, vocab_size)", "M", OpSchema::Optional)
                                .Output(0, "sequences", "Word IDs of generated sequences. Shape is (batch_size, num_return_sequences, max_sequence_length)", "I")
                                .Output(1, "sequences_scores", "Final beam score of the generated sequences. Shape is (batch_size, num_return_sequences)", "T", OpSchema::Optional)
                                .Output(2, "scores",
                                        "Processed beam scores for each vocabulary token at each generation step."
                                        "Beam scores consisting of log softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam."
                                        "Shape is (max_length - sequence_length, batch_size, num_beams, vocab_size)",
                                        "T", OpSchema::Optional)
                                .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
                                .TypeConstraint("I", {"tensor(int32)"}, "Constrain to integer types")
                                .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask to integer types")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  BeamSearchShapeInference(ctx);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(SampleOp, 1,
                            OpSchema()
                                .Input(0, "X", "input", "T")
                                .Output(0, "Y", "output", "T")
                                .TypeConstraint(
                                    "T",
                                    ONNX_NAMESPACE::OpSchema::numeric_types_for_math_reduction(),
                                    "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
                                .SetDoc(R"DOC(
Sample echo operator.)DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(MaxpoolWithMask, 1,
                            OpSchema()
                                .SetDoc(R"DOC(For internal use.)DOC")
                                .Attr(
                                    "auto_pad",
                                    "",
                                    AttributeProto::STRING,
                                    std::string("NOTSET"))
                                .Attr(
                                    "kernel_shape",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr("pads",
                                      "",
                                      AttributeProto::INTS, OPTIONAL_VALUE)
                                .Attr(
                                    "storage_order",
                                    "",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                .Input(
                                    0,
                                    "X",
                                    "",
                                    "T")
                                .Input(1, "M", "mask", "tensor(int32)")
                                .Output(
                                    0,
                                    "Y",
                                    "",
                                    "T")
                                .TypeConstraint("T", {"tensor(float)"}, "Constrain input0 and output types to float tensors")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  ONNX_NAMESPACE::convPoolShapeInference(ctx, false, true, 0, 1);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(Rfft, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Input(0, "X", "input tensor", "T")
                                .Attr("signal_ndim", "", AttributeProto::INT, static_cast<int64_t>(1))
                                .Attr("normalized", "", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("onesided", "", AttributeProto::INT, static_cast<int64_t>(1))
                                .Output(0, "Y", "output tensor", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(Irfft, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Input(0, "X", "input tensor", "T")
                                .Attr("signal_ndim", "", AttributeProto::INT)
                                .Attr("normalized", "", AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("onesided", "", AttributeProto::INT, static_cast<int64_t>(1))
                                .Output(0, "Y", "output tensor", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(ComplexMul, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Input(0, "A", "input_0", "T")
                                .Input(1, "B", "input_1", "T")
                                .Output(0, "C", "output tensor", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(ComplexMulConj, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Input(0, "A", "input_0", "T")
                                .Input(1, "B", "input_1", "T")
                                .Output(0, "C", "output tensor", "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors."));

ONNX_MS_OPERATOR_SET_SCHEMA(ConvTransposeWithDynamicPads, 1,
                            OpSchema()
                                .SetDoc(R"DOC()DOC")
                                .Attr(
                                    "kernel_shape",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr("output_padding",
                                      "",
                                      AttributeProto::INTS,
                                      OPTIONAL_VALUE)
                                .Attr(
                                    "dilations",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "strides",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "auto_pad",
                                    "",
                                    AttributeProto::STRING,
                                    std::string("NOTSET"))
                                .Attr(
                                    "group",
                                    "",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(1))
                                .Input(
                                    0,
                                    "X",
                                    "",
                                    "T")
                                .Input(
                                    1,
                                    "W",
                                    "",
                                    "T")
                                .Input(2, "Pads", "", "tensor(int64)", OpSchema::Optional)
                                .Input(3, "B", "", "T", OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "",
                                    "T")
                                .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::convTransposeWithDynamicPadsShapeInference));

ONNX_MS_OPERATOR_SET_SCHEMA(FusedConv, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
The fused convolution operator schema is the same as Conv besides it includes an attribute
activation.)DOC")
                                .Attr(
                                    "auto_pad",
                                    "",
                                    AttributeProto::STRING,
                                    std::string("NOTSET"))
                                .Attr(
                                    "kernel_shape",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "dilations",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "strides",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "pads",
                                    "",
                                    AttributeProto::INTS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "group",
                                    "",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(1))
                                .Attr(
                                    "activation",
                                    "",
                                    AttributeProto::STRING,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_params",
                                    "",
                                    AttributeProto::FLOATS,
                                    OPTIONAL_VALUE)
                                .Input(
                                    0,
                                    "X",
                                    "",
                                    "T")
                                .Input(
                                    1,
                                    "W",
                                    "",
                                    "T")
                                .Input(
                                    2,
                                    "B",
                                    "",
                                    "T",
                                    OpSchema::Optional)
                                .Input(
                                    3,
                                    "Z",
                                    "",
                                    "T",
                                    OpSchema::Optional)
                                .Output(
                                    0,
                                    "Y",
                                    "",
                                    "T")
                                .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(FusedGemm, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
The FusedGemm operator schema is the same as Gemm besides it includes attributes
activation and leaky_relu_alpha.)DOC")
                                .Input(
                                    0,
                                    "A",
                                    "Input tensor A. "
                                    "The shape of A should be (M, K) if transA is 0, "
                                    "or (K, M) if transA is non-zero.",
                                    "T")
                                .Input(
                                    1,
                                    "B",
                                    "Input tensor B. "
                                    "The shape of B should be (K, N) if transB is 0, "
                                    "or (N, K) if transB is non-zero.",
                                    "T")
                                .Input(
                                    2,
                                    "C",
                                    "Input tensor C. "
                                    "The shape of C should be unidirectional broadcastable to (M, N).",
                                    "T")
                                .Output(0, "Y", "Output tensor of shape (M, N).", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)",
                                     "tensor(float)",
                                     "tensor(double)",
                                     "tensor(uint32)",
                                     "tensor(uint64)",
                                     "tensor(int32)",
                                     "tensor(int64)"},
                                    "Constrain input and output types to float/int tensors.")
                                .Attr(
                                    "transA",
                                    "Whether A should be transposed",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "transB",
                                    "Whether B should be transposed",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "alpha",
                                    "Scalar multiplier for the product of input tensors A * B.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "beta",
                                    "Scalar multiplier for input tensor C.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "activation",
                                    "",
                                    AttributeProto::STRING,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_alpha",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_beta",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "activation_gamma",
                                    "",
                                    AttributeProto::FLOAT,
                                    OPTIONAL_VALUE)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  if (hasNInputShapes(ctx, 2)) {
                                    auto transAAttr = ctx.getAttribute("transA");
                                    bool transA =
                                        transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
                                    auto transBAttr = ctx.getAttribute("transB");
                                    bool transB =
                                        transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
                                    auto& first_input_shape = getInputShape(ctx, 0);
                                    auto& second_input_shape = getInputShape(ctx, 1);
                                    if (first_input_shape.dim_size() != 2)
                                      fail_shape_inference("First input does not have rank 2");
                                    if (second_input_shape.dim_size() != 2)
                                      fail_shape_inference("Second input does not have rank 2");
                                    updateOutputShape(
                                        ctx,
                                        0,
                                        {first_input_shape.dim(transA ? 1 : 0),
                                         second_input_shape.dim(transB ? 0 : 1)});
                                  }
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(ExpandDims, 1,
                            OpSchema()
                                .Input(0, "X", "input", "T")
                                .Input(1, "axis", "Specified axis to insert a dimension", "tensor(int32)")
                                .Output(0, "Y", "output", "T")
                                .TypeConstraint(
                                    "T",
                                    ONNX_NAMESPACE::OpSchema::all_tensor_types(),
                                    "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  // Shape inference
                                  if (!hasInputShape(ctx, 0))
                                    return;

                                  auto& input_shape = getInputShape(ctx, 0);
                                  const int rank = input_shape.dim_size();
                                  const ONNX_NAMESPACE::TensorProto* axis_initializer = ctx.getInputData(1);
                                  if (!axis_initializer)
                                    return;
                                  const int axis = axis_initializer->int32_data()[0];
                                  if (axis > rank || axis < -rank - 1) {
                                    fail_shape_inference("Input axis is invalid: ", axis);
                                  }
                                  int pos = axis >= 0 ? axis : rank + axis - 1;
                                  ONNX_NAMESPACE::TensorShapeProto output_shape;
                                  for (int i = 0; i < pos; ++i) {
                                    output_shape.add_dim();
                                    *(output_shape.mutable_dim(i)) = input_shape.dim(i);
                                  }
                                  output_shape.add_dim();
                                  output_shape.mutable_dim(pos)->set_dim_value(1);
                                  for (int i = pos + 1; i < rank + 1; ++i) {
                                    output_shape.add_dim();
                                    *(output_shape.mutable_dim(i)) = input_shape.dim(i - 1);
                                  }
                                  updateOutputShape(ctx, 0, output_shape);
                                })
                                .SetDoc(R"DOC(ExpandDims echo operator.)DOC"));

constexpr const char* Tokenizer_ver1_doc = R"DOC(
  Tokenizer divides each string in X into a vector of strings along the last axis. Allowed input shapes are [C] and [N, C].
  If the maximum number of tokens found per input string is D, the output shape would be [N, C, D] when input shape is [N, C].
  Similarly, if input shape is [C] then the output should be [C, D]. Tokenizer has two different operation modes.
  The first mode is selected when "tokenexp" is not set and "separators" is set. If "tokenexp" is set and "separators" is not set,
  the second mode will be used. The first mode breaks each input string into tokens by matching and removing separators.
  "separators" is a list of strings which are regular expressions. "tokenexp" is a single regular expression.
  Let's assume "separators" is [" "] and consider an example.
  If input is
  ["Hello World", "I love computer science !"] whose shape is [2],
  then the output would be
 [["Hello", "World", padvalue, padvalue, padvalue],
 ["I", "love", "computer", "science", "!"]]
 whose shape is [2, 5] because you can find at most 5 tokens per input string.
 Note that the input at most can have two axes, so 3-D and higher dimension are not supported.
 If "separators" contains a single empty string, the Tokenizer will enter into character tokenezation mode. This means all strings
 will be broken part into individual characters.
 For each input string, the second mode searches matches of "tokenexp" and each match will be a token in Y.
 The matching of "tokenexp" is conducted greedily (i.e., a match should be as long as possible).
 This operator searches for the first match starting from the beginning of the considered string,
 and then launches another search starting from the first remained character after the first matched token.
 If no match found, this operator will remove the first character from the remained string and do another search.
 This procedure will be repeated until reaching the end of the considered string.
  Let's consider another example to illustrate the effect of setting "mark" to true.
  If input is ["Hello", "World"],
  then the corresponding output would be [0x02, "Hello", "World", 0x03].
  This implies that if mark is true, [C]/[N, C] - input's output shape becomes [C, D+2]/[N, C, D+2].
If tokenizer removes the entire content of [C]-input, it will produce [[]].
I.e. the output shape should be [C][0] or [N][C][0] if input shape was [N][C].
If the tokenizer receives empty input of [0] then the output is [0] if empty input
of [N, 0] then [N, 0].
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(Tokenizer, 1,
                            OpSchema()
                                .Input(0, "X", "Strings to tokenize", "T")
                                .Output(0, "Y", "Tokenized strings", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(string)"},
                                    "Input/Output is a string tensor")
                                .Attr(
                                    "mark",
                                    "Boolean whether to mark the beginning/end character with start of text character (0x02)/end of text character (0x03).",
                                    AttributeProto::INT)
                                .Attr(
                                    "pad_value",
                                    "The string used to pad output tensors when the tokens extracted doesn't match the maximum number of tokens found. If start/end markers are needed, padding will appear outside the markers.",
                                    AttributeProto::STRING)
                                .Attr(
                                    "tokenexp",
                                    "An optional string. Token's regular expression in basic POSIX format"
                                    " (pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html#tag_09_03)."
                                    " If set, tokenizer may produce tokens matching the specified pattern. Note that one and only of"
                                    " 'tokenexp' and 'separators' should be set.",
                                    AttributeProto::STRING,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "separators",
                                    "an optional list of strings attribute that contains a list of separators - regular expressions to match separators"
                                    " Two consecutive segments in X connected by a separator would be divided into two tokens."
                                    " For example, if the input is \"Hello World!\" and this attribute contains only one space character,"
                                    " the corresponding output would be [\"Hello\", \"World!\"]. To achieve character-level tokenization,"
                                    " one should set the 'separators' to [\"\"], which contains an empty string.",
                                    AttributeProto::STRINGS,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "mincharnum",
                                    "Minimum number of characters allowed in the output. For example, if mincharnum is 2, tokens such as \"A\" and \"B\" would be ignored",
                                    AttributeProto::INT)
                                .SetDoc(Tokenizer_ver1_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  // Shape inference
                                  if (!hasInputShape(ctx, 0))
                                    return;

                                  ONNX_NAMESPACE::TensorShapeProto output_shape;
                                  auto& input_shape = getInputShape(ctx, 0);
                                  auto& dims = input_shape.dim();
                                  if (dims.size() < 1 || dims.size() > 2) {
                                    fail_shape_inference("Input dimensions are either [C] or [N][C] allowed");
                                  }

                                  int64_t size = 1;
                                  for (auto& dim : dims) {
                                    if (utils::HasDimValue(dim)) {
                                      size *= dim.dim_value();
                                    }
                                  }

                                  if (size > 0) {
                                    for (auto& dim : dims) {
                                      *output_shape.add_dim() = dim;
                                    }
                                    // Add the last unknown dimension
                                    // only if the input is not empty
                                    output_shape.add_dim();
                                  } else if (size == 0) {
                                    if (dims.size() == 2) {
                                      *output_shape.add_dim() = dims[0];
                                    }
                                    output_shape.add_dim()->set_dim_value(0);
                                  }
                                  updateOutputShape(ctx, 0, output_shape);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(MatMulInteger16, 1,
                            OpSchema()
                                .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
 The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.)DOC")
                                .Input(0, "A", "N-dimensional matrix A", "T1")
                                .Input(1, "B", "N-dimensional matrix B", "T2")
                                .Output(0, "Y", "Matrix multiply results from A * B", "T3")
                                .TypeConstraint("T1", {"tensor(int16)", "tensor(uint16)"}, "Constrain input A data types as 16-bit integer tensor")
                                .TypeConstraint("T2", {"tensor(int16)", "tensor(uint16)"}, "Constrain input B data types as 16-bit integer tensor")
                                .TypeConstraint("T3",
                                                {"tensor(int32)", "tensor(uint32)"},
                                                "Constrain output Y data types as 32-bit integer tensor."
                                                "T3 must be tensor(uint32) when both T1 and T2 are tensor(uint16),"
                                                "or must be tensor(int32) when either T1 or T2 is tensor(int16).")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  auto a_type = ctx.getInputType(0);
                                  auto b_type = ctx.getInputType(1);
                                  auto y_type = ctx.getOutputType(0);
                                  if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
                                      a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
                                      b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
                                    fail_type_inference(
                                        "inputs are expected to have tensor type and output type should not be null.");
                                  }

                                  // Right now we only support int32
                                  y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

                                  ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
                                }));

constexpr const char* TransposeMatMul_doc = R"DOC(
Duplicate of FusedMatMul. Going forward FusedMatMul should be used. This OP will be supported for backward compatibility.
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

constexpr const char* FusedMatMul_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(TransposeMatMul, 1,
                            OpSchema()
                                .Input(0, "A", "N-dimensional matrix A", "T")
                                .Input(1, "B", "N-dimensional matrix B", "T")
                                .Attr(
                                    "alpha",
                                    "Scalar multiplier for the product of the input tensors.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "transA",
                                    "Whether A should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "transB",
                                    "Whether B should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Output(0, "Y", "Matrix multiply results", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                    "Constrain input and output types to float tensors.")
                                .SetDoc(TransposeMatMul_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  FusedMatMulShapeInference(ctx);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(FusedMatMul, 1,
                            OpSchema()
                                .Input(0, "A", "N-dimensional matrix A", "T")
                                .Input(1, "B", "N-dimensional matrix B", "T")
                                .Attr("alpha", "Scalar multiplier for the product of the input tensors.", AttributeProto::FLOAT, 1.0f)
                                .Attr("transA", "Whether A should be transposed on the last two dimensions before doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transB", "Whether B should be transposed on the last two dimensions before doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transBatchA",
                                      "Whether A should be transposed on the 1st dimension and batch dimensions (dim-1 to dim-rank-2) before "
                                      "doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Attr("transBatchB",
                                      "Whether B should be transposed on the 1st dimension and batch dimensions (dim-1 to dim-rank-2) before "
                                      "doing multiplication",
                                      AttributeProto::INT, static_cast<int64_t>(0))
                                .Output(0, "Y", "Matrix multiply results", "T")
                                .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
                                                "Constrain input and output types to float tensors.")
                                .SetDoc(FusedMatMul_doc)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) { FusedMatMulShapeInference(ctx); }));

ONNX_MS_OPERATOR_SET_SCHEMA(SparseToDenseMatMul, 1,
                            OpSchema()
                                .Input(0, "A", "2-dimensional sparse matrix A. Either COO or CSR format", "T")
                                .Input(1, "B", "N-dimensional dense matrix B", "T1")
                                .Attr(
                                    "alpha",
                                    "Scalar multiplier for the product of the input tensors.",
                                    AttributeProto::FLOAT,
                                    1.0f)
                                .Attr(
                                    "transA",
                                    "Whether A should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Attr(
                                    "transB",
                                    "Whether B should be transposed on the last two dimensions before doing multiplication",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(0))
                                .Output(0, "Y", "Matrix multiply results", "T1")
                                .TypeConstraint(
                                    "T",
                                    {"sparse_tensor(float)", "sparse_tensor(double)", "sparse_tensor(int64)", "sparse_tensor(int32)",
                                     "sparse_tensor(uint64)", "sparse_tensor(uint32)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)",
                                     "tensor(uint64)", "tensor(uint32)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // 1- dense tensor to output
                                  propagateElemTypeFromInputToOutput(ctx, 1, 0);
                                  // TODO: replace with ONNX one when that one is fixed
                                  sparseCompatibleMatmulShapeInference(ctx, 0, 1);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(MurmurHash3, 1,
                            OpSchema()
                                .SetDoc(R"DOC(The underlying implementation is MurmurHash3_x86_32 generating low latency 32bits hash suitable for implementing lookup tables, Bloom filters, count min sketch or feature hashing.)DOC")
                                .Input(0, "X", "An input tensor to hash.", "T1")
                                .Output(0, "Y", "32-bit hash value.", "T2")
                                .TypeConstraint("T1", {"tensor(uint32)", "tensor(int32)", "tensor(uint64)", "tensor(int64)", "tensor(float)", "tensor(double)", "tensor(string)"}, "Constrain input type to unsigned or signed 32-bit integer tensor, or string tensor. It should be utf-8 encoded if using unicode.")
                                .TypeConstraint("T2", {"tensor(uint32)", "tensor(int32)"}, "Constrain output type to unsigned and signed 32-bit integer tensor.")
                                .Attr(
                                    "seed",
                                    "Seed for the hashing algorithm, unsigned 32-bit integer, default to 0.",
                                    AttributeProto::INT,
                                    (int64_t)0LL)
                                .Attr(
                                    "positive",
                                    "If value is 1, output type is uint32_t, else int32_t. Default value is 1.",
                                    AttributeProto::INT,
                                    (int64_t)1LL)
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // type inference
                                  auto positive_attr = ctx.getAttribute("positive");
                                  bool is_positive =
                                      positive_attr ? (static_cast<int>(positive_attr->i()) == 1 ? true : false) : true /* default value if attribute not present */;
                                  auto output_data_type = ctx.getOutputType(0)->mutable_tensor_type();
                                  if (is_positive) {
                                    output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32);
                                  } else {
                                    output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32);
                                  }

                                  // Shape inference
                                  if (!hasInputShape(ctx, 0))
                                    return;

                                  auto& input_shape = getInputShape(ctx, 0);
                                  updateOutputShape(ctx, 0, input_shape);
                                }));

ONNX_MS_OPERATOR_SET_SCHEMA(GatherND, 1,
                            OpSchema()
                                .Input(0, "data", "Tensor of rank r >= 1.", "T")
                                .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
                                .Output(0, "output", "Tensor of rank q-1+r-indices[-1].", "T")
                                .TypeConstraint(
                                    "T",
                                    OpSchema::all_tensor_types(),
                                    "Constrain input and output types to any tensor type.")
                                .TypeConstraint(
                                    "Tind",
                                    {"tensor(int32)", "tensor(int64)"},
                                    "Constrain indice type to int32 or int64")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  if (!hasNInputShapes(ctx, 2)) {
                                    return;
                                  }
                                  auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
                                  auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
                                  auto data_rank = data_shape.dim_size();
                                  auto indices_rank = indices_shape.dim_size();
                                  if (data_rank < 1 || indices_rank < 1) {
                                    fail_shape_inference("both data and indices tensor need to have rank larger than zero.");
                                  }
                                  auto last_indice_dimension = indices_shape.dim(indices_rank - 1).dim_value();
                                  if (last_indice_dimension > data_rank) {
                                    fail_shape_inference("last dimension of indices must not be larger and rank of data tensor");
                                  }
                                  for (int i = 0; i < indices_rank - 1; ++i) {
                                    *ctx.getOutputType(0)
                                         ->mutable_tensor_type()
                                         ->mutable_shape()
                                         ->add_dim() = indices_shape.dim(i);
                                  }
                                  for (int i = static_cast<int>(last_indice_dimension); i < data_rank; ++i) {
                                    *ctx.getOutputType(0)
                                         ->mutable_tensor_type()
                                         ->mutable_shape()
                                         ->add_dim() = data_shape.dim(i);
                                  }
                                })
                                .SetDoc(R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q >= 1, gather
slices of `data` into an output tensor of rank q - 1 + r - indices[-1].
Example 1:
  data    = [[0,1],[2,3]]
  indices = [[0,0],[1,1]]
  output  = [0,3]
Example 2:
  data    = [[0,1],[2,3]]
  indices = [[1],[0]]
  output  = [[2,3],[0,1]]
Example 3:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[0,1],[1,0]]
  output  = [[2,3],[4,5]]
Example 4:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[[0,1]],[[1,0]]]
  output  = [[[2,3]],[[4,5]]]
)DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(WordConvEmbedding, 1,
                            OpSchema()
                                .Attr(
                                    "embedding_size",
                                    "Integer representing the embedding vector size for each word."
                                    "If not provide, use the fileter size of conv weight",
                                    AttributeProto::INT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "conv_window_size",
                                    "This operator applies convolution to word from left to right with window equal to conv_window_size and stride to 1."
                                    "Take word 'example' for example, with conv_window_size equal to 2, conv is applied to [ex],[xa], [am], [mp]..."
                                    "If not provide, use the first dimension of conv kernal shape.",
                                    AttributeProto::INT,
                                    OPTIONAL_VALUE)
                                .Attr(
                                    "char_embedding_size",
                                    "Integer representing the embedding vector size for each char."
                                    "If not provide, use the char embedding size of embedding vector.",
                                    AttributeProto::INT,
                                    OPTIONAL_VALUE)
                                .Input(0, "Sequence", "Specify batchs of sequence words to embedding", "T")
                                .Input(1, "W", "Specify weights of conv", "T1")
                                .Input(2, "B", "Specify bias of conv", "T1")
                                .Input(3, "C", "Specify embedding vector of char", "T1")
                                .Output(0, "Y", "output", "T1")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(int32)"},
                                    "Constrain to tensor(int32).")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float)"},
                                    "Constrain to tensor(float).")
                                .SetDoc(R"DOC(The WordConvEmbedding takes in a batch of sequence words and embed each word to a vector.)DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(Pad, 1,
                            OpSchema()
                                .Attr(
                                    "mode",
                                    "Three modes: `constant`(default) - pads with a given constant value, "
                                    "`reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis, "
                                    "`edge` - pads with the edge values of array",
                                    AttributeProto::STRING,
                                    std::string("constant"))
                                .Input(0, "data", "Input tensor.", "T")
                                .Input(
                                    1,
                                    "pads",
                                    "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
                                    "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
                                    "`pads` should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]. "
                                    "`pads` format (1D example) should be as follow [x1_begin, x2_begin,...,x1_end, x2_end,...], "
                                    "where xi_begin is the number of pixels added at the beginning of axis `i` and "
                                    "xi_end, the number of pixels added at the end of axis `i`.",
                                    "tensor(int64)")
                                .Input(
                                    2,
                                    "value",
                                    "(Optional) A scalar or rank 1 tensor containing a single value to be filled if the mode chosen is `constant` (by default it is 0.0).",
                                    "T",
                                    OpSchema::Optional)
                                .Output(0, "output", "Tensor after padding.", "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain input and output types to float tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  // Shape inference needs the input data shape
                                  if (!hasNInputShapes(ctx, 1)) {
                                    return;
                                  }
                                  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
                                  const auto input_rank = input_shape.dim_size();

                                  // Infer output shape if 'pads' tensor is available
                                  const auto* pads_initializer = ctx.getInputData(1);
                                  if (nullptr != pads_initializer) {
                                    const auto& pads_shape = ctx.getInputType(1)->tensor_type().shape();
                                    if ((pads_initializer->dims_size() != 1 &&
                                         pads_initializer->dims_size() != 2) ||
                                        (pads_initializer->dims_size() == 2 &&
                                         pads_shape.dim(static_cast<int>(0)).dim_value() != 1) ||
                                        pads_initializer->data_type() != ONNX_NAMESPACE::TensorProto::INT64)
                                      fail_shape_inference(
                                          "'pads' input must be a 1D (shape: [input_rank]) "
                                          "or 2D tensor (shape: [1, input_rank]) of type int64");

                                    // make a copy of the returned const vector - may have to resize
                                    // this in next step
                                    std::vector<int64_t> pads_data;
                                    if (utils::HasRawData(*pads_initializer))
                                      return;
                                    else
                                      pads_data.insert(
                                          pads_data.end(),
                                          pads_initializer->int64_data().begin(),
                                          pads_initializer->int64_data().end());

                                    // fill with zeros if needed to reach appropriate size
                                    if (pads_data.size() != 2 * static_cast<size_t>(input_rank))
                                      pads_data.resize(size_t{2} * input_rank, 0);

                                    const auto& output_shape =
                                        ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
                                    for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
                                      const auto& input_dim = input_shape.dim(static_cast<int>(i));
                                      auto* output_dim = output_shape->add_dim();
                                      if (utils::HasDimValue(input_dim)) {
                                        output_dim->set_dim_value(
                                            input_dim.dim_value() + pads_data[i] + pads_data[i + input_rank]);
                                      } else if (pads_data[i] + pads_data[i + input_rank] == 0) {
                                        *output_dim = input_dim;
                                      }
                                    }
                                  } else {
                                    // Infer output shapes' rank in any case
                                    auto* output_shape_0 = getOutputShape(ctx, 0);
                                    for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
                                      output_shape_0->add_dim();
                                    }
                                  }
                                  return;
                                })
                                .SetDoc(R"DOC(
            Given `data` tensor, pads, mode, and value.
            Example:
            Insert 0 pads to the beginning of the second dimension.
            data = [
                    [1.0, 1.2],
                    [2.3, 3.4],
                    [4.5, 5.7],
                    ]
            pads = [0, 2, 0, 0]
            output = [
                    [
                    [0.0, 0.0, 1.0, 1.2],
                    [0.0, 0.0, 2.3, 3.4],
                    [0.0, 0.0, 4.5, 5.7],
                    ],
                    ]
            )DOC"));

ONNX_MS_OPERATOR_SET_SCHEMA(Unique, 1,
                            OpSchema()
                                .Input(0, "x", "A 1-D input tensor that is to be processed.", "T")
                                .Output(0, "y",
                                        "A 1-D tensor of the same type as 'x' "
                                        "containing all the unique values in 'x' sorted "
                                        "in the same order that they occur in the input 'x'",
                                        "T")
                                .Output(1, "idx",
                                        "A 1-D INT64 tensor of the same size as 'x' "
                                        "containing the indices for each value in 'x' "
                                        "in the output 'uniques'",
                                        "tensor(int64)")
                                .Output(2, "counts",
                                        "A 1-D INT64 tensor containing the "
                                        "the count of each element "
                                        "of 'uniques' in the input 'x'",
                                        "tensor(int64)")
                                .TypeConstraint("T", OpSchema::all_tensor_types(), "Input can be of any tensor type.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  // Type inference
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  ONNX_NAMESPACE::updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::INT64);
                                  ONNX_NAMESPACE::updateOutputElemType(ctx, 2, ONNX_NAMESPACE::TensorProto::INT64);

                                  // Shape inference

                                  // shape of output 'uniques' and 'counts'
                                  // depends on actual input data, but the rank is always 1
                                  ctx.getOutputType(0)
                                      ->mutable_tensor_type()
                                      ->mutable_shape()
                                      ->add_dim();

                                  ctx.getOutputType(2)
                                      ->mutable_tensor_type()
                                      ->mutable_shape()
                                      ->add_dim();

                                  // if the input shape doesn't exist, further shape inference is not possible
                                  if (!hasNInputShapes(ctx, 1)) {
                                    return;
                                  }

                                  // 'idx' output has same shape as input
                                  ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 1);

                                  return;
                                })
                                .SetDoc(R"DOC(
              Finds all the unique values (deduped list) present in the given input tensor.
              This operator returns 3 outputs.
              The first output tensor 'uniques' contains all of the unique elements of the input,
              sorted in the same order that they occur in the input.
              The second output tensor 'idx' is the same size as the input and it contains the index
              of each value of the input in 'uniques'.
              The third output tensor 'counts' contains the count of each element of 'uniques' in the input.
              Example:
                input_x = [2, 1, 1, 3, 4, 3]
                output_uniques = [2, 1, 3, 4]
                output_idx = [0, 1, 1, 2, 3, 2]
                output_counts = [1, 2, 2, 1]
              )DOC"));

// see:https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
ONNX_MS_OPERATOR_SET_SCHEMA(CDist, 1,
                            OpSchema()
                                .Attr("metric",
                                      "The distance metric to use. If a string, the distance function can be \"braycurtis\", \"canberra\", "
                                      "\"chebyshev\", \"cityblock\", \"correlation\", \"cosine\", \"dice\", \"euclidean\", \"hamming\", \"jaccard\", "
                                      "\"jensenshannon\", \"kulsinski\", \"mahalanobis\", \"matching\", \"minkowski\", \"rogerstanimoto\", \"russellrao\", "
                                      "\"seuclidean\", \"sokalmichener\", \"sokalsneath\", \"sqeuclidean\", \"wminkowski\", \"yule\".",
                                      AttributeProto::STRING, std::string("sqeuclidean"))
                                .Input(0, "A", "2D matrix with shape (M,N)", "T")
                                .Input(1, "B", "2D matrix with shape (K,N)", "T")
                                .Output(0, "C",
                                        "A 2D Matrix that represents the distance between each pair of the two collections of inputs.",
                                        "T")
                                .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Constrains input to only numeric types."));

ONNX_MS_OPERATOR_SET_SCHEMA(CropAndResize, 1,
                            OpSchema()
                                .Attr(
                                    "mode",
                                    "The pooling method. Two modes are supported: 'bilinear' and 'nearest'. "
                                    "Default is 'bilinear'.",
                                    AttributeProto::STRING,
                                    std::string("bilinear"))
                                .Attr(
                                    "extrapolation_value",
                                    "Value used for extrapolation, when applicable. "
                                    "Default is 0.0f. ",
                                    AttributeProto::FLOAT,
                                    0.f)
                                .Input(
                                    0,
                                    "X",
                                    "Input data tensor from the previous operator; "
                                    "4-D feature map of shape (N, C, H, W), "
                                    "where N is the batch size, C is the number of channels, "
                                    "and H and W are the height and the width of the data.",
                                    "T1")
                                .Input(
                                    1,
                                    "rois",
                                    "RoIs (Regions of Interest) to pool over; rois is "
                                    "2-D input of shape (num_rois, 4) given as "
                                    "[[y1, x1, y2, x2], ...]. "
                                    "The RoIs' coordinates are normalized in the coordinate system of the input image. "
                                    "Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.",
                                    "T1")
                                .Input(
                                    2,
                                    "batch_indices",
                                    "1-D tensor of shape (num_rois,) with each element denoting "
                                    "the index of the corresponding image in the batch.",
                                    "T2")
                                .Input(
                                    3,
                                    "crop_size",
                                    "1-D tensor of 2 elements: [crop_height, crop_width]. "
                                    "All cropped image patches are resized to this size. Both crop_height and crop_width need to be positive.",
                                    "T2")
                                .Output(
                                    0,
                                    "Y",
                                    "RoI pooled output, 4-D tensor of shape "
                                    "(num_rois, C, crop_height, crop_width). The r-th batch element Y[r-1] "
                                    "is a pooled feature map corresponding to the r-th RoI X[r-1].",
                                    "T1")
                                .TypeConstraint(
                                    "T1",
                                    {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                    "Constrain types to float tensors.")
                                .TypeConstraint(
                                    "T2",
                                    {"tensor(int32)"},
                                    "Constrain types to int tensors.")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  if (!hasNInputShapes(ctx, 4)) {
                                    return;
                                  }
                                  propagateElemTypeFromInputToOutput(ctx, 0, 0);

                                  auto& input_shape = getInputShape(ctx, 0);
                                  auto& rois_shape = getInputShape(ctx, 1);
                                  auto& batch_index_shape = getInputShape(ctx, 2);
                                  auto& crop_size_shape = getInputShape(ctx, 3);

                                  if (input_shape.dim_size() != 4) {
                                    fail_shape_inference("first input tensor has wrong dimension");
                                  }
                                  if (rois_shape.dim_size() != 2) {
                                    fail_shape_inference("rois input tensor has wrong dimension");
                                  }
                                  if (batch_index_shape.dim_size() != 1) {
                                    fail_shape_inference("batch_indices shape input tensor has wrong dimension");
                                  }
                                  if (crop_size_shape.dim_size() != 1) {
                                    fail_shape_inference("crop_size shape input tensor has wrong dimension");
                                  }
                                })
                                .SetDoc(R"DOC(
        Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling
        (possibly with aspect ratio change) to a common output size specified by crop_height and crop_width.
        Returns a tensor with crops from the input image at positions defined at the bounding box locations in boxes.
        The cropped boxes are all resized (with bilinear or nearest neighbor interpolation) to
        a fixed size = [crop_height, crop_width]. The result is a 4-D tensor [num_boxes, crop_height, crop_width, depth].
        The resizing is corner aligned.)DOC"));

void RegisterContribSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(AttnLSTM, RegisterAttnLSTMContribOpSchema);
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Range, RegisterRangeOpSchema);

  ONNX_CONTRIB_OPERATOR_SCHEMA(LayerNormalization)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("LayerNormalization")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .Attr("epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttributeProto::FLOAT, 1e-5f)
      .Attr("stash_type",
            "type used for stash mean/inv_std_var",
            AttributeProto::INT, static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input data tensor from the previous layer.", "T")
      .Input(1, "Scale", "Scale tensor.", "V")
      .Input(2, "B", "Bias tensor.", "V", OpSchema::Optional)
      .Output(0, "Y", "Output data tensor.", "V")
      .Output(1, "Mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
      .Output(2, "InvStdDev", "Saved inverse standard deviation used during training to speed up gradient computation.", "U", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input X type to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)", "tensor(double)"},
          "Type of Mean and InvStdDev tensors.")
      .TypeConstraint(
          "V",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain output Y, scale and bias type to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        auto type = ctx.getAttribute("stash_type")->i();
        if (ctx.getNumOutputs() > 1) {
          auto output_type = ctx.getOutputType(1);
          output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(type));
        }
        if (ctx.getNumOutputs() > 2) {
          auto output_type = ctx.getOutputType(2);
          output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(type));
        }
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        propagateShapeFromInputToOutput(ctx, 0, 0);
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        int64_t axis = -1;
        auto axis_proto = ctx.getAttribute("axis");
        if (axis_proto) {
          axis = axis_proto->i();
        }
        if (axis < 0) {
          axis += input_ndim;
        }

        if (ctx.getNumOutputs() > 1) {
          auto saved_mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          saved_mean_shape->CopyFrom(input_shape);
          for (int d = static_cast<int>(axis); d < input_ndim; ++d)
            saved_mean_shape->mutable_dim(d)->set_dim_value(1);
        }

        if (ctx.getNumOutputs() > 2) {
          auto saved_inv_std_dev_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
          saved_inv_std_dev_shape->CopyFrom(input_shape);
          for (int d = static_cast<int>(axis); d < input_ndim; ++d)
            saved_inv_std_dev_shape->mutable_dim(d)->set_dim_value(1);
        }
      })
      .SetContextDependentFunctionBodyBuilder(
          [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
            // LayerNormalization <axis, epsilon, stash_type> (X, Scale, B) => (Y, Mean?, InvStdDev?)

            auto* tp = ctx.getInputType(1);
            if ((tp == nullptr) || (!tp->has_tensor_type()))
              return false;
            int64_t V = tp->tensor_type().elem_type();

            auto type_attr = ctx.getAttribute("stash_type");
            int64_t U = (type_attr != nullptr) ? type_attr->i() : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) && (U != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
              return false;  // Error

            auto* axis_attr = ctx.getAttribute("axis");
            int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
            auto* epsilon_attr = ctx.getAttribute("epsilon");
            float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

            auto mktensor = [](int64_t val) -> ONNX_NAMESPACE::TensorProto {
              auto tp = ONNX_NAMESPACE::ToTensor(std::vector<int64_t>{val});
              tp.add_dims(1);
              return tp;
            };

            // The treatment of "axis" is different in "LayerNormalization" and in Reduction operations.
            // This complicates the function definition, requiring reshaping inputs/outputs.
            // Input X shape: [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]
            // This is treated as a 2D shape [d[0] * ... * d[axis-1], d[axis] * ... * d[rank-1]]
            // Normalization is applied to the second dimension.
            // Output Y has same shape as X
            // Outputs Mean and InvStdDev have shape: [d[0], ..., d[axis-1], 1, ..., 1]
            FunctionBuilder builder(functionProto);
            builder
                .AddOpset("", 13)
                .Const("Epsilon", ToTensor(epsilon, (TensorProto_DataType)U))
                .Add("XShape = Shape (X)")                                                    // shape of input tensor: 1D tensor
                .Add("Rank = Size (XShape)")                                                  // rank of input tensor: scalar
                .Add("Zero1D = Constant()", "value", mktensor(0))                             // [0] : 1D tensor
                .Add("Axis1D = Constant()", "value", mktensor(axis))                          // [axis] : 1D tensor
                .Add("PrefixShape = Slice (XShape, Zero1D, Axis1D)")                          // [d[0], ..., d[axis-1]]
                .Add(axis > 0                                                                 // number of axes that are reduced =
                         ? "NumReducedAxes = Sub (Rank, Axis1D)"                              // [rank - axis]: 1D tensor
                         : "NumReducedAxes = Neg (Axis1D)")                                   // [-axis] : 1D tensor
                .Add("SuffixShape = ConstantOfShape (NumReducedAxes)", "value", mktensor(1))  // [1, ..., 1] for reduced axes
                .Add("ReducedShape = Concat <axis = 0> (PrefixShape, SuffixShape)")           // [d[0], ..., d[axis-1], 1, ..., 1]
                .Add("X2D = Flatten (X)", "axis", axis)
                .Add("XU = Cast (X2D)", "to", U)
                .Add("Mean2D = ReduceMean <axes = [1]> (XU)")
                .Add("Square = Mul (XU, XU)")
                .Add("MeanOfSquare = ReduceMean <axes = [1]> (Square)")
                .Add("SquareOfMean = Mul (Mean2D, Mean2D)")
                .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
                .Add("VarPlusEpsilon = Add (Var, Epsilon)")
                .Add("StdDev = Sqrt (VarPlusEpsilon)")
                .Add("Deviation = Sub (XU, Mean2D)")
                .Add("Normalized = Div (Deviation, StdDev)")
                .Add("NormalizedV = Cast (Normalized)", "to", V)
                .Add("Scale2D = Flatten <axis = 0> (Scale)")
                .Add("Scaled = Mul (NormalizedV, Scale2D)");
            if (ctx.hasInput(2)) {
              builder.Add("B2D = Flatten <axis=0> (B)");
              builder.Add("Biased = Add (Scaled, B2D)");
            } else {
              builder.Add("Biased = Identity (Scaled)");
            }
            builder.Add("Y = Reshape (Biased, XShape)");
            builder.Add("InvStdDev2D = Reciprocal (StdDev)");
            if (ctx.hasOutput(1))
              builder.Add("Mean = Reshape (Mean2D, ReducedShape)");
            if (ctx.hasOutput(2))
              builder.Add("InvStdDev = Reshape (InvStdDev2D, ReducedShape)");

            schema.BuildFunction(functionProto);
            return true;
          });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SimplifiedLayerNormalization)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SimplifiedLayerNormalization")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .Attr("epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttributeProto::FLOAT, 1e-5f)
      .Attr("stash_type",
            "type used for stash mean/inv_std_var",
            AttributeProto::INT, static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input data tensor from the previous layer.", "T")
      .Input(1, "scale", "Scale tensor.", "V")
      .Output(0, "Y", "Output data tensor.", "V")
      .Output(1, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input X type to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)", "tensor(double)"},
          "Constrain mean and inv_std_var to be float tensors.")
      .TypeConstraint(
          "V",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain output Y and scale type to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 1, 0);
        auto type = ctx.getAttribute("stash_type")->i();
        if (ctx.getNumOutputs() > 1) {
          auto output_type = ctx.getOutputType(1);
          output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(type));
        }
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        propagateShapeFromInputToOutput(ctx, 0, 0);
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        int64_t axis = -1;
        auto axis_proto = ctx.getAttribute("axis");
        if (axis_proto) {
          axis = axis_proto->i();
        }
        if (axis < 0) {
          axis += input_ndim;
        }

        if (ctx.getNumOutputs() > 1) {
          auto saved_inv_std_var_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          saved_inv_std_var_shape->CopyFrom(input_shape);
          saved_inv_std_var_shape->mutable_dim(static_cast<int>(axis))->set_dim_value(1);
        }
      });

  static const char* EfficientNMS_TRT_ver1_doc =
      R"DOC(Efficient NMS TensorRT Plugin.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(EfficientNMS_TRT)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetDoc(EfficientNMS_TRT_ver1_doc)
      .Input(0, "boxes", "The boxes input tensor.", "T")
      .Input(1, "scores", "The scores input tensor.", "T")
      .Input(2, "anchors", "The anchors input tensor.", "T", OpSchema::Optional)
      .Output(0, "num_detections", "The num_detections output tensor.", "tensor(int32)")
      .Output(1, "detection_boxes", "The detection_boxes output tensor.", "T")
      .Output(2, "detection_scores", "The detection_scores output tensor.", "T")
      .Output(3, "detection_classes", "The detection_classes output tensor.", "tensor(int32)")
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
      .Attr("background_class", "Background class ID.", AttributeProto::INT)
      .Attr("box_coding", "Encoding type for the boxes or anchors inputs.", AttributeProto::INT)
      .Attr("iou_threshold", "Box IOU threshold value.", AttributeProto::FLOAT)
      .Attr("max_output_boxes", "Max detections to output.", AttributeProto::INT)
      .Attr("plugin_version", "Version number of the TRT plugin.", AttributeProto::STRING)
      .Attr("score_activation", "Activation function to apply to the scores input.", AttributeProto::INT)
      .Attr("score_threshold", "Score threshold value.", AttributeProto::FLOAT)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        using namespace ONNX_NAMESPACE;
        ONNX_NAMESPACE::updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::INT32);
        propagateElemTypeFromInputToOutput(ctx, 0, 1);
        propagateElemTypeFromInputToOutput(ctx, 0, 2);
        ONNX_NAMESPACE::updateOutputElemType(ctx, 3, ONNX_NAMESPACE::TensorProto::INT32);

        // Shape Inference
        if (!hasInputShape(ctx, 0)) {
          return;
        }
        int64_t max_output_boxes = 1;
        auto max_output_boxes_proto = ctx.getAttribute("max_output_boxes");
        if (max_output_boxes_proto) {
          max_output_boxes = max_output_boxes_proto->i();
        }
        if (max_output_boxes < 1) {
          fail_shape_inference("Attribute 'max_output_boxes' must be >= 1.")
        }

        Dim batch_size;
        unifyInputDim(ctx, 0, 0, batch_size);

        ONNX_NAMESPACE::TensorShapeProto num_detections_shape;
        *num_detections_shape.add_dim() = batch_size;
        num_detections_shape.add_dim()->set_dim_value(1);
        updateOutputShape(ctx, 0, num_detections_shape);

        ONNX_NAMESPACE::TensorShapeProto detection_boxes_shape;
        *detection_boxes_shape.add_dim() = batch_size;
        detection_boxes_shape.add_dim()->set_dim_value(max_output_boxes);
        detection_boxes_shape.add_dim()->set_dim_value(4);
        updateOutputShape(ctx, 1, detection_boxes_shape);

        ONNX_NAMESPACE::TensorShapeProto detection_scores_shape;
        *detection_scores_shape.add_dim() = batch_size;
        detection_scores_shape.add_dim()->set_dim_value(max_output_boxes);
        updateOutputShape(ctx, 2, detection_scores_shape);

        ONNX_NAMESPACE::TensorShapeProto detection_classes_shape;
        *detection_classes_shape.add_dim() = batch_size;
        detection_classes_shape.add_dim()->set_dim_value(max_output_boxes);
        updateOutputShape(ctx, 3, detection_classes_shape);
      });

  static const char* MultilevelCropAndResize_TRT_ver1_doc =
      R"DOC(Multilevel Crop and Resize TensorRT Plugin.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(MultilevelCropAndResize_TRT)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetDoc(MultilevelCropAndResize_TRT_ver1_doc)
      .Input(0, "boxes", "The boxes input tensor.", "T")
      .Input(1, "feature_map_0", "The first feature map input tensor.", "T")
      .Input(2, "feature_map_1", "The second feature map input tensor.", "T")
      .Input(3, "feature_map_2", "The third feature map input tensor.", "T")
      .Input(4, "feature_map_3", "The fourth feature map input tensor.", "T")
      .Output(0, "patches", "The cropped patches output tensor.", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
      .Attr("image_size", "Image size.", AttributeProto::INTS)
      .Attr("pooled_size", "Pooled size.", AttributeProto::INT)
      .Attr("plugin_version", "Version number of the TRT plugin.", AttributeProto::STRING)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape Inference
        if (!hasInputShape(ctx, 0)) {
          return;
        }
        int64_t pooled_size = 1;
        auto pooled_size_proto = ctx.getAttribute("pooled_size");
        if (pooled_size_proto) {
          pooled_size = pooled_size_proto->i();
        }
        if (pooled_size < 1) {
          fail_shape_inference("Attribute 'pooled_size' must be >= 1.")
        }

        Dim batch_size, number_boxes, channels;
        unifyInputDim(ctx, 0, 0, batch_size);
        unifyInputDim(ctx, 0, 1, number_boxes);
        unifyInputDim(ctx, 1, 1, channels);

        ONNX_NAMESPACE::TensorShapeProto output_shape;
        *output_shape.add_dim() = batch_size;
        *output_shape.add_dim() = number_boxes;
        *output_shape.add_dim() = channels;
        output_shape.add_dim()->set_dim_value(pooled_size);
        output_shape.add_dim()->set_dim_value(pooled_size);
        updateOutputShape(ctx, 0, output_shape);
      });

  static const char* PyramidROIAlign_TRT_ver1_doc =
      R"DOC(Pyramid ROI Align TensorRT Plugin.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(PyramidROIAlign_TRT)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .SetDoc(PyramidROIAlign_TRT_ver1_doc)
      .Input(0, "boxes", "The boxes input tensor.", "T")
      .Input(1, "feature_map_0", "The first feature map input tensor.", "T")
      .Input(2, "feature_map_1", "The second feature map input tensor.", "T")
      .Input(3, "feature_map_2", "The third feature map input tensor.", "T")
      .Input(4, "feature_map_3", "The fourth feature map input tensor.", "T")
      .Output(0, "patches", "The cropped patches output tensor.", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
      .Attr("pooled_size", "Pooled size.", AttributeProto::INT)
      .Attr("plugin_version", "Version number of the TRT plugin.", AttributeProto::STRING)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape Inference
        if (!hasInputShape(ctx, 0)) {
          return;
        }
        int64_t pooled_size = 1;
        auto pooled_size_proto = ctx.getAttribute("pooled_size");
        if (pooled_size_proto) {
          pooled_size = pooled_size_proto->i();
        }
        if (pooled_size < 1) {
          fail_shape_inference("Attribute 'pooled_size' must be >= 1.")
        }

        Dim batch_size, number_boxes, channels;
        unifyInputDim(ctx, 0, 0, batch_size);
        unifyInputDim(ctx, 0, 1, number_boxes);
        unifyInputDim(ctx, 1, 1, channels);

        ONNX_NAMESPACE::TensorShapeProto output_shape;
        *output_shape.add_dim() = batch_size;
        *output_shape.add_dim() = number_boxes;
        *output_shape.add_dim() = channels;
        output_shape.add_dim()->set_dim_value(pooled_size);
        output_shape.add_dim()->set_dim_value(pooled_size);
        updateOutputShape(ctx, 0, output_shape);
      });

#ifndef _OPSCHEMA_LIB_
  // Register the NCHWc schemas if supported by the platform.
  if (MlasNchwcGetBlockSize() > 1) {
    RegisterNchwcSchemas();
  }
#endif

#ifdef BUILD_MS_EXPERIMENTAL_OPS
  onnxruntime::signal::RegisterSignalSchemas();
#endif
}

}  // namespace contrib
}  // namespace onnxruntime
