// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/attn_lstm_schema_defs.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/nchwc_schema_defs.h"
#include "core/graph/contrib_ops/range_schema_defs.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/op.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"
#include "core/mlas/inc/mlas.h"

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
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void ValidateTypeAndShapeForScaleAndZP(ONNX_NAMESPACE::InferenceContext& ctx, int index, ::google::protobuf::int32 expectedType, bool isScalar, int expectedTensorSize = 0) {
  if (ctx.getNumInputs() > static_cast<size_t>(index)) {
    auto data_type = ctx.getInputType(index);
    if (nullptr == data_type) {
      fail_type_inference("Input data type does not match the expected data type");
    }
    if (data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
        data_type->tensor_type().elem_type() != expectedType) {
      fail_type_inference(
          "Input data type does not match the expected data type. Current data type is ", data_type->tensor_type().elem_type());
    }
  }

  if (hasInputShape(ctx, index)) {
    ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(index)->tensor_type().shape();
    if (isScalar) {
      if (shape.dim_size() != 0) {
        fail_type_inference("Scale and Zero-point must be a scalar");
      }
    } else {
      if (shape.dim_size() != 1) {
        fail_type_inference("Scale and Zero-point must be of rank 1");
      }

      if (shape.dim((int)0).has_dim_value() && shape.dim((int)0).dim_value() != expectedTensorSize) {
        fail_type_inference(
            "Scale and Zero-point must be of rank 1 and the number of elements should be equal to the number of rows of the corresponding input.");
      }
    }
  }
}

std::function<void(OpSchema&)> QLinearMathDocGenerator(const char* name, const char* additionalDocumentation) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Performs element-wise binary {name} on 8 bit data types (with Numpy-style broadcasting support).

{additionalDocumentation}
)DOC";
    ONNX_NAMESPACE::ReplaceAll(doc, "{name}", name);
    ONNX_NAMESPACE::ReplaceAll(doc, "{additionalDocumentation}", additionalDocumentation);
    schema.SetDoc(doc);
    schema.Input(0, "A", "First operand.", "T");
    schema.Input(
        1,
        "A_scale",
        "Input A's scale. It's a scalar, which means a per-tensor/layer quantization.",
        "tensor(float)");
    schema.Input(
        2,
        "A_zero_point",
        "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
        "T",
        OpSchema::Optional);
    schema.Input(3, "B", "Second operand.", "T");
    schema.Input(
        4,
        "B_scale",
        "Input B's scale. It's a scalar, which means a per-tensor/layer quantization.",
        "tensor(float)");
    schema.Input(
        5,
        "B_zero_point",
        "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
        "T",
        OpSchema::Optional);
    schema.Input(
        6,
        "C_scale",
        "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
        "tensor(float)");
    schema.Input(
        7,
        "C_zero_point",
        "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
        "T",
        OpSchema::Optional);
    schema.Output(0, "C", "Result, has same element type as two inputs", "T");
    schema.TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input and output types to 8 bit signed and unsigned tensors.");
    schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      auto a_type = ctx.getInputType(0);
      auto b_type = ctx.getInputType(3);

      if (nullptr == a_type || nullptr == b_type ||
          a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
          b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
      }

      // validate scale and zero points
      ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 2, a_type->tensor_type().elem_type(), true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 4, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 5, b_type->tensor_type().elem_type(), true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 6, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 7, a_type->tensor_type().elem_type(), true);

      if (hasInputShape(ctx, 0) && hasInputShape(ctx, 3))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(3)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

const char* contrib_ops_pads_doc =
    "Padding for the beginning and ending along each spatial axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
const char* contrib_ops_auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding.";

void RegisterBertSchemas() {
  static const char* Attention_ver1_doc = R"DOC(
Multi-Head Self Attention that can be either unidirectional (like GPT-2) or bidirectional (like BERT).
The mask_index input is optional. Besides raw attention mask with shape (batch_size, past_sequence_length + sequence_length),
we also support other two formats: When input has right-side padding, mask_index is one dimension with shape (batch_size),
where value of each element is the end position, or valid length of actual sequence excluding padding. When input has
left-side padding, mask_index has shape (2 * batch_size), where the values are the exclusive end positions followed by
the inclusive start positions. When unidirectional is 1, and each token only attend to previous tokens. For GPT-2, both past
and present state are optional. Present state could appear in output even when past state is not in input.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Attention)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc(Attention_ver1_doc)
      .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
      .Attr("unidirectional",
            "Whether every token can only attend to previous tokens. Default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * head_size", "T")
      .Input(1, "weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "T")
      .Input(2, "bias", "1D input tensor with shape (3 * hidden_size)", "T")
      .Input(3, "mask_index", "Attention mask with shape (batch_size, past_sequence_length + sequence_length), or index with shape (batch_size) or (2 * batch_size).", "M", OpSchema::Optional)
      .Input(4, "past", "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).", "T", OpSchema::Optional)
      .Output(0, "output", "3D output tensor with shape (batch_size, append_length, hidden_size)", "T")
      .Output(1, "present", "present state for key and value with shape (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)", "T", OpSchema::Optional)
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
      .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask index to integer types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getNumOutputs() > 1) {
          propagateElemTypeFromInputToOutput(ctx, 0, 1);
        }

        if (hasInputShape(ctx, 0)) {
          propagateShapeFromInputToOutput(ctx, 0, 0);

          if (ctx.getNumOutputs() > 1) {
            auto& input_shape = getInputShape(ctx, 0);
            auto& input_dims = input_shape.dim();
            if (input_dims.size() != 3) {
              fail_shape_inference("Inputs 0 shall be 3 dimensions");
            }

            if (hasInputShape(ctx, 4)) {
              auto& past_shape = getInputShape(ctx, 4);
              auto& past_dims = past_shape.dim();
              if (past_dims.size() != 5) {
                fail_shape_inference("Inputs 4 shall be 5 dimensions");
              }

              if (past_dims[3].has_dim_value() && input_dims[1].has_dim_value()) {
                auto all_sequence_length = past_shape.dim(3).dim_value() + input_shape.dim(1).dim_value();

                ONNX_NAMESPACE::TensorShapeProto present_shape;
                for (auto& dim : past_dims) {
                  *present_shape.add_dim() = dim;
                }
                present_shape.mutable_dim(3)->set_dim_value(all_sequence_length);

                updateOutputShape(ctx, 1, present_shape);
              }
            }
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(QAttention)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Quantization of Multi-Head Self Attention.")
      .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
      .Attr("unidirectional",
            "Whether every token can only attend to previous tokens. Default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(
          0,
          "input",
          "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * head_size",
          "T1")
      .Input(
          1,
          "weight",
          "2D input tensor with shape (hidden_size, 3 * hidden_size)",
          "T2")
      .Input(
          2,
          "bias",
          "1D input tensor with shape (3 * hidden_size)",
          "T3")
      .Input(
          3,
          "input_scale",
          "scale of quantized input tensor. It's a scalar, which means a per-tensor/layer quantization.",
          "T3")
      .Input(
          4,
          "weight_scale",
          "scale of weight scale. It's a scalar, which means a per-tensor/layer quantization.",
          "T3")
      .Input(
          5,
          "mask_index",
          "Attention mask index with shape (batch_size)",
          "T4",
          OpSchema::Optional)
      .Input(
          6,
          "input_zero_point",
          "zero point of quantized input tensor.It's a scalar, which means a per-tensor/layer quantization.",
          "T1",
          OpSchema::Optional)
      .Input(
          7,
          "weight_zero_point",
          "zero point of quantized weight tensor. It's a scalar, which means a per-tensor/layer quantization.",
          "T2",
          OpSchema::Optional)
      .Input(
          8,
          "past",
          "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).",
          "T3",
          OpSchema::Optional)
      .Output(
          0,
          "output",
          "3D output tensor with shape (batch_size, sequence_length, hidden_size)",
          "T3")
      .Output(
          1,
          "present",
          "present state for key and value with shape (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)",
          "T3",
          OpSchema::Optional)
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input and output types to int8 tensors.")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input and output types to int8 tensors.")
      .TypeConstraint("T3", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
      .TypeConstraint("T4", {"tensor(int32)"}, "Constrain mask index to integer types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 2, 0);

        // Shape inference
        // if the input shape doesn't exist, further shape inference is not possible
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 0);

        return;
      });

  static const char* EmbedLayerNormalization_ver1_doc = R"DOC(
EmbedLayerNormalization is the fusion of embedding layer in BERT model, with optional mask processing.
The embedding layer takes input_ids (word IDs) and segment_ids (sentence IDs) to look up word_embedding, position_embedding,
and segment_emedding; the embeddings are added then applied layer normalization using gamma and beta tensors.
The last input mask is optional. If mask is provided, mask index (that is position of first 0 in mask, or number of words)
will be calculated.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(EmbedLayerNormalization)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc(EmbedLayerNormalization_ver1_doc)
      .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultEmbedLayerNormEpsilon)
      .Input(0, "input_ids", "2D words IDs with shape (batch_size, sequence_length)", "T1")
      .Input(1, "segment_ids", "2D segment IDs with shape (batch_size, sequence_length)", "T1", OpSchema::Optional)
      .Input(2, "word_embedding", "2D with shape (,hidden_size)", "T")
      .Input(3, "position_embedding", "2D with shape (, hidden_size)", "T")
      .Input(4, "segment_embedding", "2D with shape (, hidden_size)", "T", OpSchema::Optional)
      .Input(5, "gamma", "1D gamma tensor for layer normalization with shape (hidden_size)", "T")
      .Input(6, "beta", "1D beta tensor for layer normalization  with shape (hidden_size)", "T")
      .Input(7, "mask", "2D attention mask with shape (batch_size, sequence_length)", "T1", OpSchema::Optional)
      .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
      .Output(1, "mask_index", "1D mask_index tensor with shape (batch_size)", "T1")
      .TypeConstraint("T1", {"tensor(int32)"}, "Constrain input and output integer tensors types")
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output float tensors types.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 2, 0);
        propagateElemTypeFromInputToOutput(ctx, 0, 1);
        if (!hasInputShape(ctx, 0))
          return;

        auto& input_ids_shape = getInputShape(ctx, 0);
        auto& input_ids_dims = input_ids_shape.dim();

        // Note that both batch size and sequence length could be symbolic.
        // So we only check dimension size here.
        if (input_ids_dims.size() != 2) {
          fail_shape_inference("Inputs 0 shall be 2 dimensions");
        }

        // get hidden_size from the last dimension of embedding
        auto& word_embedding_shape = getInputShape(ctx, 3);
        auto& word_embedding_dims = word_embedding_shape.dim();
        if (word_embedding_dims.size() != 2 ||
            !word_embedding_dims[1].has_dim_value() ||
            word_embedding_shape.dim(1).dim_value() <= 0) {
          fail_shape_inference("word_embedding should have 2 dimensions and dimension size is known.");
        }
        int64_t hidden_size = word_embedding_shape.dim(1).dim_value();

        // input shape is (batch_size, sequence_length), output shape is (batch_size, sequence_length, hidden_size)
        ONNX_NAMESPACE::TensorShapeProto output_shape;
        for (auto& dim : input_ids_dims) {
          *output_shape.add_dim() = dim;
        }
        output_shape.add_dim();
        output_shape.mutable_dim(2)->set_dim_value(hidden_size);

        updateOutputShape(ctx, 0, output_shape);

        // mask_index shape is (batch_size)
        ONNX_NAMESPACE::TensorShapeProto mask_index_shape;
        *mask_index_shape.add_dim() = input_ids_dims[0];
        updateOutputShape(ctx, 1, mask_index_shape);
      });

  static const char* FastGelu_ver1_doc = R"DOC(
GELU (Gaussian Error Linear Unit) approximation: Y=0.5*X*(1+tanh(0.797885*X+0.035677*X*X*X)) with an optional input of bias that will be added to X before GELU.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(FastGelu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc(FastGelu_ver1_doc)
      .Input(0, "X", "input tensor", "T")
      .Input(1, "bias", "bias tensor", "T", OpSchema::Optional)
      .Output(0, "Y", "output tensor", "T")
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float or half tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(SkipLayerNormalization)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Skip and Layer Normalization Fusion")
      .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon)
      .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
      .Input(1, "skip", "3D skip tensor with shape (batch_size, sequence_length, hidden_size)", "T")
      .Input(2, "gamma", "1D input tensor with shape (hidden_size)", "T")
      .Input(3, "beta", "1D skip tensor with shape (hidden_size", "T")
      .Input(4, "bias", "1D bias tensor with shape (hidden_size", "T", OpSchema::Optional)
      .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
      .Output(1, "mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
      .Output(2, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
      .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
      .TypeConstraint("U", {"tensor(float)"}, "Constrain mean and inv_std_var to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
}

void RegisterContribSchemas() {
  // Register removed experimental ops for backward compatibility.
  // Experimental operators do not have version history. However, RS5 takes bunch of experimental operators
  // as production ops. In order to maintain backward compatibility when the experimental ops are removed from ONNX
  // they need to be added in onnxruntime as contrib ops.
  // ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old version history maintenance
  static const char* Affine_ver1_doc = R"DOC(
Affine takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the affine function, y = alpha * x + beta,
is applied to the tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Affine)
      .SinceVersion(1)
      .SetDoc(Affine_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* ParametricSoftplus_ver1_doc = R"DOC(
ParametricSoftplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = alpha * ln(exp(beta * x) + 1), is applied to
the tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ParametricSoftplus)
      .SinceVersion(1)
      .SetDoc(ParametricSoftplus_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D input tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* ImageScaler_ver1_doc =
      R"DOC(Scale and bias the input image. Bias values are stored in
the same ordering as the image pixel format.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ImageScaler)
      .SinceVersion(1)
      .SetDoc(ImageScaler_ver1_doc)
      .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL_VALUE)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* Crop_ver1_doc =
      R"DOC(Crop and image to the specified spatial dimensions. If scale is given,
then optionally start the crop offset by the left/top border amounts.
If scale is not provided, crop the borders as provided.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Crop)
      .SinceVersion(1)
      .SetDoc(Crop_ver1_doc)
      .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL_VALUE)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.");

  static const char* ThresholdedRelu_ver1_doc = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise. )DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(ThresholdedRelu)
      .SinceVersion(1)
      .SetDoc(ThresholdedRelu_ver1_doc)
      .Attr("alpha", "Threshold value", AttributeProto::FLOAT, 1.0f)
      .Input(0, "X", "Input tensor", "T")
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* DynamicSlice_ver1_doc = R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` inputs to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicSlice)
      .SinceVersion(1)
      .SetDoc(DynamicSlice_ver1_doc)
      .Input(0, "data", "Tensor of data to extract slices from.", "T")
      .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
      .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
      .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
      .Output(0, "output", "Sliced data tensor.", "T")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GivenTensorFill)
      .SinceVersion(1)
      .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
      .Output(0, "X", "The filled tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .Attr("values", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
      .Attr("shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE)
      .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getAttribute("shape") != nullptr) {
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
          return;
        }
        // The type constraints above do not allow for input_as_shape
        // and may need to be fixed.
        if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
          return;
        std::vector<int64_t> extra_shape;
        getRepeatedAttribute(ctx, "extra_shape", extra_shape);
        if (hasInputShape(ctx, 0)) {
          ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
          for (auto extra_dim_val : extra_shape) {
            if (extra_dim_val < 0)
              fail_shape_inference(
                  "Negative values are not allowed in a shape specification");
            shape.add_dim()->set_dim_value(extra_dim_val);
          }
          updateOutputShape(ctx, 0, shape);
        }
      });

  static const char* Scale_ver1_doc = R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Scale)
      .SinceVersion(1)
      .Input(0, "input", "Input data to be scaled", "T")
      .Output(0, "output", "Output data after scaling", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .SetDoc(Scale_ver1_doc)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* GRUUnit_ver1_doc = R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(GRUUnit)
      .SinceVersion(1)
      .SetDoc(GRUUnit_ver1_doc)
      .Attr("drop_states",
            "Bool to determine if hidden state is zeroes or passed "
            "along for timesteps past the given sequence_length.",
            AttributeProto::INT, OPTIONAL_VALUE)
      .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
      .Input(
          1,
          "gates",
          "Unactivated gate outputs from forget, update, "
          "and output gates, pre-activation.",
          "T")
      .Input(
          2,
          "seq_lengths",
          "Array of sequence lengths.  "
          "len(seq_lengths) should equal batch size N.",
          "T")
      .Input(3, "t", "The timestep for this operation.", "T")
      .Output(
          0,
          "hidden",
          "The new GRU hidden state calculated by this op.",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GivenTensorFill)
      .SinceVersion(10)
      .Deprecate()
      .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
      .Output(0, "X", "The filled tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .Attr("values", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
      .Attr("shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE)
      .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getAttribute("shape") != nullptr) {
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
          return;
        }
        // The type constraints above do not allow for input_as_shape
        // and may need to be fixed.
        if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
          return;
        std::vector<int64_t> extra_shape;
        getRepeatedAttribute(ctx, "extra_shape", extra_shape);
        if (hasInputShape(ctx, 0)) {
          ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
          for (auto extra_dim_val : extra_shape) {
            if (extra_dim_val < 0)
              fail_shape_inference(
                  "Negative values are not allowed in a shape specification");
            shape.add_dim()->set_dim_value(extra_dim_val);
          }
          updateOutputShape(ctx, 0, shape);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Scale)
      .SinceVersion(10)
      .Deprecate()
      .Input(0, "input", "Input data to be scaled", "T")
      .Output(0, "output", "Output data after scaling", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .SetDoc(Scale_ver1_doc)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(GRUUnit)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(GRUUnit_ver1_doc)
      .Attr("drop_states",
            "Bool to determine if hidden state is zeroes or passed "
            "along for timesteps past the given sequence_length.",
            AttributeProto::INT, OPTIONAL_VALUE)
      .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
      .Input(
          1,
          "gates",
          "Unactivated gate outputs from forget, update, "
          "and output gates, pre-activation.",
          "T")
      .Input(
          2,
          "seq_lengths",
          "Array of sequence lengths.  "
          "len(seq_lengths) should equal batch size N.",
          "T")
      .Input(3, "t", "The timestep for this operation.", "T")
      .Output(
          0,
          "hidden",
          "The new GRU hidden state calculated by this op.",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_OPERATOR_SCHEMA(MeanVarianceNormalization)
      .SinceVersion(1)
      .SetDoc(R"DOC(Perform mean variance normalization.)DOC")
      .Attr("across_channels", "If 1, mean and variance are computed across channels. Default is 0.", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("normalize_variance", "If 0, normalize the mean only.  Default is 1.", AttributeProto::INT, static_cast<int64_t>(1))
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_OPERATOR_SCHEMA(ScaledTanh)
      .SinceVersion(1)
      .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Input(0, "input", "Input tensor", "T")
      .Output(
          0,
          "output",
          "The scaled hyperbolic tangent values of the input tensor "
          "computed element-wise",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(Affine)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(Affine_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ParametricSoftplus)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(ParametricSoftplus_ver1_doc)
      .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D input tensor", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ImageScaler)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(ImageScaler_ver1_doc)
      .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL_VALUE)
      .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same shape and type as input", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(Crop)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(Crop_ver1_doc)
      .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS)
      .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL_VALUE)
      .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
      .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        // Type inference
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // Shape inference
        auto* output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        if (ONNX_NAMESPACE::hasNInputShapes(ctx, 1)) {
          const auto& input_shape =
              ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank =
              input_shape.dim_size();
          if (input_rank != 4)
            fail_shape_inference("Input's shape must be 4-D");

          // parse necessary attributes for futher processing
          std::vector<int64_t> border;
          bool border_present =
              getRepeatedAttribute(ctx, "border", border);
          if (!border_present || border.size() != 4)
            fail_shape_inference(
                "'Border' attribute must be present and must contain exactly 4 values - "
                "(left_border, top_border, right_border, bottom_border)");

          std::vector<int64_t> scale;
          bool scale_present =
              getRepeatedAttribute(ctx, "scale", scale);
          if (scale_present && scale.size() != 2)
            fail_shape_inference("'Scale' must contain exactly 2 values - (height, width)");

          // actual shape inference processing
          // [N, C] can be copied over from the input as is
          *output_shape->mutable_dim(static_cast<int>(0)) = input_shape.dim(static_cast<int>(0));
          *output_shape->mutable_dim(static_cast<int>(1)) = input_shape.dim(static_cast<int>(1));

          // process 'H' and 'W'
          if (!utils::HasDimValue(input_shape.dim(static_cast<int>(2))) ||
              !utils::HasDimValue(input_shape.dim(static_cast<int>(3)))) {
            // either height and width input has symbolic dims, so can't proceed further
            // add two dims as placeholders for output_H and output_W and return
            output_shape->add_dim();
            output_shape->add_dim();
            return;
          }

          int64_t H = input_shape.dim(static_cast<int>(2)).dim_value();
          int64_t W = input_shape.dim(static_cast<int>(3)).dim_value();

          int64_t left_border = border[0],
                  top_border = border[1],
                  right_border = border[2],
                  bottom_border = border[3];

          if (H < top_border + bottom_border)
            fail_shape_inference("Input's height (", H,
                                 ") needs to be greater than or equal to "
                                 "the top_border (",
                                 top_border, ") + bottom_border (", bottom_border, ")");

          if (W < left_border + right_border)
            fail_shape_inference("Input's width (", W,
                                 ") needs to be greater than or equal to "
                                 "the left_border (",
                                 left_border, ") + right_border (", right_border, ")");

          int64_t bottom_limit = H - bottom_border;
          int64_t right_limit = W - right_border;

          // scale = (height, width)
          if (!scale.empty()) {
            bottom_limit = top_border + scale[0];
            right_limit = left_border + scale[1];

            if (H < bottom_limit)
              fail_shape_inference("Input's height (", H, ") needs to be greater than or equal to the top_border (", top_border, ") + scale[0] (", scale[0], ")");

            if (W < right_limit)
              fail_shape_inference("Input's width (", W, ") needs to be greater than or equal to the left_border (", left_border, ") + scale[1] (", scale[1], ")");
          }

          auto* h_output_dim = output_shape->add_dim();
          h_output_dim->set_dim_value(bottom_limit - top_border);

          auto* w_output_dim = output_shape->add_dim();
          w_output_dim->set_dim_value(right_limit - left_border);

        } else {
          // Rank Inference at the very least
          // (We know that the output is going to be 4-D)
          for (int i = 0; i < 4; ++i) {
            output_shape->add_dim();
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicSlice)
      .SinceVersion(10)
      .Deprecate()
      .SetDoc(DynamicSlice_ver1_doc)
      .Input(0, "data", "Tensor of data to extract slices from.", "T")
      .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
      .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
      .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
      .Output(0, "output", "Sliced data tensor.", "T")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");

  ONNX_OPERATOR_SCHEMA(ScaledTanh)
      .SinceVersion(10)
      .Deprecate()
      .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
      .Input(0, "input", "Input tensor", "T")
      .Output(
          0,
          "output",
          "The scaled hyperbolic tangent values of the input tensor "
          "computed element-wise",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  // End of ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old version history maintenance

  ONNX_CONTRIB_OPERATOR_SCHEMA(SampleOp)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          ONNX_NAMESPACE::OpSchema::numeric_types_for_math_reduction(),
          "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetDoc(R"DOC(
Sample echo operator.)DOC");

  // register schemas for more operators here
  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxpoolWithMask)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Rfft)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC()DOC")
      .Input(0, "X", "input tensor", "T")
      .Attr("signal_ndim", "", AttributeProto::INT)
      .Attr("normalized", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("onesided", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Output(0, "Y", "output tensor", "T")
      .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Irfft)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC()DOC")
      .Input(0, "X", "input tensor", "T")
      .Attr("signal_ndim", "", AttributeProto::INT)
      .Attr("normalized", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("onesided", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Output(0, "Y", "output tensor", "T")
      .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ComplexMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC()DOC")
      .Input(0, "A", "input_0", "T")
      .Input(1, "B", "input_1", "T")
      .Output(0, "C", "output tensor", "T")
      .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ComplexMulConj)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC()DOC")
      .Input(0, "A", "input_0", "T")
      .Input(1, "B", "input_1", "T")
      .Output(0, "C", "output tensor", "T")
      .TypeConstraint("T", {"tensor(float)", "tensor(double)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ConvTransposeWithDynamicPads)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::convTransposeWithDynamicPadsShapeInference);

  ONNX_CONTRIB_OPERATOR_SCHEMA(FusedConv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      .Output(
          0,
          "Y",
          "",
          "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(FusedGemm)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ExpandDims)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      .SetDoc(R"DOC(ExpandDims echo operator.)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(AttnLSTM, RegisterAttnLSTMContribOpSchema);
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Range, RegisterRangeOpSchema);

  static const char* QuantizeLinear_ver1_doc = R"DOC(
The linear quantization operator. It consumes a full precision data, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QuantizeLinear)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "axis",
          "The axis along which same quantization parameters are applied. It's optional."
          "If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars."
          "If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors.",
          AttributeProto::INT,
          false)
      .Input(
          0,
          "x",
          "N-D full precision Input tensor to be quantized.",
          "T1")
      .Input(
          1,
          "y_scale",
          "Scale for doing quantization to get 'y'. It could be a scalar or a 1-D tensor,"
          "which means a per-tensor or per-axis quantization. If it's a 1-D tensor, "
          "its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T1")
      .Input(
          2,
          "y_zero_point",
          "Zero point for doing quantization to get 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor"
          "or per-axis quantization. If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T2")
      .Output(
          0,
          "y",
          "N-D quantized output tensor. It has same shape as input 'x'.",
          "T2")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)"},
          "Constrain 'x', 'y_scale' to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain 'y_zero_point' and 'y' to 8-bit integer tensors.")
      .SetDoc(QuantizeLinear_ver1_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 2, 0);

        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
      });

  static const char* DequantizeLinear_ver1_doc = R"DOC(
The linear dequantization operator. It consumes a quantized data, a scale, a zero point and computes the full precision data.
The dequantization formula is y = (x - x_zero_point) * x_scale.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(DequantizeLinear)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("axis",
            "The axis along which same quantization parameters are applied. It's optional."
            "If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars."
            "If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors.",
            AttributeProto::INT,
            false)
      .Input(
          0,
          "x",
          "N-D quantized Input tensor to be de-quantized.",
          "T1")
      .Input(
          1,
          "x_scale",
          "Scale for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization."
          "If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T2")
      .Input(
          2,
          "x_zero_point",
          "Zero point for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization."
          "If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'.",
          "T1")
      .Output(
          0,
          "y",
          "N-D full precision output tensor. It has same shape as input 'x'.",
          "T2")
      .TypeConstraint(
          "T1",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain 'x' and 'x_zero_point' to 8-bit integer tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(float16)", "tensor(float)"},
          "Constrain 'y', 'x_scale' to float tensors.")
      .SetDoc(DequantizeLinear_ver1_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto y_type = ctx.getOutputType(0);
        // only float is supported
        y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

        if (!hasInputShape(ctx, 0))
          return;

        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape);
      });

  static const char* Tokenizer_ver1_doc = R"DOC(
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

  ONNX_CONTRIB_OPERATOR_SCHEMA(Tokenizer)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulInteger16)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(DynamicQuantizeMatMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "A", "N-dimensional matrix A", "T1")
      .Input(1, "B", "N-dimensional matrix B", "T2")
      .Input(
          2,
          "b_scale",
          "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T1")
      .Input(
          3,
          "b_zero_point",
          "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T2",
          OpSchema::Optional)
      .Input(4,
             "bias",
             "1D input tensor, whose dimension is same as B's last dimension",
             "T1",
             OpSchema::Optional)
      .Output(0, "Y", "Matrix multiply results from A * B", "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain input A, b_scale and output Y data type as float tensor.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input B data type to 8-bit integer tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulIntegerToFloat)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "A", "N-dimensional matrix A", "T1")
      .Input(1, "B", "N-dimensional matrix B", "T2")
      .Input(
          2,
          "a_scale",
          "Scale of quantized input 'A'. It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'A'.",
          "T3")
      .Input(
          3,
          "b_scale",
          "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T3")
      .Input(
          4,
          "a_zero_point",
          "Zero point tensor for input 'A'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'A'.",
          "T1",
          OpSchema::Optional)
      .Input(
          5,
          "b_zero_point",
          "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
          "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
          "of elements should be equal to the number of columns of input 'B'.",
          "T2",
          OpSchema::Optional)
      .Input(
          6,
          "bias",
          "1D input tensor, whose dimension is same as B's last dimension",
          "T3",
          OpSchema::Optional)
      .Output(0, "Y", "Matrix multiply results from A * B", "T3")
      .TypeConstraint(
          "T1",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input A data type to 8-bit integer tensor.")
      .TypeConstraint(
          "T2",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input B data type to 8-bit integer tensor.")
      .TypeConstraint(
          "T3",
          {"tensor(float)"},
          "Constrain input a_scale, b_scale and output Y data type as float tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 2, 0);
        ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
      });

  static const char* TransposeScaleMatMul_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(TransposeScaleMatMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("TransposeScaleMatMul")
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
      .SetDoc(TransposeScaleMatMul_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        auto transAAttr = ctx.getAttribute("transA");
        bool transa = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transb = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
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
          for (int i = 0; i < rank0 - 2; ++i) {
            *shape0.add_dim() = shape0_raw.dim(i);
          }
          *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 1 : rank0 - 2);
          *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 2 : rank0 - 1);
        }

        auto rank1 = shape1_raw.dim_size();
        if (rank1 == 1) {
          // for vector input, transb does not make impact on the dim.
          shape1 = shape1_raw;
        } else {
          for (int i = 0; i < rank1 - 2; ++i) {
            *shape1.add_dim() = shape1_raw.dim(i);
          }
          *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 1 : rank1 - 2);
          *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 2 : rank1 - 1);
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceSumInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Computes the sum of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.)DOC")
      .Input(0, "data", "An input tensor.", "T1")
      .Output(0, "reduced", "Reduced output tensor.", "T2")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
      .TypeConstraint("T2",
                      {"tensor(int32)", "tensor(uint32)"},
                      "Constrain output data type to 32-bit integer tensor."
                      "T2 must be tensor(uint32) when T1 is tensor(uint8),"
                      "or must be tensor(int32) when T1 is tensor(int8).")
      .Attr(
          "axes",
          "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor.",
          AttributeProto::INTS)
      .Attr(
          "keepdims",
          "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
          AttributeProto::INT);

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearAdd)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .FillUsing(QLinearMathDocGenerator("addition",
                                         "C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point"));

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .FillUsing(QLinearMathDocGenerator("multiplication",
                                         "C = ((A - A_zero_point) * (B - B_zero_point)) * (A_scale * B_scale)/C_scale + C_zero_point"));

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearReduceMean)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(
Computes the mean of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.
Input and Output scales and zero points are used to requantize the output in a new range.
This helps to improve accuracy as after ReduceMean operation the range of the output is expected to decrease.

```
"Output = Dequantize(Input) -> ReduceMean on fp32 data -> Quantize(output)",

```
)DOC")
      .Input(0, "data", "An input tensor.", "T")
      .Input(
          1,
          "data_scale",
          "Input scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          2,
          "data_zero_point",
          "Input zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Input(
          3,
          "reduced_scale",
          "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          4,
          "reduced_zero_point",
          "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Output(0, "reduced", "Reduced output tensor.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input types to 8 bit signed and unsigned tensors.")
      .Attr(
          "axes",
          "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor.",
          AttributeProto::INTS)
      .Attr(
          "keepdims",
          "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
          AttributeProto::INT)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        auto data_type = ctx.getInputType(0);
        if (nullptr == data_type || data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        // validate scale and zero points
        ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 2, data_type->tensor_type().elem_type(), true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 4, data_type->tensor_type().elem_type(), true);

        int64_t keep_dims = 1;
        auto attr_proto = ctx.getAttribute("keepdims");
        if (attr_proto) {
          keep_dims = attr_proto->i();
        }

        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        auto output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        std::vector<int64_t> axes;
        auto axes_proto = ctx.getAttribute("axes");
        if (axes_proto)
          axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] < -input_ndim || axes[i] >= input_ndim) {
            fail_shape_inference(
                "axis must be in [-rank, rank-1]. input rank was ", input_ndim);
          }
          if (axes[i] < 0)
            axes[i] += input_ndim;
        }
        // do we need to handle negative axis?
        for (int i = 0; i < input_ndim; ++i) {
          // axes empty means reduce all dim
          if (!axes.empty() &&
              std::find(axes.begin(), axes.end(), i) == axes.end()) {
            auto dim = output_shape->add_dim();
            dim->CopyFrom(input_shape.dim(i));
          } else {
            if (keep_dims == 1) {
              auto dim = output_shape->add_dim();
              dim->set_dim_value(1);
            }
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MulInteger)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(Performs element-wise binary quantized multiplication (with Numpy-style broadcasting support).
"This operator supports **multidirectional (i.e., Numpy-style) broadcasting**"
The output of this op is the int32 accumulated result of the mul operation

```
C (int32) = (A - A_zero_point) * (B - B_zero_point)
```

)DOC")
      .Input(0, "A", "First operand.", "T")
      .Input(
          1,
          "A_zero_point",
          "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Input(2, "B", "Second operand.", "T")
      .Input(
          3,
          "B_zero_point",
          "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Output(0, "C", "Constrain output to 32 bit tensor", "T1")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input types to 8 bit signed and unsigned tensors.")
      .TypeConstraint("T1", {"tensor(int32)"}, "Constrain output types to 32 bit tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto c_type = ctx.getOutputType(0);
        c_type->mutable_tensor_type()->set_elem_type(
            ONNX_NAMESPACE::TensorProto::INT32);

        auto a_type = ctx.getInputType(0);
        auto b_type = ctx.getInputType(3);
        if (nullptr == a_type || nullptr == b_type ||
            a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
            b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        ValidateTypeAndShapeForScaleAndZP(ctx, 1, a_type->tensor_type().elem_type(), true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 3, b_type->tensor_type().elem_type(), true);

        if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2)) {
          bidirectionalBroadcastShapeInference(
              ctx.getInputType(0)->tensor_type().shape(),
              ctx.getInputType(2)->tensor_type().shape(),
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }
      });

  const char* QLinearAveragePoolDoc_ver1 = R"DOC(
 QLinearAveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```

The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).

Input and output scales and zero points are used to convert the output to a new quantization range.
Output = Dequantize(Input) -> AveragePool on fp32 data -> Quantize(output)
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearAveragePool)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(QLinearAveragePoolDoc_ver1)
      .Attr(
          "count_include_pad",
          "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "kernel_shape",
          "The size of the kernel along each axis.",
          AttributeProto::INTS)
      .Attr(
          "strides",
          "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
          AttributeProto::INTS,
          OPTIONAL_VALUE)
      .Attr(
          "auto_pad",
          contrib_ops_auto_pad_doc,
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr("pads", contrib_ops_pads_doc, AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr(
          "ceil_mode",
          "Whether to use ceil or floor (default) to compute the output shape.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(
          0,
          "X",
          "Input data tensor from the previous operator; "
          "dimensions for image case are (N x C x H x W), "
          "where N is the batch size, C is the number of "
          "channels, and H and W are the height and the "
          "width of the data. For non image case, the "
          "dimensions are in the form of "
          "(N x C x D1 x D2 ... Dn), where N is the batch "
          "size. Optionally, if dimension denotation is "
          "in effect, the operation expects the input "
          "data tensor to arrive with the dimension denotation "
          "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
          "T")
      .Input(
          1,
          "x_scale",
          "Input scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          2,
          "x_zero_point",
          "Input zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Input(
          3,
          "y_scale",
          "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
          "tensor(float)")
      .Input(
          4,
          "y_zero_point",
          "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
          "T",
          OpSchema::Optional)
      .Output(
          0,
          "Y",
          "Output data tensor from average or max pooling across "
          "the input tensor. Dimensions will vary based "
          "on various kernel, stride, and pad sizes. Floor value of "
          "the dimension is used",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to 8 bit tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

        auto data_type = ctx.getInputType(0);
        if (nullptr == data_type || data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        // validate scale and zero points
        ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 2, data_type->tensor_type().elem_type(), true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT, true);
        ValidateTypeAndShapeForScaleAndZP(ctx, 4, data_type->tensor_type().elem_type(), true);

        ONNX_NAMESPACE::convPoolShapeInference(ctx, false, true, 0, 5);
      });

  const char* QLinearLeakyReluDoc_ver1 = R"DOC(
QLinearLeakyRelu takes quantized input data (Tensor), an argument alpha, and quantize parameter for output,
and produces one output data (Tensor<T>) where the function `f(x) = quantize(alpha * dequantize(x)) for dequantize(x) < 0`,
`f(x) = quantize(dequantize(x)) for dequantize(x) >= 0`, is applied to the data tensor elementwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearLeakyRelu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(QLinearLeakyReluDoc_ver1)
      .Attr("alpha", "Coefficient of leakage.", AttributeProto::FLOAT, 0.01f)
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "X_scale",
             "Input X's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(2, "X_zero_point",
             "Input X's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Input(3, "Y_scale",
             "Output Y's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(4, "Y_zero_point",
             "Output Y's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to 8 bit tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  const char* QLinearSigmoidDoc_ver1 = R"DOC(
QLinearSigmoid takes quantized input data (Tensor), and quantize parameter for output, and produces one output data 
(Tensor<T>) where the function `f(x) = quantize(Sigmoid(dequantize(x)))`, is applied to the data tensor elementwise.
Wwhere the function `Sigmoid(x) = 1 / (1 + exp(-x))` )DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(QLinearSigmoid)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(QLinearSigmoidDoc_ver1)
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "X_scale",
             "Input X's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(2, "X_zero_point",
             "Input X's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Input(3, "Y_scale",
             "Output Y's scale. It's a scalar, which means a per-tensor/layer quantization.",
             "tensor(float)")
      .Input(4, "Y_zero_point",
             "Output Y's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization.",
             "T", OpSchema::Optional)
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(uint8)", "tensor(int8)"},
          "Constrain input and output types to 8 bit tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(MurmurHash3)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherND)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(WordConvEmbedding)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      .SetDoc(R"DOC(The WordConvEmbedding takes in a batch of sequence words and embed each word to a vector.)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Pad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
            )DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Unique)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
              )DOC");

  //see:https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
  ONNX_CONTRIB_OPERATOR_SCHEMA(CDist)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Constrains input to only numeric types.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(CropAndResize)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
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
        The resizing is corner aligned.)DOC");

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
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input data tensor from the previous layer.", "T")
      .Input(1, "scale", "Scale tensor.", "T")
      .Input(2, "B", "Bias tensor.", "T")
      .Output(0, "Y", "Output data tensor.", "T")
      .Output(1, "mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
      .Output(2, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types (except mean and inv_std_var) to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)"},
          "Constrain mean and inv_std_var to be float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
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
          saved_mean_shape->mutable_dim(static_cast<int>(axis))->set_dim_value(1);
        }

        if (ctx.getNumOutputs() > 2) {
          auto saved_inv_std_var_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
          saved_inv_std_var_shape->CopyFrom(input_shape);
          saved_inv_std_var_shape->mutable_dim(static_cast<int>(axis))->set_dim_value(1);
        }
      });

  // Register the NCHWc schemas if supported by the platform.
  if (MlasNchwcGetBlockSize() > 1) {
    RegisterNchwcSchemas();
  }

  static const char* Gelu_ver1_doc =
      R"DOC(Gaussian Error Linear Unit.
A high-performing neural network activation function.The GELU nonlinearity is
the expected transformation of a stochastic regularizer which randomly applies
the identity or zero map to a neuron's input. The GELU nonlinearity weights
inputs by their magnitude, rather than gates inputs by their sign as in ReLUs.)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Gelu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc(Gelu_ver1_doc)
      .Input(0, "X", "The input data as Tensor.", "T")
      .Output(0, "Y", "The output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  static const char* BiasGelu_ver1_doc =
      R"DOC(Bias Gelu.
It's an extension of Gelu. It takes the sum of input A and bias input B as the input of Gelu activation. )DOC";
  ONNX_CONTRIB_OPERATOR_SCHEMA(BiasGelu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc(BiasGelu_ver1_doc)
      .Input(0, "A", "The normal input data.", "T")
      .Input(1, "B", "The bias input data that is a 1D tensor.", "T")
      .Output(0, "C", "The output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  // Used to be ONNX 1.7 Inverse(12)
  // Comment out docs not to increase the binary size
  //
  //  static const char* Inverse_ver1_doc = R"DOC(
  //Calculates inverse of a square matrix or batches of square matrices.
  //Inverse takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
  //and the inner-most 2 dimensions form square matrices. These matrices must be invertible (full-rank).
  //The behavior where one of the matrices is not invertible is undefined. The implementation can choose
  //to throw an error or output (garbage) results as is. The output is a tensor of shape `[*, M, M]`,
  //containing the individual inverses of all input submatrices.
  //)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(Inverse)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
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
      });

  static const char* Trilu_ver1_doc = R"DOC(
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

  ONNX_CONTRIB_OPERATOR_SCHEMA(Trilu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(Trilu_ver1_doc)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
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
      });

  RegisterBertSchemas();
}
}  // namespace contrib
}  // namespace onnxruntime
