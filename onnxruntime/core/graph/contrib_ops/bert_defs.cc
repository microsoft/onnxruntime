// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/quantization_defs.h"
#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/graph/contrib_ops/shape_inference_functions.h"
// Suppress a warning: global initializer calls a non-constexpr function 'symbol' which is from
// ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif
using namespace ::ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
void matmulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx);

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {

void DecoderAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 1) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 1);
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 2);
  }
  // Shape inference
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    updateOutputShape(ctx, 0, query_shape);
  }
  if (ctx.getNumOutputs() > 1) {
    if (hasInputShape(ctx, 6) && hasInputShape(ctx, 7)) {
      auto& cache_shape = getInputShape(ctx, 6);
      auto& cache_dims = cache_shape.dim();
      if (cache_dims.size() != 4) {
        fail_shape_inference("key and value cache shall be 4 dimensions");
      }
      // has_dim_value() will return false if value is dynamic
      if (cache_dims[0].has_dim_value() &&
          cache_dims[1].has_dim_value() &&
          cache_dims[2].has_dim_value() &&
          cache_dims[3].has_dim_value()) {
        ONNX_NAMESPACE::TensorShapeProto new_cache_shape;
        *new_cache_shape.add_dim() = cache_shape.dim(0);
        *new_cache_shape.add_dim() = cache_shape.dim(1);
        new_cache_shape.add_dim();
        *new_cache_shape.add_dim() = cache_shape.dim(3);

        updateOutputShape(ctx, 1, new_cache_shape);
        updateOutputShape(ctx, 2, new_cache_shape);
      }
    }
  }
}

void RemovePaddingTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Input 0: (batch_size, sequence_length, hidden_size)
  // Output 0: (total_tokens, hidden_size)
  // Output 1: (batch_size, sequence_length)
  // Output 2: (batch_size + 1)
  // Output 3: (1)
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 1, 1);

  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim().size() != 3) {
      fail_shape_inference("input shall be 3 dimensions");
    }

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    output_shape.add_dim();
    *output_shape.add_dim() = input_shape.dim(2);
    updateOutputShape(ctx, 0, output_shape);

    ONNX_NAMESPACE::TensorShapeProto token_offset_shape;
    *token_offset_shape.add_dim() = input_shape.dim(0);
    *token_offset_shape.add_dim() = input_shape.dim(1);
    updateOutputShape(ctx, 1, token_offset_shape);

    ONNX_NAMESPACE::TensorShapeProto cumulated_seq_len_shape;
    auto dim = cumulated_seq_len_shape.add_dim();
    if (input_shape.dim(0).has_dim_value()) {
      dim->set_dim_value(1 + input_shape.dim(0).dim_value());
    }
    updateOutputShape(ctx, 2, cumulated_seq_len_shape);

    ONNX_NAMESPACE::TensorShapeProto max_seq_len_shape;
    max_seq_len_shape.add_dim()->set_dim_value(1);
    updateOutputShape(ctx, 3, max_seq_len_shape);
  }
}

void RestorePaddingTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Input 0:  (total_tokens, hidden_size)
  // Input 1:  (batch_size, sequence_length)
  // Output 0: (batch_size, sequence_length, hidden_size)
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
    auto& input_shape = getInputShape(ctx, 0);
    auto& token_offset_shape = getInputShape(ctx, 1);

    if (input_shape.dim().size() != 2) {
      fail_shape_inference("input shall be 2 dimensions");
    }

    if (token_offset_shape.dim().size() != 2) {
      fail_shape_inference("token_offset shall be 2 dimensions");
    }

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    *output_shape.add_dim() = token_offset_shape.dim(0);
    *output_shape.add_dim() = token_offset_shape.dim(1);
    *output_shape.add_dim() = input_shape.dim(1);
    updateOutputShape(ctx, 0, output_shape);
  }
}

void MultiHeadAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx,
                                             int past_key_index,
                                             bool dmmha_packing = false) {
  // Output 0 has shape (batch_size, sequence_length, v_hidden_size)

  // Q, K and V without packing:
  //   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
  //   Input 1 (key) has shape (batch_size, kv_sequence_length, hidden_size)
  //   Input 2 (value) has shape (batch_size, kv_sequence_length, v_hidden_size)

  // Q, K and V without packing and past (cross attention):
  //   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
  //   Input 1 (key) has shape (batch_size, num_head, kv_sequence_length, head_size)
  //   Input 2 (value) has shape (batch_size, num_head, kv_sequence_length, head_size)

  // Packed KV:
  //   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
  //   Input 1 (batch_size, kv_sequence_length, num_heads, 2, head_size)
  //   Input 2  nullptr

  // Packed QKV:
  //   Input 0 (batch_size, sequence_length, num_heads, 3, head_size) or
  //           (batch_size, sequence_length, 3 * hidden_size))
  //           for DecoderMaskedMultiHeadAttention.
  //   Input 1  nullptr
  //   Input 2  nullptr

  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  int64_t sequence_length = 0;
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();

    if (query_dims.size() != 3 && query_dims.size() != 5) {
      fail_shape_inference("Inputs 0 (query) shall be 3 or 5 dimensions");
    }

    if (query_dims.size() == 5) {  // packed QKV
      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1];
      *output_shape.add_dim() = query_dims[2] * query_dims[4];
      updateOutputShape(ctx, 0, output_shape);
    } else if (hasInputShape(ctx, 2)) {
      auto& value_shape = getInputShape(ctx, 2);
      auto& value_dims = value_shape.dim();
      if (value_dims.size() != 3 && value_dims.size() != 4) {
        fail_shape_inference("Inputs 2 (value) shall be 3 or 4 dimensions");
      }

      if (value_dims.size() == 3) {
        sequence_length = value_dims[1].dim_value();
      }

      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1];
      *output_shape.add_dim() = value_dims.size() == 3
                                    ? (dmmha_packing ? value_dims[2] / 3 : value_dims[2])
                                    : value_dims[1] * value_dims[3];
      updateOutputShape(ctx, 0, output_shape);
    } else if (hasInputShape(ctx, 1)) {
      auto& key_shape = getInputShape(ctx, 1);
      if (key_shape.dim().size() == 5) {  // packed KV
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
      }
    }
  }

  if (ctx.getNumOutputs() > 1) {  // has present output
    if (hasInputShape(ctx, past_key_index)) {
      auto& past_shape = getInputShape(ctx, past_key_index);
      auto& past_dims = past_shape.dim();
      if (past_dims.size() != 4) {
        fail_shape_inference("The past_key input shall be 4 dimensions");
      }

      auto past_present_share_buffer = getAttribute(ctx, "past_present_share_buffer", 0);
      if (past_present_share_buffer) {
        propagateElemTypeFromInputToOutput(ctx, past_key_index, 1);
        propagateElemTypeFromInputToOutput(ctx, static_cast<size_t>(past_key_index) + 1, 2);
      } else {
        if (sequence_length > 0 && past_dims[2].has_dim_value()) {
          int64_t total_sequence_length = sequence_length + past_dims[2].dim_value();

          ONNX_NAMESPACE::TensorShapeProto present_shape;
          for (auto& dim : past_dims) {
            *present_shape.add_dim() = dim;
          }
          present_shape.mutable_dim(2)->set_dim_value(total_sequence_length);

          updateOutputShape(ctx, 1, present_shape);
          updateOutputShape(ctx, 2, present_shape);
        }
      }
    }
  }
}

void GroupQueryAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int past_key_index) {
  // Output 0 has shape (batch_size, sequence_length, hidden_size)

  // Q, K and V:
  //   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
  //   Input 1 (key) has shape (batch_size, kv_sequence_length, kv_hidden_size)
  //   Input 2 (value) has shape (batch_size, kv_sequence_length, kv_hidden_size)

  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();

    if (query_dims.size() != 3) {
      fail_shape_inference("Inputs 0 (query) shall be 3 dimensions");
    }

    if (hasInputShape(ctx, 2)) {
      auto& value_shape = getInputShape(ctx, 2);
      auto& value_dims = value_shape.dim();
      if (value_dims.size() != 3) {
        fail_shape_inference("Inputs 2 (value) shall be 3 dimensions");
      }

      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1];
      *output_shape.add_dim() = query_dims[2];
      updateOutputShape(ctx, 0, output_shape);
      return;
    } else {
      fail_shape_inference("Missing input 2 (value)");
    }
  }

  if (ctx.getNumOutputs() > 1) {  // has present output
    if (hasInputShape(ctx, past_key_index)) {
      auto& past_shape = getInputShape(ctx, past_key_index);
      auto& past_dims = past_shape.dim();
      if (past_dims.size() != 4) {
        fail_shape_inference("The past_key input shall be 4 dimensions");
      }
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, past_key_index, 1);
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, static_cast<size_t>(past_key_index) + 1, 2);
      ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, past_key_index, 1);
      ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, static_cast<size_t>(past_key_index) + 1, 2);
    }
  }
}

constexpr const char* Attention_ver1_doc = R"DOC(
Multi-Head Attention that can be either unidirectional (like GPT-2) or bidirectional (like BERT).

The weights for input projection of Q, K and V are merged. The data is stacked on the second dimension. Its shape
is (input_hidden_size, hidden_size + hidden_size + v_hidden_size). Here hidden_size is the hidden dimension of Q and K,
and v_hidden_size is that of V.

The mask_index is optional. Besides raw attention mask with shape (batch_size, total_sequence_length)
or (batch_size, sequence_length, total_sequence_length) with value 0 for masked and 1 otherwise,
we support other two formats: When input has right-side padding, mask_index is one dimension with shape (batch_size),
where value is actual sequence length excluding padding. When input has left-side padding, mask_index has
shape (2 * batch_size), where the values are the exclusive end positions followed by the inclusive start positions.

When unidirectional is 1, each token only attends to previous tokens.

Both past and present state are optional. They shall be used together, and not allowed to use only one of them.
The qkv_hidden_sizes is required only when K and V have different hidden sizes.

When there is past state, hidden dimension for Q, K and V shall be the same.

The total_sequence_length is past_sequence_length + kv_sequence_length. Here kv_sequence_length is the length of K or V.
For self attention, kv_sequence_length equals to sequence_length (sequence length of Q).
For cross attention, query and key might have different lengths.
)DOC";

// Currently, the `convert_generation.py` script renames the `Attention` nodes to `DecoderMaskedSelfAttention`
// if the user requests it. Hence, the schemas of `DecoderMaskedSelfAttention` and `Attention` schemas
// are tightly coupled. A change in Attention also needs corresponding schema updates in `DecoderMaskedSelfAttention`
// and its kernel.
// TODO(hasesh): Decouple the schema of `DecoderMaskedSelfAttention` from the schema of the `Attention` operator
// by making appropriate tool changes.

ONNX_MS_OPERATOR_SET_SCHEMA(
    Attention, 1,
    OpSchema()
        .SetDoc(Attention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("unidirectional",
              "Whether every token can only attend to previous tokens. Default value is 0.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("qkv_hidden_sizes",
              "Hidden dimension of Q, K, V: hidden_size, hidden_size and v_hidden_size",
              AttributeProto::INTS,
              OPTIONAL_VALUE)
        .Attr("past_present_share_buffer",
              "Corresponding past and present are same tensor, its size is "
              "(2, batch_size, num_heads, max_sequence_length, head_size)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("do_rotary",
              "Whether to use rotary position embedding. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("mask_filter_value",
              "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, sequence_length, input_hidden_size)",
               "T")
        .Input(1,
               "weights",
               "Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
               "T")
        .Input(2,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) for input projection",
               "T",
               OpSchema::Optional)
        .Input(3,
               "mask_index",
               "Attention mask with shape (batch_size, 1, max_sequence_length, max_sequence_length), "
               "(batch_size, total_sequence_length) or (batch_size, sequence_length, total_sequence_length), "
               "or index with shape (batch_size) or (2 * batch_size) or (3 * batch_size + 2)",
               "M",
               OpSchema::Optional)
        .Input(4,
               "past",
               "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size)"
               "When past_present_share_buffer is set, "
               "its shape is (2, batch_size, num_heads, max_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(5,
               "relative_position_bias",
               "additional add to QxK' with shape (batch_size, num_heads, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_sequence_length",
               "When past_present_share_buffer is used, it is required to specify past_sequence_length (could be 0).",
               "M",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present",
                "past state for key and value with shape (2, batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (2, batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          constexpr int past_input_index = 4;
          AttentionTypeAndShapeInference(ctx, past_input_index);
        }));

constexpr const char* PackingAttention_ver1_doc = R"DOC(
This is the packed version of Attention.

Sequences in one batch usually don't have same length and they are padded to have same length,
e.g., below is a batch with 3 sequences and tokens* are padded.
  Sequence_0:   0,  1*, 2*,  3*
  Sequence_1:   4,  5,  6*,  7*
  Sequence_2:   8,  9,  10,  11

PackedAttention is designed to takes in packed input, i.e., only the real tokens without padding.
An input as above will be packed into 3 tensors like below:
 - input ([h0, h4, h5, h8, h9, h10, h11])
 - token_offset: 0, 4, 5, 8, 9, 10, 11,  1*, 2*, 3*, 6*, 7*
 - cumulated_token_count: 0, 1, 1+2, 1+2+4

Input tensors contains the hidden embedding of real tokens.
Token_offset records the offset of token in the unpacked input.
cumulated_token_count records cumulated length of each sequnces length.

The operator only supports BERT like model with padding on right now.

)DOC";

// Shape inference for PackedAttention. Here are the shapes of inputs and output:
// Input 'input':                      (token_count, input_hidden_size)
// Input 'weights':                    (input_hidden_size, hidden_size + hidden_size + v_hidden_size)
// Input 'bias':                       (hidden_size + hidden_size + v_hidden_size)
// Input 'token_offset':               (batch_size, sequence_length)
// Input 'cumulative_sequence_length': (batch_size + 1)
// Input 'relative_position_bias':     (batch_size, num_heads, sequence_length, sequence_length)
// Output 'output':                    (token_count, v_hidden_size)
void PackedAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2)) {
    auto& input_shape = getInputShape(ctx, 0);
    auto& input_dims = input_shape.dim();
    int input_dim_size = input_dims.size();
    if (input_dim_size != 2) {
      fail_shape_inference("Inputs 0 shall be 2 dimensions");
    }

    auto& bias_shape = getInputShape(ctx, 2);
    auto& bias_dims = bias_shape.dim();
    if (bias_dims.size() != 1) {
      fail_shape_inference("Invalid bias shape");
    }

    int64_t v_hidden_size = -1;
    std::vector<int64_t> qkv_hidden_sizes;
    getRepeatedAttribute(ctx, "qkv_hidden_sizes", qkv_hidden_sizes);

    if (qkv_hidden_sizes.size() != 0) {
      if (qkv_hidden_sizes.size() != 3) {
        fail_shape_inference("qkv_hidden_sizes should have 3 elements")
      }
      v_hidden_size = qkv_hidden_sizes[2];
    } else {
      v_hidden_size = bias_shape.dim(0).dim_value() / 3;
    }

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    for (auto& dim : input_dims) {
      *output_shape.add_dim() = dim;
    }

    output_shape.mutable_dim(input_dim_size - 1)->set_dim_value(v_hidden_size);
    updateOutputShape(ctx, 0, output_shape);
  }
}

ONNX_MS_OPERATOR_SET_SCHEMA(
    PackedAttention, 1,
    OpSchema()
        .SetDoc(PackingAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("qkv_hidden_sizes",
              "Hidden dimension of Q, K, V: hidden_size, hidden_size and v_hidden_size",
              AttributeProto::INTS,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "Input tensor with shape (token_count, input_hidden_size)",
               "T")
        .Input(1,
               "weights",
               "Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
               "T")
        .Input(2,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) for input projection",
               "T")
        .Input(3,
               "token_offset",
               "In packing mode, it specifies the offset of each token(batch_size, sequence_length).",
               "M")
        .Input(4,
               "cumulative_sequence_length",
               "A tensor with shape (batch_size + 1). It specifies the cumulative sequence length.",
               "M")
        .Input(5,
               "relative_position_bias",
               "A tensor with shape (batch_size, num_heads, sequence_length, sequence_length)"
               "or (1, num_heads, sequence_length, sequence_length)."
               "It specifies the additional bias to QxK'",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "2D output tensor with shape (token_count, v_hidden_size)",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          PackedAttentionTypeAndShapeInference(ctx);
        }));

constexpr const char* PackedMultiHeadAttention_ver1_doc = R"DOC(
This is the packed version of MultiHeadAttention.

Sequences in one batch usually don't have same length and they are padded to have same length,
e.g., below is a batch with 3 sequences and * is padding token.
  Sequence_0:   0,  1*, 2*,  3*
  Sequence_1:   4,  5,  6*,  7*
  Sequence_2:   8,  9,  10,  11

PackedMultiHeadAttention is designed to takes in packed input, i.e., only the real tokens without padding.
An input as above will be packed into 3 tensors like below:
 - query ([q0, q4, q5, q8, q9, q10, q11])
 - key ([k0, k4, k5, k8, k9, k10, k11])
 - value ([v0, v4, v5, v8, v9, v10, v11])
 - token_offset: 0, 4, 5, 8, 9, 10, 11,  1*, 2*, 3*, 6*, 7*
 - cumulative_sequence_length: 0, 1, 1+2, 1+2+4

The query, key and value tensors contain result of hidden embedding of real tokens after input projections.
Token_offset records the offset of token in the unpacked input.
cumulative_sequence_length records cumulated length of each sequnces length.

The operator only supports BERT like model with padding on right now.
)DOC";

// Shape inference for PackedMultiHeadAttention. Here are the shapes of inputs and output:
// When Q, K and V are not packed:
//   Input 'query':                      (token_count, hidden_size)
//   Input 'key':                        (token_count, hidden_size)
//   Input 'value':                      (token_count, v_hidden_size)
// When Q, K and V are packed:
//   Input 'query':                      (token_count, num_heads, 3, head_size)
//   Input 'key':                        None
//   Input 'value':                      None
// Input 'bias':                         (hidden_size + hidden_size + v_hidden_size)
// Input 'token_offset':                 (batch_size, sequence_length)
// Input 'cumulative_sequence_length':   (batch_size + 1)
// Input 'relative_position_bias':       (batch_size or 1, num_heads, sequence_length, sequence_length) or None
// Output 'output':                      (token_count, v_hidden_size)
void PackedMultiHeadAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();

    if (query_dims.size() != 2 && query_dims.size() != 4) {
      fail_shape_inference("Inputs 0 (query) shall be 2 or 4 dimensions");
    }

    if (query_dims.size() == 4) {  // packed QKV
      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = query_dims[1] * query_dims[3];
      updateOutputShape(ctx, 0, output_shape);
      return;
    }

    if (hasInputShape(ctx, 2)) {
      auto& value_shape = getInputShape(ctx, 2);
      auto& value_dims = value_shape.dim();
      if (value_dims.size() != 2) {
        fail_shape_inference("Inputs 2 (value) shall be 2 dimensions");
      }

      ONNX_NAMESPACE::TensorShapeProto output_shape;
      *output_shape.add_dim() = query_dims[0];
      *output_shape.add_dim() = value_dims[1];
      updateOutputShape(ctx, 0, output_shape);
      return;
    }
  }
}

ONNX_MS_OPERATOR_SET_SCHEMA(
    PackedMultiHeadAttention, 1,
    OpSchema()
        .SetDoc(PackedMultiHeadAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("mask_filter_value", "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (token_count, hidden_size) or packed qkv with shape (token_count, num_heads, 3, head_size)",
               "T")
        .Input(1,
               "key",
               "Key with shape (token_count, hidden_size)",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (token_count, v_hidden_size)",
               "T",
               OpSchema::Optional)
        .Input(3,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) from input projection",
               "T",
               OpSchema::Optional)
        .Input(4,
               "token_offset",
               "Offset of each token before packing, with shape (batch_size, sequence_length).",
               "M")
        .Input(5,
               "cumulative_sequence_length",
               "A tensor with shape (batch_size + 1). It specifies the cumulative sequence length.",
               "M")
        .Input(6,
               "relative_position_bias",
               "It specifies the additional bias to QxK'. The shape is (batch_size, num_heads, sequence_length, sequence_length)"
               " or (1, num_heads, sequence_length, sequence_length)",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "output tensor with shape (token_count, v_hidden_size)",
                "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask, offset and sequence length to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          PackedMultiHeadAttentionTypeAndShapeInference(ctx);
        }));

void propagateShapeAndTypeFromFirstInputAndParam(ONNX_NAMESPACE::InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
  // fix output_shape
  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < output_shape->dim_size(); i++) {
    auto* dim_i = output_shape->mutable_dim(i);
    if (dim_i->has_dim_param() && dim_i->dim_value() == 0) {
      dim_i->set_dim_value(-1);
    }
  }
}

constexpr const char* PagedAttention_ver1_doc = R"DOC(
PagedAttention is from https://vllm.ai/
It consists of two types of attention.
1. packed attention
2. Single token attention with paged kv-cache.
It requires a input_metadata for now from the python side, which contains the following information:
1. batch_size
2. sequence_length
3. Promot length
4. generation token length
5. table mapping
)DOC";
ONNX_MS_OPERATOR_SET_SCHEMA(
    PagedAttention, 1,
    OpSchema()
        .SetDomain(kMSDomain)
        .SetDoc(PagedAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("num_kv_heads", "Number of attention  kv heads, GQA/MQA, shared heads", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("head_size", "Hidden dimension of Q, K, V: hidden_size, hidden_size and v_hidden_size", AttributeProto::INT)
        .Attr("scale", "Custom scale will be used if specified. Default value is 1/sqrt(head_size)", AttributeProto::FLOAT)
        .Attr("mask_type", "position_mask_type, support [normal, alibi, RoPE]", AttributeProto::STRING)
        //.AllowUncheckedAttributes()
        .Input(0, "query", "The input Q-Tensor with shape(batch,seqlen,num-heads, head-size).", "T")
        .Input(1, "key", "The input K-Tensor with shape(batch,seqlen,num-heads, head-size).", "T")
        .Input(2, "value", "The input V-Tensor with shape(batch,seqlen,num-heads, head-size).", "T")
        .Input(3, "key_cache", "Blocked key cache in this layer.", "T2")
        .Input(4, "value_cache", "Blocked value cache in this layer.", "T2")
        .Input(5, "input_metadata", "Block mapping for each token, and some other eseential infos in InputMetadata, This input Tensor has shape [1], the value is a pointer of struct InputMetadata. It should be converted into a class and used then", "T1", OpSchema::Optional)
        .Input(6, "positions", "positions used for RoPE embedding", "T1", OpSchema::Optional)
        .Input(7, "cos_sin_cache_or_alibi_bais", "cos_sin_cache used for RoPE embedding, alibi for alibi embinding", "T3", OpSchema::Optional)
        .Input(8, "kv_quant_param", "quantization param for kvcache, like scale and zeropoint", "T", OpSchema::Optional)
        .Output(0, "output", "Attention output", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(bfloat16)"},
                        "Constrain input and output types to float/ tensors.")
        .TypeConstraint("T1", {"tensor(int64)"}, "Constrain input META types to pointer tensors.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(float16)", "tensor(float)", "tensor(bfloat16)"}, "kvcache and quant scale")
        .TypeConstraint("T3", {"tensor(float16)", "tensor(float)", "tensor(bfloat16)"}, "alibi scopt or cos_sin_cache")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInputAndParam(ctx);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    RmsNormalization, 1,
    OpSchema()
        .SetDomain(kMSDomain)
        .SetDoc("RMS Normalization Fusion")
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon)
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Input(1, "weight", "2D input tensor with shape (hidden_size)", "T")
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
              propagateShapeAndTypeFromFirstInputAndParam(ctx);
            }));

void SiluMulShapeInfer(InferenceContext& ctx) {
  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  int last_dim = input_shape.dim_size() - 1;
  for (int i = 0; i < input_shape.dim_size(); i++) {
    int64_t value = input_shape.dim(i).dim_value();
    if (i == last_dim) {
        value = value / 2;
    }
    output_shape->add_dim()->set_dim_value(value);
  }
}
ONNX_MS_OPERATOR_SET_SCHEMA(
    SiluAndMul, 1,
    OpSchema()
        .SetDomain(kMSDomain)
        .SetDoc("silu and mul ")
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          SiluMulShapeInfer(ctx);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    DebugStep, 1,
    OpSchema()
        .SetDomain(kMSDomain)
        .SetDoc("debug ")
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* DecoderMaskedSelfAttention_ver1_doc = R"DOC(
Self attention that supports input sequence length of 1.

The weights for input projection of Q, K and V are merged. The data is stacked on the second dimension. Its shape
is (input_hidden_size, hidden_size + hidden_size + v_hidden_size). Here hidden_size is the hidden dimension of Q and K,
and v_hidden_size is that of V.

The mask_index is optional. If it is provided, only raw attention mask with shape (batch_size, total_sequence_length) is supported currently.

Both past and present state need to be provided.

The qkv_hidden_sizes is required only when K and V have different hidden sizes.

The total_sequence_length is past_sequence_length + kv_sequence_length. Here kv_sequence_length is the length of K or V.
Currently, only self attention is supported which means that kv_sequence_length equals to sequence_length (sequence length of Q).
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DecoderMaskedSelfAttention, 1,
    OpSchema()
        .SetDoc(DecoderMaskedSelfAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("past_present_share_buffer",
              "Corresponding past and present are same tensor, its size is "
              "(2, batch_size, num_heads, max_sequence_length, head_size)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("mask_filter_value",
              "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("do_rotary",
              "Whether to use rotary position embedding. Default value is 0.",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, 1, input_hidden_size)",
               "T")
        .Input(1,
               "weights",
               "Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
               "T")
        .Input(2,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) for input projection",
               "T")
        .Input(3,
               "mask_index",
               "Mask values of shape (batch_size, total_sequence_length)",
               "M",
               OpSchema::Optional)
        .Input(4,
               "past",
               "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size)"
               "When past_present_share_buffer is set, "
               "its shape is (2, batch_size, num_heads, max_sequence_length, head_size). "
               "The first `batch_size * num_heads * max_sequence_length * head_size` elements correspond to keys "
               "and the next `batch_size * num_heads * max_sequence_length * head_size` elements correspond to values. "
               "The keys buffer is re-ordered in such a way that its virtual sub-tensor of shape "
               "(batch_size, num_heads, max_sequence_length, head_size) which may be perceived as being of shape "
               "(batch_size, num_heads, max_sequence_length, head_size / x, x) is reordered to "
               "become (batch_size, num_heads, head_size / x, max_sequence_length, x) where `x = 16 / sizeof(T)`.",
               "T")
        .Input(5,
               "relative_position_bias",
               "additional add to QxK' with shape (batch_size, num_heads, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_sequence_length",
               "When past_present_share_buffer is used, it is required to specify past_sequence_length (could be 0).",
               "M")
        .Input(7,
               "beam_width",
               "The beam width that is being used while decoding."
               "If not provided, the beam width will be assumed to be 1.",
               "M",
               OpSchema::Optional)
        .Input(8,
               "cache_indirection",
               "A buffer of shape [batch_size, beam_width, max_output_length] where an [i, j, k] entry specifies"
               "which beam the 'k' th token came from for the 'j' th beam for batch 'i' in the current iteration",
               "M",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present",
                "past state for key and value with shape (2, batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (2, batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          constexpr int past_input_index = 4;
          AttentionTypeAndShapeInference(ctx, past_input_index);
        }));

constexpr const char* DecoderMaskedMultiHeadAttention_ver1_doc = R"DOC(
Multihead attention that supports input sequence length of 1.
Similar to DecoderMaskedSelfAttention but this op excludes QKV MatMul and Bias.
This op supports both Self and Cross Attention.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DecoderMaskedMultiHeadAttention, 1,
    OpSchema()
        .SetDoc(DecoderMaskedMultiHeadAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("past_present_share_buffer",
              "Corresponding past and present are same tensor, its size is "
              "(batch_size, num_heads, max_sequence_length, head_size)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("mask_filter_value",
              "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("output_qk",
              "Need output the cross attention MatMul(Q, K)",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (batch_size, 1, hidden_size) or packed QKV with shape "
               "(batch_size, 1, 2 * hidden_size + v_hidden_size)",
               "T")
        .Input(1,
               "key",
               "Key with shape (batch_size, 1, hidden_size) for self attention "
               "or past_key with shape (batch_size, num_heads, kv_sequence_length, head_size) for cross attention",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (batch_size, 1, v_hidden_size) for self attention "
               "or past_value with shape (batch_size, num_heads, kv_sequence_length, head_size) for cross attention",
               "T",
               OpSchema::Optional)
        .Input(3,
               "mask_index",
               "Mask values of shape (batch_size, total_sequence_length) or (batch_size, kv_sequence_length)",
               "M",
               OpSchema::Optional)
        .Input(4,
               "relative_position_bias",
               "additional add to QxK' with shape (batch_size, num_heads, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(5,
               "past_key",
               "past state for key with shape (batch_size, num_heads, past_sequence_length, head_size) for self attention"
               "When past_present_share_buffer is set, "
               "its shape is (batch_size, num_heads, max_sequence_length, head_size). "
               // The re-ordering happens only for CUDA EP at the moment. We probably shall support 4 or 5D shape or
               // attribute to distinguish whether it is re-ordered or not.
               "The keys buffer is re-ordered in such a way that its virtual sub-tensor of shape "
               "(batch_size, num_heads, max_sequence_length, head_size) which may be perceived as being of shape "
               "(batch_size, num_heads, max_sequence_length, head_size / x, x) is reordered to "
               "become (batch_size, num_heads, head_size / x, max_sequence_length, x) where `x = 16 / sizeof(T)`.",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_value",
               "past state for value with shape (batch_size, num_heads, past_sequence_length, head_size) for self attention"
               "When past_present_share_buffer is set, "
               "its shape is (batch_size, num_heads, max_sequence_length, head_size). ",
               "T",
               OpSchema::Optional)
        .Input(7,
               "past_sequence_length",
               "When past_present_share_buffer is used, it is required to specify past_sequence_length (could be 0)."
               "Cross Attention doesn't need this input.",
               "M",
               OpSchema::Optional)
        .Input(8,
               "beam_width",
               "The beam width that is being used while decoding."
               "If not provided, the beam width will be assumed to be 1.",
               "M",
               OpSchema::Optional)
        .Input(9,
               "cache_indirection",
               // This input is useful for CUDA EP only.
               "A buffer of shape [batch_size, beam_width, max_output_length] where an [i, j, k] entry specifies"
               "which beam the 'k' th token came from for the 'j' th beam for batch 'i' in the current iteration",
               "M",
               OpSchema::Optional)
        .Input(10,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) from input projection",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present_key",
                "present state for key with shape (batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T",
                OpSchema::Optional)
        .Output(2,
                "present_value",
                "present state for value with shape (batch_size, num_heads, total_sequence_length, head_size). "
                "If past_present_share_buffer is set, "
                "its shape is (batch_size, num_heads, max_sequence_length, head_size), "
                "while effective_seq_length = (past_sequence_length + kv_sequence_length).",
                "T",
                OpSchema::Optional)
        .Output(3,
                "qk",
                "normalized Q * K, of shape (batch_size, num_heads, 1, head_size). ",
                "V",
                OpSchema::Optional)
        .TypeConstraint("V", {"tensor(float)"}, "Constrain qk output types to float32 tensors.")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain mask index to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          bool is_dmmha_packing = !hasInputShape(ctx, 1) && !hasInputShape(ctx, 2);
          MultiHeadAttentionTypeAndShapeInference(ctx, 5, is_dmmha_packing);
        }));

constexpr const char* MultiHeadAttention_ver1_doc = R"DOC(
Multi-Head Self/Cross Attention. Bias from input projection is included.

The key padding mask is optional. When its shape is (batch_size, kv_sequence_length), value 0
means padding or 1 otherwise. When key has right-side padding, its shape could be (batch_size): it is actual length of
each key sequence excluding paddings.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    MultiHeadAttention, 1,
    OpSchema()
        .SetDoc(MultiHeadAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("mask_filter_value", "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (batch_size, sequence_length, hidden_size), or packed QKV with shape (batch_size, kv_sequence_length, num_heads, 3, head_size)",
               "T")
        .Input(1,
               "key",
               "Key with shape (batch_size, kv_sequence_length, hidden_size), or packed KV with shape (batch_size, kv_sequence_length, num_heads, 2, head_size), "
               "or past_key with shape (batch_size, num_heads, kv_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(2,
               "value",
               "Value with shape (batch_size, kv_sequence_length, v_hidden_size), or past_value with shape (batch_size, num_heads, kv_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(3,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) from input projection",
               "T",
               OpSchema::Optional)
        .Input(4,
               "key_padding_mask",
               "Key padding mask with shape (batch_size) or (3 * batch_size + 2) or (batch_size, kv_sequence_length)",
               "M",
               OpSchema::Optional)
        .Input(5,
               "relative_position_bias",
               "relative position bias: addition to QxK' with shape (batch_size, num_heads, sequence_length, total_sequence_length)"
               " or (1, num_heads, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(6,
               "past_key",
               "past state for self attention key with shape (batch_size, num_heads, past_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(7,
               "past_value",
               "past state for self attention value with shape (batch_size, num_heads, past_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present_key",
                "present state for cross attention key with shape (batch_size, num_heads, kv_sequence_length, head_size)"
                "or present state for self attention key with shape (batch_size, num_heads, total_sequence_length, head_size)",
                "T",
                OpSchema::Optional)
        .Output(2,
                "present_value",
                "present state for cross attention value with shape (batch_size, num_heads, kv_sequence_length, head_size)"
                "or present state for self attention value with shape (batch_size, num_heads, total_sequence_length, head_size)",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain mask to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          MultiHeadAttentionTypeAndShapeInference(ctx, 6);
        }));

constexpr const char* GroupQueryAttention_ver1_doc = R"DOC(
Group Query Self/Cross Attention.

Supports different number of heads for q and kv.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GroupQueryAttention, 1,
    OpSchema()
        .SetDoc(GroupQueryAttention_ver1_doc)
        .Attr("num_heads", "Number of attention heads for q", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of attention heads for k and v", AttributeProto::INT)
        .Attr("unidirectional",
              "Whether every token can only attend to previous tokens. Default value is 1.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("is_past_bsnh",
              "Whether past kv uses BSNH, otherwise BNSH. Default value is 1 (BSNH).",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("scale",
              "Custom scale will be used if specified. Default value is 1/sqrt(head_size)",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Input(0,
               "query",
               "Query with shape (batch_size, sequence_length, hidden_size)",
               "T")
        .Input(1,
               "key",
               "Key with shape (batch_size, kv_sequence_length, kv_hidden_size) ",
               "T")
        .Input(2,
               "value",
               "Value with shape (batch_size, kv_sequence_length, kv_hidden_size)",
               "T")
        .Input(3,
               "past_key",
               "past state key with support for format BSNH or BNSH. When past_key uses same tensor as present_key"
               "(k-v cache), it is of length max_sequence_length... otherwise of length past_sequence_length.",
               "T",
               OpSchema::Optional)
        .Input(4,
               "past_value",
               "past state value with support for format BSNH or BNSH. When past_value uses same tensor as present_value"
               "(k-v cache), it is of length max_sequence_length... otherwise of length past_sequence_length.",
               "T",
               OpSchema::Optional)
        .Input(5,
               "past_sequence_length",
               "When buffered past_key and past_value is used (present_key uses same tensor as past_key), required"
               "to specify past_sequence_length (could be 0). Otherwise, past_sequence_length inferred from past_key.",
               "M",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, hidden_size)",
                "T")
        .Output(1,
                "present_key",
                "present state key with support for format BSNH or BNSH. When past_key uses same tensor as present_key"
                "(k-v buffer), it is of length max_sequence_length... otherwise of length past_sequence_length +"
                "kv_sequence_length.",
                "T",
                OpSchema::Optional)
        .Output(2,
                "present_value",
                "present state value with support for format BSNH or BNSH. When past_value uses same tensor as present_value"
                "(k-v buffer), it is of length max_sequence_length... otherwise of length past_sequence_length +"
                "kv_sequence_length.",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float16)"}, "Constrain input and output to float tensors.")
        .TypeConstraint("M", {"tensor(int32)", "tensor(int64)"}, "Constrain past sequence length to int tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          GroupQueryAttentionTypeAndShapeInference(ctx, 3);
        }));

constexpr const char* Longformer_Attention_doc = R"DOC(
Longformer Self Attention with a local context and a global context. Tokens attend locally: Each token
attends to its W previous tokens and W succeeding tokens with W being the window length. A selected few tokens
attend globally to all other tokens.

The attention mask is of shape (batch_size, sequence_length), where sequence_length is a multiple of 2W after padding.
Mask value < 0 (like -10000.0) means the token is masked, 0 otherwise.

Global attention flags have value 1 for the tokens attend globally and 0 otherwise.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    LongformerAttention, 1,
    OpSchema()
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(Longformer_Attention_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("window", "One sided attention windows length W, or half of total window length", AttributeProto::INT)
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * head_size", "T")
        .Input(1, "weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "T")
        .Input(2, "bias", "1D input tensor with shape (3 * hidden_size)", "T")
        .Input(3, "mask", "Attention mask with shape (batch_size, sequence_length)", "T")
        .Input(4, "global_weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "T")
        .Input(5, "global_bias", "1D input tensor with shape (3 * hidden_size)", "T")
        .Input(6, "global", "Global attention flags with shape (batch_size, sequence_length)", "G")
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("G", {"tensor(int32)"}, "Constrain to integer types")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Decoder_Attention_doc = R"DOC(
This DecoderAttention supports self attention and cross attention, key and value cache, and key_padding_mask. The attention mask is not support at the moment.
Some boolean parameters are passed by runtime input for generic purpose
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    DecoderAttention, 1,
    OpSchema()
        .SetDoc(Decoder_Attention_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Attr("mask_filter_value", "The value to be filled in the attention mask. Default value is -10000.0f",
              AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "query", "3D input tensor with shape (sequence_length, batch_size, hidden_size), hidden_size = num_heads * head_size", "T")
        .Input(1, "key", "3D input tensor with shape (total_sequence_length, batch_size, hidden_size)", "T")
        .Input(2, "q_weight", "2D input tensor with shape (hidden_size, hidden_size)", "T")
        .Input(3, "kv_weight", "2D input tensor with shape (hidden_size, 2 * hidden_size)", "T")
        .Input(4, "bias", "1D input tensor with shape (3 * hidden_size)", "T")
        .Input(5, "key_padding_mask", "2D input tensor with shape (batch_size, total_sequence_length)", "B", OpSchema::Optional)
        .Input(6, "key_cache", "input tensor with shape (batch_size, num_heads, sequence_length or total_sequence_length, head_size)", "T", OpSchema::Optional)    // self & cross
        .Input(7, "value_cache", "input tensor with shape (batch_size, num_heads, sequence_length or total_sequence_length, head_size)", "T", OpSchema::Optional)  // self & cross
        .Input(8, "static_kv", "If static_kv = true, cross-attention; else self-attention", "B")
        .Input(9, "use_past", "If use_past = true, use cache; else no cache", "B")
        .Input(10, "has_layer_state", "If has_layer_state = true, layer_state = {} or [a,b]; else layer_state = None", "B")
        .Input(11, "has_key_padding_mask", "has_key_padding_mask or not", "B")
        .Output(0, "output", "3D output tensor with shape (sequence_length, batch_size, hidden_size)", "T")
        .Output(1, "new_key_cache", "output tensor with shape (batch_size, num_heads, new sequence_length, head_size)", "T", OpSchema::Optional)    // self & cross
        .Output(2, "new_value_cache", "output tensor with shape (batch_size, num_heads, new sequence_length, head_size)", "T", OpSchema::Optional)  // self & cross
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float and float16 tensors.")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain key_padding_mask to bool tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          DecoderAttentionTypeAndShapeInference(ctx);
        }));

constexpr const char* EmbedLayerNormalization_ver1_doc = R"DOC(
EmbedLayerNormalization is the fusion of embedding layer in BERT model, with optional mask processing.
The embedding layer takes input_ids (word IDs) and segment_ids (sentence IDs) to look up word_embedding, position_embedding,
and segment_emedding; the embeddings are added then applied layer normalization using gamma and beta tensors.
The last input mask is optional. If mask is provided, mask index (that is position of first 0 in mask, or number of words)
will be calculated.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    EmbedLayerNormalization, 1,
    OpSchema()
        .SetDoc(EmbedLayerNormalization_ver1_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultEmbedLayerNormEpsilon)
        .Attr("mask_index_type", "The mask index tensor type for shape inference (0: None, 1: 1D mask_index)", AttributeProto::INT, OPTIONAL_VALUE)
        .Input(0, "input_ids", "2D words IDs with shape (batch_size, sequence_length)", "T1")
        .Input(1, "segment_ids", "2D segment IDs with shape (batch_size, sequence_length)", "T1", OpSchema::Optional)
        .Input(2, "word_embedding", "2D with shape (,hidden_size)", "T")
        .Input(3, "position_embedding", "2D with shape (, hidden_size)", "T")
        .Input(4, "segment_embedding", "2D with shape (, hidden_size)", "T", OpSchema::Optional)
        .Input(5, "gamma", "1D gamma tensor for layer normalization with shape (hidden_size)", "T")
        .Input(6, "beta", "1D beta tensor for layer normalization  with shape (hidden_size)", "T")
        .Input(7, "mask", "2D attention mask with shape (batch_size, sequence_length)", "T1", OpSchema::Optional)
        .Input(8, "position_ids", "2D position ids with shape (batch_size, sequence_length) or (1, sequence_length)", "T1", OpSchema::Optional)
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Output(1, "mask_index", "1D mask_index tensor with shape (batch_size)", "T1", OpSchema::Optional)
        .Output(2, "embedding_sum", "sum of word_embedding and position_embedding without layer normalization", "T", OpSchema::Optional)
        .TypeConstraint("T1", {"tensor(int32)"}, "Constrain input and output integer tensors types")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output float tensors types.")
        .TypeAndShapeInferenceFunction(EmbedLayerNormalizationShapeInference));

constexpr const char* FastGelu_ver1_doc = R"DOC(
GELU (Gaussian Error Linear Unit) approximation: Y=0.5*X*(1+tanh(0.797885*X+0.035677*X*X*X)) with an optional input of bias that will be added to X before GELU.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    FastGelu, 1,
    OpSchema()
        .SetDoc(FastGelu_ver1_doc)
        .Input(0, "X", "input tensor", "T")
        .Input(1, "bias", "bias tensor", "T", OpSchema::Optional)
        .Output(0, "Y", "output tensor", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float or half tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
          // fastgelu(x) =
          auto* tp = ctx.getInputType(0);
          if ((tp == nullptr) || (!tp->has_tensor_type()))
            return false;
          auto elem_type = (TensorProto_DataType)(tp->tensor_type().elem_type());

          // Optional input 1 indicates a bias to be added to input 0.
          auto hasBias = ctx.hasInput(1);

          FunctionBuilder builder(functionProto);
          builder
              .AddOpset("", 13)
              .Const("a", ToTensor(0.5, elem_type))
              .Const("b", ToTensor(0.797885, elem_type))
              .Const("c", ToTensor(0.035677, elem_type))
              .Const("one", ToTensor(1.0, elem_type))
              .Add(hasBias ? "X_bias = Add (X, bias)" : "X_bias = Identity (X)")
              .Add(R"(
                T1 = Mul (X_bias, X_bias)
                T2 = Mul (c, T1)
                T3 = Add (b, T2)
                T4 = Mul (X_bias, T3)
                T5 = Tanh (T4)
                T6 = Add (one, T5)
                T7 = Mul (X_bias, T6)
                Y = Mul (a, T7)
            )");

          schema.BuildFunction(functionProto);
          return true;
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    RelativePositionBias, 1,
    OpSchema()
        .SetDoc("Compute binned relative position bias for T5 model. ref: https://arxiv.org/abs/1803.02155v2")
        .Attr("max_distance", "Max distance", AttributeProto::INT)
        .Attr("is_bidirectional", "Default value is 0.", AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "bias_table", "2D input tensor with shape (num_buckets, num_heads), COL-major(See UT for example)", "T")
        .Input(1, "query_length", "The length of query. Self Attention requires query_length = key_length", "U")
        .Input(2, "key_length", "The length of key.", "U")
        .Output(0, "output", "4D output tensor with shape (1, num_heads, sequence_length, sequence_length)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
        .TypeConstraint("U", {"tensor(int64)"}, "Constrain sequence_length to int tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto& bias_table_shape = getInputShape(ctx, 0);
          TensorShapeProto output_shape;
          output_shape.add_dim()->set_dim_value(1);
          *output_shape.add_dim() = bias_table_shape.dim(1);
          output_shape.add_dim();
          output_shape.add_dim();
          updateOutputShape(ctx, 0, output_shape);
        }));

ONNX_MS_OPERATOR_SET_SCHEMA(
    SkipLayerNormalization, 1,
    OpSchema()
        .SetDoc("Skip and Layer Normalization Fusion")
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon)
        .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Input(1, "skip", "3D skip tensor with shape (batch_size, sequence_length, hidden_size) or (1, sequence_length, hidden_size) or (sequence_length, hidden_size)", "T")
        .Input(2, "gamma", "1D input tensor with shape (hidden_size)", "T")
        .Input(3, "beta", "1D skip tensor with shape (hidden_size", "T", OpSchema::Optional)
        .Input(4, "bias", "1D bias tensor with shape (hidden_size", "T", OpSchema::Optional)
        .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Output(1, "mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
        .Output(2, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
        .Output(3, "input_skip_bias_sum", "Sum of the input and skip inputs (and bias if it exists) with shape (batch_size, sequence_length, hidden_size).", "T", OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
        .TypeConstraint("U", {"tensor(float)"}, "Constrain mean and inv_std_var to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_MS_OPERATOR_SET_SCHEMA(
    SkipSimplifiedLayerNormalization, 1,
    OpSchema()
        .SetDoc("Skip and Root Mean Square Layer Normalization")
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon)
        .Input(0,
               "input",
               "3D input tensor with shape (batch_size, sequence_length, hidden_size)"
               "Or 2D input tensor with shape (token_count, hidden_size)",
               "T")
        .Input(1,
               "skip",
               "3D input tensor with shape (batch_size, sequence_length, hidden_size)"
               "Or 2D input tensor with shape (token_count, hidden_size)",
               "T")
        .Input(2,
               "gamma",
               "1D input tensor with shape (hidden_size)",
               "T")
        .Input(3,
               "bias",
               "1D bias tensor with shape (hidden_size",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, hidden_size)"
                "Or 2D output tensor with shape (token_count, hidden_size)",
                "T")
        .Output(1,
                "mean",
                "Saved mean used during training to speed up gradient computation",
                "U",
                OpSchema::Optional)
        .Output(2,
                "inv_std_var",
                "Saved inverse standard variance used during training to speed up gradient computation.",
                "U",
                OpSchema::Optional)
        .Output(3,
                "input_skip_bias_sum",
                "Sum of the input and skip inputs (and bias if it exists)"
                "with shape (batch_size, sequence_length, hidden_size) or (token_count, hidden_size).",
                "T",
                OpSchema::Optional)
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
        .TypeConstraint("U", {"tensor(float)"}, "Constrain mean and inv_std_var to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* NGramRepeatBlock_ver1_doc = R"DOC(
Enforce no repetition of n-grams. Scores are set to `-inf` for tokens that form a repeated n-gram if added to the back of the input_ids.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    NGramRepeatBlock, 1,
    OpSchema().SetDoc(NGramRepeatBlock_ver1_doc).Attr("ngram_size", "The NGram size.", AttributeProto::INT).Input(0, "input_ids", "2D input tensor with shape (batch_size, sequence_length)", "Tid").Input(1, "scores", "2D input tensor with shape (batch_size, vocab_size)", "T").Output(0, "scores_out", "2D output tensor with shape (batch_size, vocab_size)", "T").TypeConstraint("Tid", {"tensor(int64)"}, "Constrain indices to integer types").TypeConstraint("T", {"tensor(float)"}, "Constrain scores input and output types to float tensors.").TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 1, 0);
      if (!hasInputShape(ctx, 1)) {
        return;
      }
      propagateShapeFromInputToOutput(ctx, 1, 0);
    }));

constexpr const char* BifurcationDetector_ver1_doc = R"DOC(
Component for aggressive decoding. Find the bifurcation index of predicted tokens, between source tokens,
starting from previous suffix match index, and predicted tokens.
Concat predicted tokens, starting from bifurcation index, to the back
of current tokens. This forms the output tokens.
Detect suffix match index in source tokens, between source tokens and output tokens.
Detection is based on finding the appearances of last n-gram in output tokens
in source tokens.
A match is considered found if source tokens contain a single matching n-gram.
Return the index of the start of the n-gram in source tokens.
No matching if found if src tokens contain multiple or zero matching n-grams. Return -1.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    BifurcationDetector, 1,
    OpSchema()
        .SetDoc(BifurcationDetector_ver1_doc)
        .Attr("min_ngram_size", "The minimum NGram size for suffix matching.", AttributeProto::INT, static_cast<int64_t>(1))
        .Attr("max_ngram_size", "The maximum NGram size for suffix matching.", AttributeProto::INT, static_cast<int64_t>(3))
        .Input(0, "src_tokens", "Encoder input ids.", "T")
        .Input(1, "cur_tokens", "Decoder input ids.", "T")
        .Input(2, "prev_suffix_match_idx", "Previous suffix match index", "T")
        .Input(3, "pred_tokens", "Predicted token ids from aggressive decoding", "T", OpSchema::Optional)
        .Output(0, "tokens", "Decoder input ids after merging predicted tokens", "T")
        .Output(1, "suffix_match_idx", "new suffix match index", "T")
        .TypeConstraint("T", {"tensor(int64)"}, "Constrain to integer types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          propagateElemTypeFromInputToOutput(ctx, 2, 1);
          if (hasInputShape(ctx, 2)) {
            propagateShapeFromInputToOutput(ctx, 2, 1);
          }
          // output tokens lengths is dynamic as it depends on the bifurcation index of predicted tokens and source tokens,
          // and current tokens length.
          // tokens_length = cur_tokens_length + bifurcation_index + 1.
        }));

constexpr const char* GemmFastGelu_ver1_doc = R"DOC(
It's a fusion of MatMul and FastGelu.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GemmFastGelu, 1,
    OpSchema()
        .SetDoc(GemmFastGelu_ver1_doc)
        .Input(0, "X", "input tensor", "T")
        .Input(1, "W", "input tensor", "T")
        .Input(2, "bias", "bias tensor", "T", OpSchema::Optional)
        .Output(0, "Y", "output tensor", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                        "Constrain input and output types to float or half tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
          ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
        }));

constexpr const char* RemovePadding_ver1_doc = R"DOC(
Compress transformer input by removing paddings. It assumes padding is on the right side of sequence.

The input has padding with shape (batch_size, sequence_length, hidden_size). This will generate two outputs:
output has shape (total_tokens, hidden_size); token_offset with shape (batch_size, sequence_length).

token_offset has offsets of all non-padding tokens first, then offset of all padding tokens. It is
a list of batch_size * sequence_length elements, which is reshaped to 2D for convenience of shape inference.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    RemovePadding, 1,
    OpSchema()
        .SetDoc(RemovePadding_ver1_doc)
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, sequence_length, hidden_size)",
               "T")
        .Input(1,
               "sequence_token_count",
               "Number of non-padding tokens in each sequence with shape (batch_size).",
               "M")
        .Output(0,
                "output",
                "output tensor with shape (total_tokens, hidden_size)",
                "T")
        .Output(1,
                "token_offset",
                "Offset of non-padding tokens, and those of padding tokens. Its shape is (batch_size, sequence_length)",
                "M")
        .Output(2,
                "cumulated_seq_len",
                "Cumulated sequence lengths. Its shape is (batch_size + 1)",
                "M")
        .Output(3,
                "max_seq_len",
                "Max sequence length without padding. Its shape is (1)",
                "M")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain sequence_token_count and token_offset to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          RemovePaddingTypeAndShapeInference(ctx);
        }));

constexpr const char* RestorePadding_ver1_doc = R"DOC(
Restore paddings and fill padding with zeros.

The input has padding with shape (total_tokens, hidden_size) and token_offset with shape (batch_size, sequence_length).
The output has shape (batch_size, sequence_length, hidden_size).
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    RestorePadding, 1,
    OpSchema()
        .SetDoc(RestorePadding_ver1_doc)
        .Input(0,
               "input",
               "Input tensor with shape (total_tokens, hidden_size)",
               "T")
        .Input(1,
               "token_offset",
               "Offset of non-padding tokens and paddings. Its shape is (batch_size, sequence_length)",
               "M")
        .Output(0,
                "output",
                "output tensor with shape (batch_size, sequence_length, hidden_size)",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeConstraint("M",
                        {"tensor(int32)"},
                        "Constrain token_offset to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          RestorePaddingTypeAndShapeInference(ctx);
        }));

constexpr const char* GatedRelativePositionBias_ver1_doc = R"DOC(
  query_layer = (query_layer + query_bias).reshape(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
  gate_u, gate_r = torch.sigmoid(
      self.gate_ur_linear(query_layer).view(batch_size, num_head, seq_len, 2, D/2).sum(-1, keepdim=False)
  ).chunk(2, dim=-1)
  gate_u_1 = gate_u * (gate_r * self.eco_a - 1.0) + 2.0
  rel_pos_bias = gate_u_1 * rel_pos
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GatedRelativePositionBias, 1,
    OpSchema()
        .SetDoc(GatedRelativePositionBias_ver1_doc)
        .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
        .Input(0, "query_layer", "tensor with shape (batch_size, seq_len, num_heads x head_size) or (token_count, num_heads x head_size)", "T")
        .Input(1, "query_bias", "1-d tensor with shape (num_heads x head_size)", "T")
        .Input(2, "rel_pos", "tensor with shape (1, num_head, seq_len, seq_len)", "T")
        .Input(3, "weight", "gemm weight for the gated_ur_linear, shape (head_size, D), D is divisible by 2", "T")
        .Input(4, "bias", "bias for the gated_ur_linear, shape (D)", "T")
        .Input(5, "eco_a", "tensor of shape (1, num_heads, 1, 1)", "T")
        .Input(6, "token_offset", "offset of each token with shape (batch_size, seq_len)", "M", OpSchema::Optional)
        .Output(0, "output", "output tensor with shape (batch_size, num_heads, seq_len, seq_len)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("M", {"tensor(int32)"}, "Constrain token_offset to integer types")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          int64_t num_heads = getAttribute(ctx, "num_heads", -1L);

          // When padding is removed:
          //   query_layer: (token_count, num_heads x head_size)
          //   token_offset: (batch_size, seq_len)
          // Otherwise:
          //   query_layer: (batch_size, seq_len, num_heads x head_size)
          //   token_offset: None
          // Output shape: (batch_size, num_heads, seq_len, seq_len)
          if (hasInputShape(ctx, 6)) {
            auto& token_offset_shape = getInputShape(ctx, 6);
            TensorShapeProto output_shape;
            *output_shape.add_dim() = token_offset_shape.dim(0);
            output_shape.add_dim()->set_dim_value(num_heads);
            *output_shape.add_dim() = token_offset_shape.dim(1);
            *output_shape.add_dim() = token_offset_shape.dim(1);
            updateOutputShape(ctx, 0, output_shape);
          } else if (hasInputShape(ctx, 0)) {
            auto& query_layer_shape = getInputShape(ctx, 0);
            if (query_layer_shape.dim().size() == 3) {
              TensorShapeProto output_shape;
              *output_shape.add_dim() = query_layer_shape.dim(0);
              output_shape.add_dim()->set_dim_value(num_heads);
              *output_shape.add_dim() = query_layer_shape.dim(1);
              *output_shape.add_dim() = query_layer_shape.dim(1);
              updateOutputShape(ctx, 0, output_shape);
            }
          }
        }));

}  // namespace contrib
}  // namespace onnxruntime
