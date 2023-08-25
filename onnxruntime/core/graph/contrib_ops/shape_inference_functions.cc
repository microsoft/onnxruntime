// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/contrib_ops/shape_inference_functions.h"
#include "core/graph/onnx_protobuf.h"
#include <onnx/defs/shape_inference.h>
#include <iostream>

namespace onnxruntime {
namespace contrib {
void EmbedLayerNormalizationShapeInference(::ONNX_NAMESPACE::InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 2, 0);
  auto mask_index_type = getAttribute(ctx, "mask_index_type", 1);
  if (mask_index_type > 0) {
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
  }
  if (!hasInputShape(ctx, 0)) {
    // TODO(kreeger): In this case update the output to (?, ?, hidden_size).
    return;
  }

  auto& input_ids_shape = getInputShape(ctx, 0);
  auto& input_ids_dims = input_ids_shape.dim();

  // Note that both batch size and sequence length could be symbolic.
  // So we only check dimension size here.
  if (input_ids_dims.size() != 2) {
    fail_shape_inference("input_ids shall be 2 dimensions");
  }

  bool has_segment = hasInputShape(ctx, 1);
  if (has_segment) {
    // Ensure that segment_ids has the same shape.
    auto& segment_ids_shape = getInputShape(ctx, 1);
    auto& segment_ids_dims = segment_ids_shape.dim();
    if (segment_ids_dims.size() != 2) {
      fail_shape_inference("segment_ids input shall be 2 dimensions");
    }
  }

  // get hidden_size from the last dimension of embedding
  auto& word_embedding_shape = getInputShape(ctx, 2);
  auto& word_embedding_dims = word_embedding_shape.dim();
  if (word_embedding_dims.size() != 2 ||
      !word_embedding_dims[1].has_dim_value() ||
      word_embedding_shape.dim(1).dim_value() <= 0) {
    fail_shape_inference("word_embedding should have 2 dimensions and dimension size is known.");
  }
  int64_t hidden_size = word_embedding_shape.dim(1).dim_value();

  // Ensure that all embeddings + the gamma/beta tensors have the same hidden_size:
  auto& position_embedding_shape = getInputShape(ctx, 3);
  auto& position_embedding_dims = position_embedding_shape.dim();
  if (position_embedding_dims.size() != 2 ||
      !position_embedding_dims[1].has_dim_value() ||
      position_embedding_shape.dim(1).dim_value() != hidden_size) {
    fail_shape_inference(
        "position_embedding should have 2 dimensions, dimension size known, "
        "and same hidden size as word_embedding.");
  }

  if (has_segment) {
    auto& segment_embedding_shape = getInputShape(ctx, 4);
    auto& segment_embedding_dims = segment_embedding_shape.dim();
    if (segment_embedding_dims.size() != 2 ||
        !segment_embedding_dims[1].has_dim_value() ||
        segment_embedding_shape.dim(1).dim_value() != hidden_size) {
      fail_shape_inference(
          "segment_embedding should have 2 dimensions, dimension size known, "
          "and same hidden size as word_embedding.");
    }
  }

  auto& gamma_shape = getInputShape(ctx, 5);
  auto& gamma_dims = gamma_shape.dim();
  if (gamma_dims.size() != 1 ||
      !gamma_dims[0].has_dim_value() ||
      gamma_shape.dim(0).dim_value() != hidden_size) {
    fail_shape_inference(
        "gamma should have 2 dimension, dimension size known, "
        "and same hidden size as word_embedding.");
  }

  auto& beta_shape = getInputShape(ctx, 6);
  auto& beta_dims = gamma_shape.dim();
  if (beta_dims.size() != 1 ||
      !beta_dims[0].has_dim_value() ||
      beta_shape.dim(0).dim_value() != hidden_size) {
    fail_shape_inference(
        "beta should have 1 dimension, dimension size known, "
        "and same hidden size as word_embedding.");
  }

  // input shape is (batch_size, sequence_length), output shape is (batch_size, sequence_length, hidden_size)
  ONNX_NAMESPACE::TensorShapeProto output_shape;
  *output_shape.add_dim() = input_ids_dims[0];
  *output_shape.add_dim() = input_ids_dims[1];

  output_shape.add_dim();
  output_shape.mutable_dim(2)->set_dim_value(hidden_size);

  updateOutputShape(ctx, 0, output_shape);

  // mask_index shape is (batch_size)
  if (mask_index_type > 0) {
    ONNX_NAMESPACE::TensorShapeProto mask_index_shape;
    *mask_index_shape.add_dim() = input_ids_dims[0];
    updateOutputShape(ctx, 1, mask_index_shape);
  }

  if (ctx.getNumOutputs() == 3 || (ctx.getNumOutputs() == 2 && mask_index_type == 0)) {
    updateOutputShape(ctx, 2, output_shape);
    propagateElemTypeFromInputToOutput(ctx, 0, 2);
  }
}

// Shape inference for Attention and QAttention
void AttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int past_input_index) {
  // Input 0, 1, 2 are input, weights and bias.
  // The other inputs may vary in Attention and QAttention. For example, past_input_index is 4 for Attention,
  // and 8 for QAttention.
  //
  // Input 0 has 3D shape (batch_size, sequence_length, input_hidden_size)
  // Input 1 has 2D shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)
  // Input 2 has 1D shape (hidden_size + hidden_size + v_hidden_size)
  // The past input has shape (2, batch_size, num_heads, past_sequence_length, head_size)
  // Output 0 has 3D shape (batch_size, sequence_length, v_hidden_size)
  // Output 1 (present) has 5D shape (2, batch_size, num_heads, total_sequence_length, head_size)
  // When past and present share buffer, they shape is same: (2, batch_size, num_heads, max_sequence_length, head_size)

  // Type inference
  if (hasInputShape(ctx, 2)) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 2, 0);
    if (ctx.getNumOutputs() > 1) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 2, 1);
    }
  } else {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (ctx.getNumOutputs() > 1) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 1);
    }
  }

  // Shape inference
  if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2)) {
    auto& input_shape = getInputShape(ctx, 0);
    auto& input_dims = input_shape.dim();
    if (input_dims.size() != 3) {
      fail_shape_inference("Inputs 0 shall be 3 dimensions");
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

    output_shape.mutable_dim(2)->set_dim_value(v_hidden_size);
    updateOutputShape(ctx, 0, output_shape);

    if (ctx.getNumOutputs() > 1) {  // has present output
      if (hasInputShape(ctx, past_input_index)) {
        auto& past_shape = getInputShape(ctx, past_input_index);
        auto& past_dims = past_shape.dim();
        if (past_dims.size() != 5) {
          fail_shape_inference("The past input shall be 5 dimensions");
        }

        auto past_present_share_buffer = getAttribute(ctx, "past_present_share_buffer", 0);
        if (past_present_share_buffer) {
          propagateElemTypeFromInputToOutput(ctx, past_input_index, 1);
        } else {
          if (input_dims[1].has_dim_value() && past_dims[3].has_dim_value()) {
            int64_t total_sequence_length = input_dims[1].dim_value() + past_shape.dim(3).dim_value();

            ONNX_NAMESPACE::TensorShapeProto present_shape;
            for (auto& dim : past_dims) {
              *present_shape.add_dim() = dim;
            }
            present_shape.mutable_dim(3)->set_dim_value(total_sequence_length);

            updateOutputShape(ctx, 1, present_shape);
          }
        }
      }
    }
  }
}

}  // namespace contrib
}  // namespace onnxruntime
