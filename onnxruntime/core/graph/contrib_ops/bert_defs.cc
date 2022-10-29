// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/quantization_defs.h"
#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/graph/contrib_ops/shape_inference_functions.h"

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

When weights is not provided, key and value are required. In this situation, MatMul for input projection is excluded,
and input is the query after projection. The bias is included for performance consideration.

The qkv_hidden_sizes is required only when K and V have different hidden sizes.

When there is past state, hidden dimension for Q, K and V shall be the same.

The total_sequence_length is past_sequence_length + kv_sequence_length. Here kv_sequence_length is the length of K or V.
For self attention, kv_sequence_length equals to sequence_length (sequence length of Q).
For cross attention, query and key might have different lengths.
)DOC";

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
        .Input(0,
               "input",
               "Input tensor with shape (batch_size, sequence_length, input_hidden_size) when weights is available, "
               "or query tensor with shape (batch_size, sequence_length, hidden_size) when weights is not available.",
               "T",
               OpSchema::Optional)
        .Input(1,
               "weights",
               "Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
               "T",
               OpSchema::Optional)
        .Input(2,
               "bias",
               "Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) for input projection",
               "T")
        .Input(3,
               "mask_index",
               "Attention mask with shape (batch_size, 1, max_sequence_length, max_sequence_length), "
               "(batch_size, total_sequence_length) or (batch_size, sequence_length, total_sequence_length), "
               "or index with shape (batch_size) or (2 * batch_size).",
               "M",
               OpSchema::Optional)
        .Input(4,
               "past",
               "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size)",
               "T",
               OpSchema::Optional)
        .Input(5,
               "extra_add",
               "additional add to QxK' with shape (batch_size, num_heads, sequence_length, total_sequence_length)",
               "T",
               OpSchema::Optional)
        .Input(6,
               "key",
               "Input for key with shape (batch_size, kv_sequence_length, hidden_size). "
               "Required when weights is not available.",
               "T",
               OpSchema::Optional)
        .Input(7,
               "value",
               "Input for key with shape (batch_size, kv_sequence_length, v_hidden_size). "
               "Required when weights is not available.",
               "T",
               OpSchema::Optional)
        .Output(0,
                "output",
                "3D output tensor with shape (batch_size, sequence_length, v_hidden_size)",
                "T")
        .Output(1,
                "present",
                "past state for key and value with shape (2, batch_size, num_heads, total_sequence_length, head_size)",
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

constexpr const char* Longformer_Attention_doc = R"DOC(
Longformer Self Attention with a local context and a global context. Tokens attend locally: Each token
attends to its W previous tokens and W succeeding tokens with W being the window length. A selected few tokens
attend globally to all other tokens.

The attention mask is of shape (batch_size, sequence_length), where sequence_length is a multiple of 2W after padding.
Mask value < 0 (like -10000.0) means the token is masked, 0 otherwise.

Global attention flags have value 1 for the tokens attend globally and 0 otherwise.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(LongformerAttention, 1,
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

ONNX_MS_OPERATOR_SET_SCHEMA(DecoderAttention, 1,
                            OpSchema()
                                .SetDoc(Decoder_Attention_doc)
                                .Attr("num_heads", "Number of attention heads", AttributeProto::INT)
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

ONNX_MS_OPERATOR_SET_SCHEMA(EmbedLayerNormalization, 1,
                            OpSchema()
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
                                .Input(8, "position_ids", "2D position ids with shape (batch_size, sequence_length)", "T1", OpSchema::Optional)
                                .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
                                .Output(1, "mask_index", "1D mask_index tensor with shape (batch_size)", "T1")
                                .Output(2, "embedding_sum", "sum of word_embedding and position_embedding without layer normalization", "T", OpSchema::Optional)
                                .TypeConstraint("T1", {"tensor(int32)"}, "Constrain input and output integer tensors types")
                                .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output float tensors types.")
                                .TypeAndShapeInferenceFunction(EmbedLayerNormalizationShapeInference));

constexpr const char* FastGelu_ver1_doc = R"DOC(
GELU (Gaussian Error Linear Unit) approximation: Y=0.5*X*(1+tanh(0.797885*X+0.035677*X*X*X)) with an optional input of bias that will be added to X before GELU.)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(FastGelu, 1,
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

ONNX_MS_OPERATOR_SET_SCHEMA(SkipLayerNormalization, 1,
                            OpSchema()
                                .SetDoc("Skip and Layer Normalization Fusion")
                                .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon)
                                .Input(0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
                                .Input(1, "skip", "3D skip tensor with shape (batch_size, sequence_length, hidden_size)", "T")
                                .Input(2, "gamma", "1D input tensor with shape (hidden_size)", "T")
                                .Input(3, "beta", "1D skip tensor with shape (hidden_size", "T", OpSchema::Optional)
                                .Input(4, "bias", "1D bias tensor with shape (hidden_size", "T", OpSchema::Optional)
                                .Output(0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "T")
                                .Output(1, "mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
                                .Output(2, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation.", "U", OpSchema::Optional)
                                .TypeConstraint("T", {"tensor(float)", "tensor(float16)"}, "Constrain input and output types to float or half tensors.")
                                .TypeConstraint("U", {"tensor(float)"}, "Constrain mean and inv_std_var to float tensors.")
                                .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* NGramRepeatBlock_ver1_doc = R"DOC(
Enforce no repetition of n-grams. Scores are set to `-inf` for tokens that form a repeated n-gram if added to the back of the input_ids.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(NGramRepeatBlock, 1,
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

ONNX_MS_OPERATOR_SET_SCHEMA(BifurcationDetector, 1,
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

ONNX_MS_OPERATOR_SET_SCHEMA(GemmFastGelu, 1,
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

}  // namespace contrib
}  // namespace onnxruntime
