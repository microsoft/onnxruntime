// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <string>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

namespace {

enum class DeepSeekV4AttentionMode {
  kSliding,
  kCsa,
  kHca,
};

constexpr int kFirstHcaInputIndex = 17;
constexpr int kLastHcaInputIndex = 23;
constexpr int kFirstCsaInputIndex = 24;
constexpr int kLastCsaInputIndex = 43;

inline bool HasInput(const OpKernelContext* context, int index) {
  return index < context->InputCount() && context->Input<Tensor>(index) != nullptr;
}

inline bool HasAnyInputInRange(const OpKernelContext* context, int begin, int end) {
  for (int i = begin; i <= end; ++i) {
    if (HasInput(context, i)) {
      return true;
    }
  }
  return false;
}

DeepSeekV4AttentionMode ParseAttentionMode(const std::string& mode) {
  if (mode == "sliding") {
    return DeepSeekV4AttentionMode::kSliding;
  }
  if (mode == "csa") {
    return DeepSeekV4AttentionMode::kCsa;
  }
  if (mode == "hca") {
    return DeepSeekV4AttentionMode::kHca;
  }
  ORT_THROW("DeepSeekV4Attention: unsupported attention_mode='", mode, "'.");
}

class DeepSeekV4Attention final : public OpKernel {
 public:
  explicit DeepSeekV4Attention(const OpKernelInfo& info) : OpKernel(info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0,
                "DeepSeekV4Attention: num_heads must be > 0.");

    int64_t head_size = 0;
    ORT_ENFORCE(info.GetAttr("head_size", &head_size).IsOK() && head_size > 0,
                "DeepSeekV4Attention: head_size must be > 0.");

    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads == 1,
                "DeepSeekV4Attention: kv_num_heads must be 1.");

    int64_t q_lora_rank = 0;
    ORT_ENFORCE(info.GetAttr("q_lora_rank", &q_lora_rank).IsOK() && q_lora_rank > 0,
                "DeepSeekV4Attention: q_lora_rank must be > 0.");

    int64_t o_groups = 0;
    ORT_ENFORCE(info.GetAttr("o_groups", &o_groups).IsOK() && o_groups > 0,
                "DeepSeekV4Attention: o_groups must be > 0.");

    int64_t o_lora_rank = 0;
    ORT_ENFORCE(info.GetAttr("o_lora_rank", &o_lora_rank).IsOK() && o_lora_rank > 0,
                "DeepSeekV4Attention: o_lora_rank must be > 0.");

    int64_t rotary_dim = 0;
    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim).IsOK() && rotary_dim > 0 && rotary_dim <= head_size,
                "DeepSeekV4Attention: rotary_dim must be in (0, head_size].");

    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1,
                "DeepSeekV4Attention: rotary_interleaved must be 1.");
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("rotary_trailing", 0) == 1,
                "DeepSeekV4Attention: rotary_trailing must be 1.");
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("do_output_derotate", 0) == 1,
                "DeepSeekV4Attention: do_output_derotate must be 1.");

    int64_t local_window_size = 0;
    ORT_ENFORCE(info.GetAttr("local_window_size", &local_window_size).IsOK() && local_window_size > 0,
                "DeepSeekV4Attention: local_window_size must be > 0.");

    float rms_norm_epsilon = 0.0f;
    ORT_ENFORCE(info.GetAttr("rms_norm_epsilon", &rms_norm_epsilon).IsOK() && rms_norm_epsilon > 0.0f,
                "DeepSeekV4Attention: rms_norm_epsilon must be > 0.");

    static_cast<void>(info.GetAttrOrDefault<float>("scale", 1.0f / std::sqrt(static_cast<float>(head_size))));

    const std::string attention_mode = info.GetAttrOrDefault<std::string>("attention_mode", "sliding");
    attention_mode_ = ParseAttentionMode(attention_mode);

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa ||
        attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      float compress_rate = 0.0f;
      ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate).IsOK() && compress_rate > 0.0f,
                  "DeepSeekV4Attention: compress_rate must be provided and > 0 for csa/hca mode.");
    }

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      int64_t index_topk = 0;
      int64_t index_num_heads = 0;
      int64_t index_head_dim = 0;
      ORT_ENFORCE(info.GetAttr("index_topk", &index_topk).IsOK() && index_topk > 0,
                  "DeepSeekV4Attention: index_topk must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_num_heads", &index_num_heads).IsOK() && index_num_heads > 0,
                  "DeepSeekV4Attention: index_num_heads must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_head_dim", &index_head_dim).IsOK() && index_head_dim > 0,
                  "DeepSeekV4Attention: index_head_dim must be provided and > 0 for csa mode.");
    }
  }

  Status Compute(OpKernelContext* context) const override {
    const bool has_hca_inputs = HasAnyInputInRange(context, kFirstHcaInputIndex, kLastHcaInputIndex);
    const bool has_csa_inputs = HasAnyInputInRange(context, kFirstCsaInputIndex, kLastCsaInputIndex);

    if (attention_mode_ == DeepSeekV4AttentionMode::kSliding) {
      ORT_RETURN_IF(has_hca_inputs || has_csa_inputs,
                    "DeepSeekV4Attention(sliding): compressor inputs are not allowed.");
    } else if (attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      ORT_RETURN_IF(!has_hca_inputs, "DeepSeekV4Attention(hca): HCA inputs are required.");
      ORT_RETURN_IF(has_csa_inputs, "DeepSeekV4Attention(hca): CSA inputs are not allowed.");
    } else {
      ORT_RETURN_IF(!has_csa_inputs, "DeepSeekV4Attention(csa): CSA inputs are required.");
      ORT_RETURN_IF(has_hca_inputs, "DeepSeekV4Attention(csa): HCA inputs are not allowed.");
    }

    return ORT_MAKE_STATUS(
        ONNXRUNTIME, NOT_IMPLEMENTED,
        "DeepSeekV4Attention CPU kernel is a frontend contract placeholder and is not implemented yet.");
  }

 private:
  DeepSeekV4AttentionMode attention_mode_;
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    DeepSeekV4Attention,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("P", DataTypeImpl::GetTensorType<int64_t>()),
    DeepSeekV4Attention);

}  // namespace contrib
}  // namespace onnxruntime
