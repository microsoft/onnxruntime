// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/framework/kv_cache.h"
#include "core/common/safeint.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace onnxruntime {
namespace {

class LogicalMask {
 public:
  LogicalMask(const Tensor* mask, int64_t batch, int64_t heads, int64_t queries, int64_t end)
      : mask_(mask) {
    if (!mask_) return;
    ORT_ENFORCE(mask_->IsDataType<float>() || mask_->IsDataType<bool>(),
                "AttentionWithCache supports only float32 or bool mask");
    const auto& shape = mask_->Shape();
    ORT_ENFORCE(shape.NumDimensions() <= 4, "Mask must broadcast to B,Nq,Sq,total_computed_keys");
    const int64_t target[4]{batch, heads, queries, end};
    size_t stride = 1;
    for (int i = 3; i >= 0; --i) {
      const int j = i - (4 - static_cast<int>(shape.NumDimensions()));
      const int64_t dim = j >= 0 ? shape[j] : 1;
      ORT_ENFORCE(dim == 1 || dim == target[i], "Mask dimension ", i, " does not match logical attention");
      strides_[i] = dim == 1 ? 0 : stride;
      stride = SafeInt<size_t>(stride) * dim;
    }
    if (mask_->IsDataType<float>()) {
      for (float x : mask_->DataAsSpan<float>()) {
        ORT_ENFORCE(!std::isnan(x) && x != std::numeric_limits<float>::infinity(),
                    "NaN and positive infinity mask values are unsupported");
      }
    }
  }

  float Bias(int64_t b, int64_t h, int64_t q, int64_t p) const {
    if (!mask_) return 0;
    const size_t index = b * strides_[0] + h * strides_[1] + q * strides_[2] + p * strides_[3];
    if (mask_->IsDataType<bool>()) {
      return mask_->Data<bool>()[index] ? 0 : -std::numeric_limits<float>::infinity();
    }
    return mask_->Data<float>()[index];
  }

 private:
  const Tensor* mask_;
  size_t strides_[4]{};
};

void CheckBNSD(const Tensor& tensor, int64_t b, int64_t n, int64_t s, int64_t d, const char* name) {
  ORT_ENFORCE(tensor.IsDataType<float>() && tensor.Shape().NumDimensions() == 4 &&
                  tensor.Shape()[0] == b && tensor.Shape()[1] == n &&
                  tensor.Shape()[2] == s && tensor.Shape()[3] == d,
              name, " must be float32 BNSD matching cache geometry");
  for (float x : tensor.DataAsSpan<float>()) {
    ORT_ENFORCE(std::isfinite(x), name, " contains nonfinite payload (unsupported)");
  }
}

class AttentionWithCache final : public OpKernel {
 public:
  explicit AttentionWithCache(const OpKernelInfo& info) : OpKernel(info) {
    causal_ = info.GetAttrOrDefault<int64_t>("is_causal", 0);
    window_ = info.GetAttrOrDefault<int64_t>("sliding_window", 0);
    softcap_ = info.GetAttrOrDefault<float>("softcap", 0);
    scale_ = info.GetAttrOrDefault<float>("scale", 0);
    has_scale_ = info.node().GetAttributes().count("scale") != 0;
    q_heads_ = info.GetAttrOrDefault<int64_t>("q_num_heads", 0);
    kv_heads_ = info.GetAttrOrDefault<int64_t>("kv_num_heads", 0);
    const auto precision = info.GetAttrOrDefault<int64_t>("softmax_precision", ONNX_NAMESPACE::TensorProto::FLOAT);
    ORT_ENFORCE(precision == ONNX_NAMESPACE::TensorProto::FLOAT, "Only float32 softmax precision supported");
    ORT_ENFORCE((causal_ == 0 || causal_ == 1) && window_ >= 0 &&
                    (window_ == 0 || causal_ == 1) && softcap_ >= 0 &&
                    std::isfinite(softcap_) && std::isfinite(scale_) &&
                    q_heads_ >= 0 && kv_heads_ >= 0,
                "Unsupported AttentionWithCache attributes");
  }

  Status Compute(OpKernelContext* ctx) const override {
    const auto& input = ctx->RequiredInput<KVCacheValue>(3);
    ORT_ENFORCE(input.cache, "Uninitialized KV cache input");
    const auto& old = *input.cache;
    const auto& o = old.Options();
    const auto& q = ctx->RequiredInput<Tensor>(0);
    const auto& k = ctx->RequiredInput<Tensor>(1);
    const auto& v = ctx->RequiredInput<Tensor>(2);
    ORT_ENFORCE(q.Shape().NumDimensions() == 4, "Q must be BNSD");
    const int64_t heads = q.Shape()[1];
    const int64_t tokens = q.Shape()[2];
    ORT_ENFORCE(heads > 0 && heads % o.num_kv_heads == 0 && tokens > 0 &&
                    (q_heads_ == 0 || heads == q_heads_) &&
                    (kv_heads_ == 0 || o.num_kv_heads == kv_heads_),
                "Invalid query heads, KV heads, or append count");
    ORT_ENFORCE(window_ == o.sliding_window, "Cache and attention sliding windows differ");
    ORT_ENFORCE(window_ != 0 || old.Begin() == 0, "Full attention requires unevicted history");
    const int64_t required_begin = window_ ? std::max<int64_t>(0, old.End() - (window_ - 1)) : 0;
    ORT_ENFORCE(old.Begin() <= required_begin, "Required logical history was evicted");
    CheckBNSD(q, o.batch_size, heads, tokens, o.key_head_size, "Q");
    CheckBNSD(k, o.batch_size, o.num_kv_heads, tokens, o.key_head_size, "K_new");
    CheckBNSD(v, o.batch_size, o.num_kv_heads, tokens, o.value_head_size, "V_new");
    ORT_ENFORCE(tokens <= std::numeric_limits<int64_t>::max() - old.End(), "Computed length overflow");
    const int64_t end = old.End() + tokens;
    if (const Tensor* positions = ctx->Input<Tensor>(5)) {
      ORT_ENFORCE(positions->IsDataType<int64_t>() &&
                      positions->Shape() == TensorShape({o.batch_size, tokens}),
                  "positions must be int64[B,Snew]");
      const auto* p = positions->Data<int64_t>();
      for (int64_t b = 0; b < o.batch_size; ++b) {
        for (int64_t t = 0; t < tokens; ++t) {
          ORT_ENFORCE(p[b * tokens + t] == old.End() + t,
                      "Sparse, noncontiguous, or mismatched logical positions are unsupported");
        }
      }
    }
    LogicalMask mask(ctx->Input<Tensor>(4), o.batch_size, heads, tokens, end);
    const auto staged = old.Append(k.DataAsSpan<float>(), v.DataAsSpan<float>(), tokens);
    auto* y = ctx->RequiredOutput(0, TensorShape({o.batch_size, heads, tokens, o.value_head_size})).MutableData<float>();
    const double scale = has_scale_ ? scale_ : 1.0 / std::sqrt(static_cast<double>(o.key_head_size));
    const int64_t group = heads / o.num_kv_heads;
    for (int64_t b = 0; b < o.batch_size; ++b) {
      for (int64_t h = 0; h < heads; ++h) {
        const int64_t kv_head = b * o.num_kv_heads + h / group;
        for (int64_t t = 0; t < tokens; ++t) {
          const auto* query = q.Data<float>() + ((b * heads + h) * tokens + t) * o.key_head_size;
          InlinedVector<double> accumulator(static_cast<size_t>(o.value_head_size), 0);
          double maximum = -std::numeric_limits<double>::infinity();
          double normalizer = 0;
          const int64_t query_position = old.End() + t;
          staged->Visit([&](const KVCacheBlock& block) {
            for (int64_t j = 0; j < block.count; ++j) {
              const int64_t p = block.position + j;
              if ((causal_ && p > query_position) ||
                  (window_ && p < query_position - (window_ - 1))) continue;
              const float bias = mask.Bias(b, h, t, p);
              if (bias == -std::numeric_limits<float>::infinity()) continue;
              const auto* key = block.keys + (kv_head * block.head_stride + j) * o.key_head_size;
              const auto* value = block.values + (kv_head * block.head_stride + j) * o.value_head_size;
              double score = 0;
              for (int64_t d = 0; d < o.key_head_size; ++d) score += static_cast<double>(query[d]) * key[d];
              score *= scale;
              if (softcap_ > 0) score = softcap_ * std::tanh(score / softcap_);
              score += bias;
              ORT_ENFORCE(std::isfinite(score), "Nonfinite attention score");
              const double next_maximum = std::max(maximum, score);
              const double correction = std::exp(maximum - next_maximum);
              const double weight = std::exp(score - next_maximum);
              normalizer = normalizer * correction + weight;
              for (int64_t d = 0; d < o.value_head_size; ++d) {
                accumulator[static_cast<size_t>(d)] =
                    accumulator[static_cast<size_t>(d)] * correction + weight * value[d];
              }
              maximum = next_maximum;
            }
          });
          auto* row = y + ((b * heads + h) * tokens + t) * o.value_head_size;
          for (int64_t d = 0; d < o.value_head_size; ++d) {
            row[d] = normalizer == 0 ? 0 : static_cast<float>(accumulator[static_cast<size_t>(d)] / normalizer);
          }
        }
      }
    }
    auto next = staged->Seal();
    ctx->Output<KVCacheValue>(1)->cache = std::move(next);
    return Status::OK();
  }

 private:
  int64_t causal_, window_, q_heads_, kv_heads_;
  float softcap_, scale_;
  bool has_scale_;
};

class KVCacheCreate final : public OpKernel {
 public:
  explicit KVCacheCreate(const OpKernelInfo& info)
      : OpKernel(info),
        options_{info.GetAttrOrDefault<int64_t>("batch_size", 0),
                 info.GetAttrOrDefault<int64_t>("num_kv_heads", 0),
                 info.GetAttrOrDefault<int64_t>("key_head_size", 0),
                 info.GetAttrOrDefault<int64_t>("value_head_size", 0)} {
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("elem_type", 0) == ONNX_NAMESPACE::TensorProto::FLOAT,
                "KVCacheCreate CPU requires float32");
    options_.sliding_window = info.GetAttrOrDefault<int64_t>("sliding_window", 0);
    options_.storage = ParseKVCacheStorage(info.GetConfigOptions().GetConfigOrDefault("session.kv_cache.backend",
                                                                                      "contiguous"));
    // Geometry and backend compatibility are validated before session execution.
    CreateKVCache(options_);
  }
  Status Compute(OpKernelContext* ctx) const override {
    ctx->Output<KVCacheValue>(0)->cache = CreateKVCache(options_);
    return Status::OK();
  }

 private:
  KVCacheOptions options_;
};

class KVCacheDescribe final : public OpKernel {
 public:
  explicit KVCacheDescribe(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override {
    const auto& payload = ctx->RequiredInput<KVCacheValue>(0);
    ORT_ENFORCE(payload.cache, "Uninitialized KV cache");
    const auto& cache = *payload.cache;
    const int64_t batch = cache.Options().batch_size;
    auto* lengths = ctx->RequiredOutput(0, TensorShape({batch})).MutableData<int64_t>();
    auto* range = ctx->RequiredOutput(1, TensorShape({batch, 2})).MutableData<int64_t>();
    for (int64_t b = 0; b < batch; ++b) {
      lengths[b] = cache.End();
      range[2 * b] = cache.Begin();
      range[2 * b + 1] = cache.End();
    }
    return Status::OK();
  }
};

class KVCacheExport final : public OpKernel {
 public:
  explicit KVCacheExport(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override {
    const auto& payload = ctx->RequiredInput<KVCacheValue>(0);
    ORT_ENFORCE(payload.cache, "Uninitialized KV cache");
    const auto& cache = *payload.cache;
    const auto& o = cache.Options();
    const int64_t length = cache.End() - cache.Begin();
    auto* keys = ctx->RequiredOutput(0, TensorShape({o.batch_size, o.num_kv_heads, length, o.key_head_size}))
                     .MutableData<float>();
    auto* values = ctx->RequiredOutput(1, TensorShape({o.batch_size, o.num_kv_heads, length, o.value_head_size}))
                       .MutableData<float>();
    auto* positions = ctx->RequiredOutput(2, TensorShape({o.batch_size, length})).MutableData<int64_t>();
    cache.Visit([&](const KVCacheBlock& block) {
      for (int64_t h = 0; h < o.batch_size * o.num_kv_heads; ++h) {
        std::copy_n(block.keys + h * block.head_stride * o.key_head_size, block.count * o.key_head_size,
                    keys + (h * length + block.position - cache.Begin()) * o.key_head_size);
        std::copy_n(block.values + h * block.head_stride * o.value_head_size, block.count * o.value_head_size,
                    values + (h * length + block.position - cache.Begin()) * o.value_head_size);
      }
    });
    for (int64_t b = 0; b < o.batch_size; ++b) {
      for (int64_t p = 0; p < length; ++p) positions[b * length + p] = cache.Begin() + p;
    }
    return Status::OK();
  }
};

}  // namespace

ONNX_CPU_OPERATOR_KERNEL(
    AttentionWithCache, 29,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TC", DataTypeImpl::GetType<KVCacheValue>())
        .TypeConstraint("U", BuildKernelDefConstraints<bool, float>()),
    AttentionWithCache);

ONNX_CPU_OPERATOR_KERNEL(
    KVCacheCreate, 29,
    KernelDefBuilder().TypeConstraint("TC", DataTypeImpl::GetType<KVCacheValue>()),
    KVCacheCreate);

ONNX_CPU_OPERATOR_KERNEL(
    KVCacheDescribe, 29,
    KernelDefBuilder().TypeConstraint("TC", DataTypeImpl::GetType<KVCacheValue>()),
    KVCacheDescribe);

ONNX_CPU_OPERATOR_KERNEL(
    KVCacheExport, 29,
    KernelDefBuilder().TypeConstraint("TC", DataTypeImpl::GetType<KVCacheValue>()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    KVCacheExport);

}  // namespace onnxruntime
