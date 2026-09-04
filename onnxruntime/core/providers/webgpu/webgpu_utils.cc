// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/webgpu_utils.h"

#include <string>
#include "core/platform/env_var.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Test/measurement override for Split-K gating, read from `ORT_WEBGPU_SPLIT_K`. `on` and `off` let a
// single binary produce both arms of an A/B comparison, and let tests exercise the Split-K path on
// adapters whose vendor table leaves it disabled.
enum class SplitKOverride {
  Default,
  ForceOff,
  ForceOn,
};

SplitKOverride ReadSplitKOverride() {
  const std::string value = onnxruntime::detail::GetEnvironmentVar("ORT_WEBGPU_SPLIT_K");
  if (value == "off" || value == "0") {
    return SplitKOverride::ForceOff;
  }
  if (value == "on" || value == "1") {
    return SplitKOverride::ForceOn;
  }
  return SplitKOverride::Default;
}

}  // namespace

TensorShape ReduceShapeByComponents(const TensorShape& shape, int64_t components) {
  // Reduce the last dimensions by components creating a new tensor shape.
  TensorShapeVector shape_vector = shape.AsShapeVector();
  ORT_ENFORCE(!shape_vector.empty(), "The input shape must not be empty.");
  size_t reduce_index = shape_vector.size() - 1;
  // Find the last dimension that is divisible by components.
  while (shape_vector[reduce_index] % components != 0 && reduce_index > 0) {
    ORT_ENFORCE(components % shape_vector[reduce_index] == 0, "The components must divide dims");
    components /= shape_vector[reduce_index];
    shape_vector[reduce_index] = 1;
    reduce_index--;
  }
  ORT_ENFORCE(shape_vector[reduce_index] % components == 0, "The last non-unit dimension of the input shape must be divisible by the number of components.");
  shape_vector[reduce_index] /= components;
  return TensorShape(shape_vector);
}

SplitKConfig::SplitKConfig(const wgpu::AdapterInfo& adapter_info) {
  const SplitKOverride override_mode = ReadSplitKOverride();
  if (override_mode == SplitKOverride::ForceOff) {
    // `enable_split_k_` defaults to false, and `UseSplitK` short-circuits on it.
    return;
  }

  if (adapter_info.vendor == std::string_view{"intel"}) {
    // Disable Split-K on old Intel GPUs.
    if (adapter_info.architecture == std::string_view{"gen-7"} ||
        adapter_info.architecture == std::string_view{"gen-8"} ||
        adapter_info.architecture == std::string_view{"gen-9"} ||
        adapter_info.architecture == std::string_view{"gen-11"}) {
      enable_split_k_ = false;
    } else if (adapter_info.architecture == std::string_view{"xe-2lpg"} ||
               adapter_info.architecture == std::string_view{"xe-2hpg"} ||
               adapter_info.architecture == std::string_view{"gen-12hp"}) {
      // Below thresholds are only verified on Intel discrete GPUs and Lunar Lake iGPUs.
      enable_split_k_ = true;

      max_batch_size_ = 8;
      split_dim_inner_ = 256;
      min_dim_inner_with_split_k_ = split_dim_inner_ * 2;

      configs_per_dim_inner_range_.emplace_back(768, 52.0);
      configs_per_dim_inner_range_.emplace_back(2304, 35.0);
      configs_per_dim_inner_range_.emplace_back(3072, 21.5);
      configs_per_dim_inner_range_.emplace_back(4096, 16.0);
    } else if (adapter_info.architecture == std::string_view{"xe-3lpg"}) {
      // Below thresholds are only verified on Intel Panther Lake iGPUs (12Xe).
      enable_split_k_ = true;

      max_batch_size_ = 8;
      split_dim_inner_ = 256;
      min_dim_inner_with_split_k_ = split_dim_inner_ * 2;

      configs_per_dim_inner_range_.emplace_back(768, 40.0);
      configs_per_dim_inner_range_.emplace_back(1792, 22.0);
      configs_per_dim_inner_range_.emplace_back(3072, 18.0);
      configs_per_dim_inner_range_.emplace_back(4096, 10.0);
    } else {
      // Below are the default thresholds on newer Intel GPUs. These values are chosen on
      // Intel "gen-12lp" GPU with 32EUs.
      enable_split_k_ = true;

      max_batch_size_ = 8;
      split_dim_inner_ = 256;
      min_dim_inner_with_split_k_ = split_dim_inner_ * 2;

      configs_per_dim_inner_range_.emplace_back(768, 20.0);
      configs_per_dim_inner_range_.emplace_back(1792, 13.0);
      configs_per_dim_inner_range_.emplace_back(3072, 8.0);
      configs_per_dim_inner_range_.emplace_back(4096, 6.0);
    }
  }

  if (override_mode == SplitKOverride::ForceOn && !enable_split_k_) {
    // The override must produce a usable config on an adapter the vendor table left disabled, so it
    // carries its own thresholds.
    enable_split_k_ = true;

    max_batch_size_ = 8;
    split_dim_inner_ = 256;
    min_dim_inner_with_split_k_ = split_dim_inner_ * 2;

    configs_per_dim_inner_range_.emplace_back(768, 52.0);
    configs_per_dim_inner_range_.emplace_back(2304, 35.0);
    configs_per_dim_inner_range_.emplace_back(3072, 21.5);
    configs_per_dim_inner_range_.emplace_back(4096, 16.0);
  }
}

SplitKConfig::ConfigAtRange::ConfigAtRange(uint32_t max_dim_inner, double rate)
    : max_dim_inner_with_rate(max_dim_inner), max_dim_a_outer_x_dim_b_outer_x_batch_size_divides_dim_inner(rate) {}

uint32_t SplitKConfig::GetMaxDimInnerWithSplitK() const {
  assert(!configs_per_dim_inner_range_.empty());
  return configs_per_dim_inner_range_.back().max_dim_inner_with_rate;
}

bool SplitKConfig::UseSplitK(
    bool is_vec4,
    ActivationKind activation_kind,
    uint64_t batch_size,
    uint32_t dim_a_outer,
    uint32_t dim_b_outer,
    uint32_t dim_inner,
    bool is_channels_last) const {
  if (!enable_split_k_) {
    return false;
  }

  bool use_split_k = true;

  // TODO: support the cases below.
  use_split_k &= activation_kind == ActivationKind::None;
  use_split_k &= is_vec4;

  // Larger batches increase parallelism on their own, so we temporarily set a batch size threshold
  // for using Split-K.
  use_split_k &= batch_size <= max_batch_size_;

  // `is_channels_last` should only affect Split-K gating when bias is applied in the non-GEMM
  // MatMul/Conv|MatMul path. For GEMM and for MatMul or Conv|MatMul without bias, we need to
  // use `true` as `is_channels_last` to make `UseSplitK` ignore `is_channels_last`.
  // When `is_channels_last` has a valid value here, it is required to be true because we only
  // generate `vec4` shaders in `MatMulSplitKReduceProgram`, which is where bias is applied.
  use_split_k &= is_channels_last;

  // Split-K works best when `dim_inner` is relatively large compared with `dim_a_outer` and
  // `dim_b_outer`. Currently we use the factor between `(dim_a_outer * dim_b_outer * batch_size)`
  // and `dim_inner` as the metric to decide whether to use Split-K or not.
  use_split_k &= dim_inner >= min_dim_inner_with_split_k_;
  use_split_k &= dim_inner <= GetMaxDimInnerWithSplitK();

  if (!use_split_k) {
    return false;
  }

  const double rate = static_cast<double>(dim_a_outer) * static_cast<double>(dim_b_outer) * static_cast<double>(batch_size) / static_cast<double>(dim_inner);
  for (const auto& config_at_range : configs_per_dim_inner_range_) {
    if (dim_inner <= config_at_range.max_dim_inner_with_rate) {
      return rate <= config_at_range.max_dim_a_outer_x_dim_b_outer_x_batch_size_divides_dim_inner;
    }
  }
  return false;
}

uint32_t SplitKConfig::GetSplitDimInner() const {
  return split_dim_inner_;
}

}  // namespace webgpu
}  // namespace onnxruntime
