// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

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

SplitKConfig SplitKConfig::GetSplitKConfig(const wgpu::AdapterInfo& adapter_info) {
  SplitKConfig config = {};

  if (adapter_info.vendor == std::string_view{"intel"}) {
    if (adapter_info.architecture == std::string_view{"xe-2lpg"} ||
        adapter_info.architecture == std::string_view{"xe-2hpg"} ||
        adapter_info.architecture == std::string_view{"xe-lpg"} ||
        adapter_info.architecture == std::string_view{"gen-12hp"}) {
      config.enable_split_k_ = true;

      // Below thresholds are only verified on the above Intel GPUs without any regressions. The
      // proper value of `max_dim_a_outer_multiplies_dim_b_outer_divides_dim_inner_` may be
      // reduced when we support a larger `dim_inner` because larger `dim_inner` will bring more
      // atomic calls for each output value.
      config.split_dim_inner_ = 256;
      config.min_dim_inner_with_split_k_ = config.split_dim_inner_ * 2;
      config.max_dim_inner_with_split_k_ = config.split_dim_inner_ * 9;
      config.max_dim_a_outer_multiplies_dim_b_outer_divides_dim_inner_ = 35.0f;
    }
  }
  return config;
}

bool SplitKConfig::UseSplitK(
    bool is_vec4,
    ActivationKind activation_kind,
    uint64_t batch_size,
    bool is_channels_last,
    uint32_t dim_a_outer,
    uint32_t dim_b_outer,
    uint32_t dim_inner) const {
  if (!enable_split_k_) {
    return false;
  }

  bool use_split_k = true;

  // TODO: support the cases below.
  use_split_k &= activation_kind == ActivationKind::None;
  use_split_k &= is_vec4;
  use_split_k &= batch_size == 1;
  // Now `is_channels_last` is only supported because we only generate vec4 shaders in
  // `MatMulFillBiasOrZeroBeforeSplitKProgram`.
  use_split_k &= is_channels_last;

  // Split-K works best when `dim_inner` is relatively large compared with `dim_a_outer` and
  // `dim_b_outer`. Currently we use the factor between `(dim_a_outer * dim_b_outer)` and
  // `dim_inner)` as the metric to decide whether to use Split-K or not.
  use_split_k &= (dim_inner >= min_dim_inner_with_split_k_);
  use_split_k &= (dim_inner <= max_dim_inner_with_split_k_);
  use_split_k &= ((dim_a_outer * dim_b_outer * 1.0f / dim_inner) <= max_dim_a_outer_multiplies_dim_b_outer_divides_dim_inner_);

  return use_split_k;
}

uint32_t SplitKConfig::GetSplitDimInner() const {
  return split_dim_inner_;
}

}  // namespace webgpu
}  // namespace onnxruntime
