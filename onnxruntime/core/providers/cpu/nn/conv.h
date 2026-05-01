// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

template <typename T>
class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;
};

template <>
class Conv<float> : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info), channels_last_(info.GetKernelDef().OpName() == "NhwcFusedConv") {
    activation_.ActivationKind = MlasIdentityActivation;
    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());

#if defined(USE_KLEIDIAI) && defined(__aarch64__) && defined(__linux__)
    if (channels_last_) {
      const auto& input_defs = info.node().InputDefs();
      const bool has_bias_input = input_defs.size() >= 3 && input_defs[2] != nullptr;
      info.TryGetConstantInput(1, &constant_filter_tensor_);
      if (has_bias_input) {
        info.TryGetConstantInput(2, &constant_bias_tensor_);
      }

      can_cache_packed_filter_ =
          constant_filter_tensor_ != nullptr && (!has_bias_input || constant_bias_tensor_ != nullptr);
    }
#endif
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  MLAS_ACTIVATION activation_;

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;

  ConvAttributes conv_attrs_;
  bool channels_last_{false};

#if defined(USE_KLEIDIAI) && defined(__aarch64__) && defined(__linux__)
 private:
  Status EnsurePackedChannelsLastFilter(concurrency::ThreadPool* thread_pool,
                                        size_t filter_count_per_group,
                                        size_t input_channels_per_group,
                                        const TensorShapeVector& kernel_shape,
                                        const TensorShapeVector& dilations) const;

  const Tensor* constant_filter_tensor_{nullptr};
  const Tensor* constant_bias_tensor_{nullptr};
  bool can_cache_packed_filter_{false};
  mutable std::once_flag packed_filter_once_;
  mutable Status packed_filter_status_;
  mutable IAllocatorUniquePtr<void> packed_filter_;
  mutable size_t packed_filter_group_stride_{0};
#endif
};

}  // namespace onnxruntime
