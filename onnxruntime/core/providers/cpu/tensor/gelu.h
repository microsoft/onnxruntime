// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

template <typename T>
class Gelu final : public OpKernel {
 public:
  explicit Gelu(const OpKernelInfo& info) : OpKernel(info) {
    approximation_algorithm_ = info.GetAttrOrDefault<std::string>("approximate", "none");
    use_gelu_erf_minimax_approximation_ =
        info.GetConfigOptions().GetConfigOrDefault(kOrtSessionOptionsMlasGeluErfUseMinimaxApproximation, "0") == "1";
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
  std::string approximation_algorithm_;
  bool use_gelu_erf_minimax_approximation_ = false;
};

}  // namespace onnxruntime
