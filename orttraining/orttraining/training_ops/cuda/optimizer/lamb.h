// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD_NORM, typename T_MIXED_PRECISION_FP>
class LambOptimizer final : public CudaKernel {
 public:
  LambOptimizer(const OpKernelInfo& info) : CudaKernel(info) {
    alpha_ = info.GetAttrsOrDefault("alpha", std::vector<float>(1024, 0.9f));
    beta_ = info.GetAttrsOrDefault("beta", std::vector<float>(1024, 0.999f));
    lambda_ = info.GetAttrsOrDefault("lambda", std::vector<float>(1024, 0.0f));
    epsilon_ = info.GetAttrsOrDefault("epsilon", std::vector<float>(1024, 1e-6f));
    max_norm_clip_ = info.GetAttrsOrDefault("max_norm_clip", std::vector<float>(1024, 1.0f));
    ORT_ENFORCE(info.GetAttr<float>("ratio_min", &ratio_min_).IsOK(), "Missing/Invalid 'ratio_min' attribute value");
    ORT_ENFORCE(info.GetAttr<float>("ratio_max", &ratio_max_).IsOK(), "Missing/Invalid 'ratio_max' attribute value");
    for (const auto& max_norm : max_norm_clip_) {
      ORT_ENFORCE(max_norm != 0, "max_norm_clip must NOT be 0.");
    }

    int64_t tmp_flag = static_cast<int64_t>(0);
    ORT_ENFORCE(info.GetAttr<int64_t>("do_bias_correction", &tmp_flag).IsOK(), "Missing/Invalid do_bias_correction");
    ORT_ENFORCE(tmp_flag == 0 || tmp_flag == 1, "do_bias_correction must be either 0 or 1.");
    do_bias_correction_ = tmp_flag != 0 ? true : false;
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
  std::vector<float> lambda_;
  std::vector<float> epsilon_;
  std::vector<float> max_norm_clip_;
  float ratio_min_;
  float ratio_max_;
  bool do_bias_correction_;
};

}  // namespace cuda
}  // namespace onnxruntime
