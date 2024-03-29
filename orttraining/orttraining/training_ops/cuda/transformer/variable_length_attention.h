// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include "core/framework/random_generator.h"
// #include "orttraining/training_ops/cpu/triton/triton_op.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class MultiHeadAttentionVarLength final : public CudaKernel {
 public:
  MultiHeadAttentionVarLength(const OpKernelInfo& info) : CudaKernel(info) {
    // ORT_ENFORCE(info.GetAttr<int64_t>("training_mode", &training_mode_).IsOK(),
    //             "Missing 'training_mode' attribute value");

    // ORT_ENFORCE(info.GetAttr<std::string>("pre_scale", &pre_scale_).IsOK(),
    //             "Missing 'pre_scale' attribute value");
    // ORT_ENFORCE(info.GetAttr<int64_t>("mid_scale", &mid_scale_).IsOK(),
    //             "Missing 'mid_scale' attribute value");
    // ORT_ENFORCE(info.GetAttr<int64_t>("post_scale", &post_scale_).IsOK(),
    //             "Missing 'post_scale' attribute value");

    // ONNX_NAMESPACE::GraphProto proto;
    // ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("pre_scale", &proto).IsOK());
    // ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("mid_scale", &proto).IsOK());
    // ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("post_scale", &proto).IsOK());
    // ORT_IGNORE_RETURN_VALUE(proto);
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // int64_t training_mode_;
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
  // std::string pre_scale_;
  // std::string mid_scale_;
  // std::string post_scale_;
};

}  // namespace cuda
}  // namespace onnxruntime
