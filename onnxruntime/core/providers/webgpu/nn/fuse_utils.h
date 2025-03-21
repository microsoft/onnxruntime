// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <string>
#include "core/providers/webgpu/webgpu_kernel.h"

#pragma once
namespace onnxruntime {
namespace webgpu {
    enum class ActivationKind{
        None,
        Relu,
        Sigmoid,
        Clip,
        HardSigmoid,
        LeakyRelu,
        Tanh
      } ;

      typedef struct Activation {
        ActivationKind activation_kind = ActivationKind::None;
        typedef union Parameter {
          struct {
            float alpha;
          } LeakyRelu;
          struct {
            float minimum_;
            float maximum_;
          } Clip;
          struct {
            float alpha_;
            float beta_;
          } HardSigmoid;
          float values_[2];
        } ActivationParameters;
        ActivationParameters activation_params_ = {};
      } Activation;

    Status GetFusedActivationAttr(const OpKernelInfo& info, Activation& activation);
    std::string GetActivationSnippet(const Activation &activation, std::string value_type, std::string base_type);

}  // namespace webgpu
}  // namespace onnxruntime
