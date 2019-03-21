// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/training/gradient_builder_base.h"

namespace onnxruntime {
namespace training {
// TODO: maybe group the gradient builders and split them into different files.
#define DECLARE_GRADIENT_BUILDER(name)                     \
  class name : public GradientBuilderBase {                \
    using GradientBuilderBase::GradientBuilderBase;        \
    std::vector<NodeDef> GetGradientDefs() const override; \
  };

#define DECLARE_GRADIENT_BUILDER_DISABLE_COPY_ATTRIBUTES(name) \
  class name : public GradientBuilderBase {                    \
    using GradientBuilderBase::GradientBuilderBase;            \
    std::vector<NodeDef> GetGradientDefs() const override;     \
    bool CopyAttributes() const override {                     \
      return false;                                            \
    }                                                          \
  };

DECLARE_GRADIENT_BUILDER(GetSinGradient)
DECLARE_GRADIENT_BUILDER(GetMatMulGradient)
DECLARE_GRADIENT_BUILDER(GetSplitGradient)
DECLARE_GRADIENT_BUILDER(GetReluGradient)
DECLARE_GRADIENT_BUILDER(GetAddGradient)
DECLARE_GRADIENT_BUILDER(GetSubGradient)
DECLARE_GRADIENT_BUILDER(GetReduceMeanGradient)
DECLARE_GRADIENT_BUILDER(GetPowGradient)
DECLARE_GRADIENT_BUILDER(GetConcatGradient)
DECLARE_GRADIENT_BUILDER(GetReshapeGradient)
DECLARE_GRADIENT_BUILDER(GetPoolGradient)
DECLARE_GRADIENT_BUILDER(GetLRNGradient)
DECLARE_GRADIENT_BUILDER_DISABLE_COPY_ATTRIBUTES(GetDropoutGradient)
DECLARE_GRADIENT_BUILDER(GetConvGradient)
DECLARE_GRADIENT_BUILDER(GetSoftmaxGradient)
DECLARE_GRADIENT_BUILDER(GetSoftmaxCrossEntropyGradient)
DECLARE_GRADIENT_BUILDER(GetGlobalAveragePoolGradient)
DECLARE_GRADIENT_BUILDER_DISABLE_COPY_ATTRIBUTES(GetGemmGradient)

}  // namespace training
}  // namespace onnxruntime
