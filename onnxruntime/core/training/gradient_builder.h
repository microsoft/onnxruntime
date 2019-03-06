// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/training/gradient_builder_base.h"

namespace onnxruntime {
namespace training {
// TODO: maybe group the gradient builders and split them into different files.
#define DECLEAR_GRADIENT_BUILDER(name)                     \
  class name : public GradientBuilderBase {                \
    using GradientBuilderBase::GradientBuilderBase;        \
    std::vector<NodeDef> GetGradientDefs() const override; \
  };

DECLEAR_GRADIENT_BUILDER(GetSinGradient)
DECLEAR_GRADIENT_BUILDER(GetMatmulGradient)
DECLEAR_GRADIENT_BUILDER(GetSplitGradient)
DECLEAR_GRADIENT_BUILDER(GetReluGradient)
DECLEAR_GRADIENT_BUILDER(GetAddGradient)
DECLEAR_GRADIENT_BUILDER(GetSubGradient)
DECLEAR_GRADIENT_BUILDER(GetReduceMeanGradient)
DECLEAR_GRADIENT_BUILDER(GetPowGradient)

}  // namespace training
}  // namespace onnxruntime
