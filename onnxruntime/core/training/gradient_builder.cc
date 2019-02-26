// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/gradient_builder.h"
#include "core/training/gradient_registry.h"

namespace onnxruntime {
class GetSinGradient : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;

  std::vector<OpDef> GetGradientDefs() override {
    return std::vector<OpDef>{
        OpDef("Cos",
              {I(0)},
              {IA("cosx")}),
        OpDef(
            "Mul",
            {IA("cosx"), GO(0)},
            {GI(0)})};
  }
};

class GetMatmulGradient : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;

  std::vector<OpDef> GetGradientDefs() {
    std::vector<OpDef> result = {};

    // is GI(0) required
    if (IsGradientRequiredForSrcNodeInput(0)) {
      result.push_back(
          OpDef("Transpose",
                {I(1)},
                {IA("I1_t")}));

      result.push_back(
          OpDef("Matmul",
                {IA("I1_t"), GO(0)},
                {GI(0)}));
    }

    // is GI(1) required
    if (IsGradientRequiredForSrcNodeInput(1)) {
      result.push_back(
          OpDef("Transpose",
                {I(0)},
                {IA("I0_t")}));

      result.push_back(
          OpDef("Matmul",
                {IA("I0_t"), GO(0)},
                {GI(1)}));
    }

    return result;
  }
};

class GetSplitGradient : public GradientBuilderBase {
  using GradientBuilderBase::GradientBuilderBase;

  std::vector<OpDef> GetGradientDefs() {
    std::vector<OpDef> result = {};
    std::vector<ArgDef> input_args;

    for (int i = 0; i < GetSrcNodeOutputSize(); i++) {
      if (IsGradientAvailableForSrcNodeOutput(i)) {
        input_args.push_back(GO(i));
      }
    }

    if (!input_args.empty()) {
      result.push_back(
          OpDef("Concat",
                input_args,
                {GI(0)}));
    }

    return result;
  }
};

void RegisterGradientBuilders() {
  REGISTER_GRADIENT_BUILDER("Sin", GetSinGradient);
  REGISTER_GRADIENT_BUILDER("MatMul", GetMatmulGradient);
  REGISTER_GRADIENT_BUILDER("Split", GetSplitGradient);
}

}  // namespace onnxruntime
