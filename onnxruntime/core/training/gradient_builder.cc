// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/gradient_builder.h"
#include "core/training/gradient_builder_registry.h"
#include "core/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {

#define IMPLEMENT_GRADIENT_BUILDER(name) \
  std::vector<NodeDef> name::GetGradientDefs() const

IMPLEMENT_GRADIENT_BUILDER(GetSinGradient) {
  return std::vector<NodeDef>{
      NodeDef("Cos",
              {I(0)},
              {IA("cosx")}),
      NodeDef("Mul",
              {IA("cosx"), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetMatmulGradient) {
  std::vector<NodeDef> result = {};

  // is GI(0) required
  if (IsGradientRequiredForSrcNodeInput(0)) {
    // dx = dz * transpose(y)

    //TODO: default perm attrbiute is used here. Explict specify here?
    result.push_back(
        NodeDef("Transpose",
                {I(1)},
                {IA("I1_t")}));
    result.push_back(
        NodeDef("MatMul",
                {GO(0), IA("I1_t")},
                {GI(0)}));
  }

  // is GI(1) required
  if (IsGradientRequiredForSrcNodeInput(1)) {
    // dy = transpose(x) * y
    result.push_back(
        NodeDef("Transpose",
                {I(0)},
                {IA("I0_t")}));
    result.push_back(
        NodeDef("MatMul",
                {IA("I0_t"), GO(0)},
                {GI(1)}));
  }

  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetSplitGradient) {
  std::vector<NodeDef> result = {};
  std::vector<ArgDef> input_args;

  for (int i = 0; i < GetSrcNodeOutputSize(); i++) {
    if (IsGradientAvailableForSrcNodeOutput(i)) {
      input_args.push_back(GO(i));
    }
  }

  if (!input_args.empty()) {
    result.push_back(
        NodeDef("Concat",
                input_args,
                {GI(0)}));
  }

  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetReluGradient) {
  return std::vector<NodeDef>{
      NodeDef("ReluGrad",
              {GO(0), I(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetAddGradient) {
  return std::vector<NodeDef>{
      NodeDef("AddGrad",
              {GO(0)},
              {GI(0), GI(1)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSubGradient) {
  return std::vector<NodeDef>{
      NodeDef("SubGrad",
              {GO(0)},
              {GI(0), GI(1)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetReduceMeanGradient) {
  return std::vector<NodeDef>{
      NodeDef("ReduceMeanGrad",
              {GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetPowGradient) {
  return std::vector<NodeDef>{
      NodeDef("PowGrad",
              {GO(0), I(0), I(1)},
              {GI(0), GI(1)})};
}

}  // namespace training
}  // namespace onnxruntime
