// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlGemm {
 public:
  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_C = 2
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlGemm();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  float GetAlpha(DnnlNode& node);
  float GetBeta(DnnlNode& node);
  bool GetTransA(DnnlNode& node);
  bool GetTransB(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime