// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlMatMulInteger {
 public:
  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_ZERO_POINT = 2,
    IN_B_ZERO_POINT = 3
  };

  enum OutputTensors : int { OUT_Y = 0 };

  DnnlMatMulInteger();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
