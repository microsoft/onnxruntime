// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlMatMul {
 public:
  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlMatMul();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime