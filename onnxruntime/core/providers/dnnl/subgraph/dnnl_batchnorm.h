// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlBatchNorm {
 public:
  enum InputTensors : int {
    IN_X = 0,
    IN_SCALE = 1,
    IN_B = 2,
    IN_MEAN = 3,
    IN_VAR = 4
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };
  DnnlBatchNorm();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
  float ReadEpsilon(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime