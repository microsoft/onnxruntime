// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlRelu {
 public:
  enum InputTensors : int {
    IN_X = 0,
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlRelu();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime