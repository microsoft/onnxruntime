// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

// class DnnlSubgraphPrimitive;
// class DnnlNode;

namespace onnxruntime {
namespace ort_dnnl {

class DnnlSum {
 public:
  enum InputTensors : int {
    IN_DATA_0 = 0,
  };

  enum OutputTensors : int {
    OUT_SUM = 0,
  };

  DnnlSum();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime