// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
    
namespace onnxruntime {
namespace ort_dnnl {

class DnnlReduceMean {
 public:
  enum InputTensors : int {
    IN_X = 0
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };
  DnnlReduceMean();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
  std::vector<int64_t> ReadAxes(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime