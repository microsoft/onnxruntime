// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlReduce {
 public:
  enum InputTensors : int {
    IN_DATA = 0,
    IN_AXES = 1
  };

  enum OutputTensors : int {
    OUT_REDUCED = 0
  };
  DnnlReduce();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
  std::vector<int64_t> ReadAxes(DnnlNode& node);
  bool Keepdims(DnnlNode& node);
  bool NoOpWithEmptyAxes(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime