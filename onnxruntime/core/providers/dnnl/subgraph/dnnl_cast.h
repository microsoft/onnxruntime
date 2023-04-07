// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlCast {
 public:
  enum InputTensors : int {
    IN_INPUT = 0
  };

  enum OutputTensors : int {
    OUT_OUTPUT = 0
  };

  DnnlCast();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  int64_t GetTo(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
