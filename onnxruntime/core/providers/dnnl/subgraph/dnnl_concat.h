// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

// class DnnlSubgraphPrimitive;
// class DnnlNode;

namespace onnxruntime {
namespace ort_dnnl {

class DnnlConcat {
 public:
  enum InputTensors : int {
    IN_DATA_0 = 0,
  };

  enum OutputTensors : int {
    OUT_CONCAT = 0,
  };

  DnnlConcat();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  int64_t GetAxis(DnnlNode& node, int64_t input_rank);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime