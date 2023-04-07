// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlTranspose {
 public:
  enum InputTensors : int {
    IN_DATA = 0,
  };

  enum OutputTensors : int {
    OUT_TRANSPOSED = 0
  };

  DnnlTranspose();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  std::vector<int64_t> GetPerm(DnnlNode& node);
};


}  // namespace ort_dnnl
}  // namespace onnxruntime