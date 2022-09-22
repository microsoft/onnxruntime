// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include <string>

namespace onnxruntime {
namespace ort_dnnl {

class DnnlMatMul {
 public:
  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_BINARY_0 = 2  // the first binary input due to matmul + binary fusion
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlMatMul();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);
 
 private:
  bool GetTransA(DnnlNode& node);
  bool GetTransBatchA(DnnlNode& node);
  bool GetTransB(DnnlNode& node);
  bool GetTransBatchB(DnnlNode& node);
  float GetAlpha(DnnlNode& node);
  float GetFloatAttr(DnnlNode& node, std::string attr_name, float default_value);
  dnnl::memory::dims GetStrides(dnnl::memory::dims& data_dims,
                                bool trans,
                                bool transBatch,
                                dnnl::memory::dims& transposed_dims);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime