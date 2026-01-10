// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlDequantizeLinear {
 public:
  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,  // Optional
  };

  enum OutputTensors : int {
    OUT_Y = 0,
  };

  DnnlDequantizeLinear() = default;
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  bool isZeroPointNonZero(dnnl::memory* zp_mem);
  int64_t GetAxis(DnnlNode& node, size_t x_dims);
  void Padd(dnnl::memory::desc* target, size_t front_pad, size_t back_pad);
  void ValidateDims(DnnlSubgraphPrimitive& sp, DnnlNode& node);
  void ValidateType(DnnlSubgraphPrimitive& sp, DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime