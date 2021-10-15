// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlReshape{
 public:
  enum InputTensors : int {
    IN_DATA = 0,
    IN_SHAPE =1,
  };

  enum OutputTensors : int {
    OUT_RESHAPED = 0
  };

  DnnlReshape();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

  private:
  bool IsMemoryInExpectedOrtFormat(const dnnl::memory::desc& desc);
  bool GetAllowZero(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime