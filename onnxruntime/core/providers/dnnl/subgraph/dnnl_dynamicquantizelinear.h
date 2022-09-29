// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlDynamicQuantizeLinear {
 public:
  enum InputTensors : int {
    IN_X = 0,  // Input tensor float32
  };

  enum OutputTensors : int {
    OUT_Y = 0,        // Quantized output tensor, tensor uint8
    OUT_Y_SCALE = 1,  // Output scale. It's a scalar, which means a per-tensor/layer quantization, tensor float32
    OUT_Y_ZP = 2,     // Output zero point. It's a scalar, which means a per-tensor/layer quantization, tensor uint8
  };

  DnnlDynamicQuantizeLinear() = default;
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  void WriteZeroToMem(dnnl::memory& mem);
  dnnl::memory::desc ChangeMemoryDescDataType(dnnl::memory::desc md, dnnl::memory::data_type dt);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime