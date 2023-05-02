// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlLayerNorm {
 public:
  typedef std::pair<dnnl::layer_normalization_forward, std::unordered_map<int, dnnl::memory>> ln_components;

  enum InputTensorsSLN : int {
    IN_INPUT = 0,
    IN_SKIP = 1,
    IN_SLN_GAMMA = 2,
    IN_BETA = 3,     // Optional
    IN_SLN_BIAS = 4  // Optional
  };

  enum InputTensorsLN : int {
    // IN_INPUT = 0,
    IN_LN_GAMMA = 1,
    IN_LN_BIAS = 2  // Optional
  };

  enum OutputTensors : int {
    OUT_OUTPUT = 0,
    OUT_MEAN = 1,        // Optional
    OUT_INV_STD_VAR = 2  // Optional
  };
  DnnlLayerNorm();

  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  dnnl::memory BuildSLN(DnnlSubgraphPrimitive& sp, DnnlNode& node, dnnl::engine dnnl_engine);
  void ValidateDims(DnnlSubgraphPrimitive& sp, DnnlNode& node);
  float GetEpsilon(DnnlNode& node);
  dnnl::memory CastAndTransformMemory(DnnlSubgraphPrimitive& sp, dnnl::memory& src_mem, dnnl::memory::data_type dst_datatype, dnnl::memory::dims dst_strides);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime