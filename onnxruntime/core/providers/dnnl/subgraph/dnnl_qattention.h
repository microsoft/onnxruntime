// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include <cmath>
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlQAttention {
 public:
  enum InputTensors : int {
    INPUT = 0,
    WEIGHTS = 1,
    BIAS = 2,
    INPUT_SCALE = 3,
    WEIGHTS_SCALE = 4,
    MASK_INDEX = 5,
    INPUT_ZP = 6,
    WEIGHTS_ZP = 7,
    PAST = 8  // not suppoted
  };

  enum OutputTensors : int {
    OUTPUT = 0,
    PRESENT = 1  // not supported
  };

  DnnlQAttention();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  dnnl::memory ComputeTotalScale(DnnlSubgraphPrimitive& sp, DnnlNode& node);
  dnnl::memory::dim GetNumHeads(DnnlNode& node);
  dnnl::memory CopySubMemory(DnnlSubgraphPrimitive& sp, dnnl::memory& src_mem, dnnl::memory::dims sub_mem_dims, dnnl::memory::dims sub_mem_offset);
  dnnl::memory CastMemory(DnnlSubgraphPrimitive& sp, dnnl::memory& src_mem, dnnl::memory::data_type dst_datatype);
  dnnl::memory CastAndTransformMemory(DnnlSubgraphPrimitive& sp, dnnl::memory& src_mem, dnnl::memory::data_type dst_datatype, dnnl::memory::dims dst_strides);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
