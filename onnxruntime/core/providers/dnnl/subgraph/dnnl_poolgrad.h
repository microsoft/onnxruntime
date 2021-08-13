// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlPoolGrad {
 public:
  enum InputTensors : int {
    IN_DY = 0,
    IN_INDICES = 1
  };

  enum OutputTensors : int {
    OUT_DX = 0
  };

  enum PoolShape : size_t {
    SHAPE_UNKNOWN = 0,
    SHAPE_1D = 1,
    SHAPE_2D = 2,
    SHAPE_3D = 3
  };

  DnnlPoolGrad();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  AutoPadType GetAutoPad(DnnlNode& node);
  int64_t GetCeilMode(DnnlNode& node);
  int64_t GetCountIncludePadding(DnnlNode& node);
  dnnl::memory::dims GetDilations(DnnlNode& node, PoolShape shape);
  dnnl::memory::dims GetKernelShape(DnnlNode& node);
  /* This will return the calculated padding taking into account the DEPRECATED auto_pad attribute */
  std::vector<int64_t> InferPadding(DnnlNode& node, const dnnl::memory::dims& src_dims, const dnnl::memory::dims& kernel_shape, const dnnl::memory::dims& strides);
  std::vector<int64_t> GetPadding(DnnlNode& node, PoolShape shape);
  dnnl::memory::dims GetPaddingLeft(const std::vector<int64_t> padding);
  dnnl::memory::dims GetPaddingRight(const std::vector<int64_t> padding);
  int64_t GetStorageOrder(DnnlNode& node);
  dnnl::memory::dims GetStrides(DnnlNode& node, PoolShape shape);

  dnnl::memory::dims InferOutputDims(DnnlNode& node, const dnnl::memory::dims& src_dims, const dnnl::memory::dims& kernel_shape, const dnnl::memory::dims& strides);
  bool IsGlobalPooling(DnnlNode& node) const;
};

}  // namespace ort_dnnl
}  // namespace onnxruntime