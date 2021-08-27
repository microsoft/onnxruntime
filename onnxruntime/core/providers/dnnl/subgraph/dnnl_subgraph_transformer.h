// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlGraphTransformer {
 public:
  void Apply(DnnlSubgraph& subgraph);
  DnnlGraphTransformer() = default;

 private:
  void ConvRelu(DnnlSubgraph& subgraph);
  void MatMulAdd(DnnlSubgraph& subgraph);
  void ResolveFusion(DnnlSubgraph& subgraph, std::vector<size_t> old_indices, std::unique_ptr<DnnlNode> new_node);
  bool ProduceGraphOutput(DnnlSubgraph& subgraph, DnnlNode& node);
  bool IsGraphOutput(DnnlSubgraph& subgraph, DnnlTensor& tensor);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
