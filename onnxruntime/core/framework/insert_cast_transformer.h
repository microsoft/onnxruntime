// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_transformer.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class InsertCastTransformer : public onnxruntime::GraphTransformer {
 public:
  InsertCastTransformer(const std::string& name)
      : onnxruntime::GraphTransformer(name, "Transformer to insert cast node that casts float16 to float for cpu nodes"),
        force_cpu_fp32_(true) {
  }

  void AddKernelRegistries(const std::vector<const KernelRegistry*>& kernels) {
    for (auto* kernel : kernels) {
      if (kernel)
        kernels_registries_.push_back(kernel);
    }
  }

  void AddKernelRegistry(const KernelRegistry& kernel) {
    kernels_registries_.push_back(&kernel);
  }

  Status Apply(onnxruntime::Graph& graph, bool& modified) const override;

 private:
  bool NeedInsertCast(const onnxruntime::Node* node, const onnxruntime::NodeArg* input) const;

  std::vector<const KernelRegistry*> kernels_registries_;
  // Currently because we only have very few cpu kernels support float16, place those nodes on float16
  // will introduce many cast between fp32 and fp16, which will slow the execution. 
  // A better solution is to have a cost model to evaluate does it works to place the node on float16.
  // Here for simplify, we only force the single-node-float16 sub-graph to float32
  bool force_cpu_fp32_;
};
}  // namespace onnxruntime
