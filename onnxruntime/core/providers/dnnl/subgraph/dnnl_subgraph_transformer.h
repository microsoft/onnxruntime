// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlGraphTransformer {
 public:
  void Apply(DnnlSubgraph& subgraph);
  DnnlGraphTransformer() {
    const std::string debug_log_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_DEBUG_LOG");
    if (!debug_log_env.empty()) {
      debug_log_ = (std::stoi(debug_log_env) == 0 ? false : true);
    }
  }

 private:
  void Gelu(DnnlSubgraph& subgraph);
  void FastGelu(DnnlSubgraph& subgraph);
  bool FastGeluFirstFormula(DnnlSubgraph& subgraph, DnnlNode* node, int& fastgelu_index);
  void FastGeluSecondFormula(DnnlSubgraph& subgraph, DnnlNode* node, int& fastgelu_index);
  bool FastGeluFormulaCommon(DnnlSubgraph& subgraph, DnnlNode* gelu_start_node, int32_t x_input_index, DnnlNode* tanh_node, std::vector<size_t>& gelu_indices, int& fastgelu_index);
  bool IsInitilizedWithExpectedValue(DnnlSubgraph& subgraph, DnnlTensor& input_arg, float expected_value);
  void ConvRelu(DnnlSubgraph& subgraph);
  void MatMulAdd(DnnlSubgraph& subgraph);
  void RemoveMatMulIntegerZP(DnnlSubgraph& subgraph);
  // This function checks a few things
  //   - the node in question has a single output
  //   - The output of the node is only consumed by a one other node
  //   - the output tensor from the node is going to another node within the subgraph
  // If all of the above is true this will return true. It will return false otherwise.
  // 
  // It is possible for a node to fail one or more of the checks above and still be fusable.
  //
  // The name of the function was chosen because this check is required for most of the node fusions
  // found in the code.
  // 
  // The last node in a fusion does not need to pass this check.
  bool IsNodeFusable(DnnlSubgraph& subgraph, DnnlNode* node) const;
  void ResolveFusion(DnnlSubgraph& subgraph, std::vector<size_t> old_indices, std::unique_ptr<DnnlNode> new_node);
  bool ProduceGraphOutput(DnnlSubgraph& subgraph, DnnlNode& node);
  bool IsGraphOutput(DnnlSubgraph& subgraph, DnnlTensor& tensor);
  bool debug_log_ = false;
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
