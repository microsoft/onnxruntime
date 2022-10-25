// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlGraphTransformer {
 public:
  // The passed in onnx subgraph viewer is only valid during "Compile" phase, 
  // so keep a reference to that onnx subgraph in DnnlSubgraph is risky.
  // passed in the onnx subgraph viewer explicitly to make sure we manage the lifetime correctly.
  void Apply(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer_);
  DnnlGraphTransformer() {
    const std::string debug_log_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_DEBUG_LOG");
    if (!debug_log_env.empty()) {
      debug_log_ = (std::stoi(debug_log_env) == 0 ? false : true);
    }
  }

 private:
  void Gelu(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer);
  void FastGelu(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer);
  bool FastGeluFirstFormula(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlNode* node, int& fastgelu_index);
  void FastGeluSecondFormula(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlNode* node, int& fastgelu_index);
  bool FastGeluFormulaCommon(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlNode* gelu_start_node, int32_t x_input_index, DnnlNode* tanh_node, std::vector<size_t>& gelu_indices, int& fastgelu_index);
  bool IsInitilizedWithExpectedValue(const onnxruntime::GraphViewer& onnx_subgraph_viewer, DnnlTensor& input_arg, float expected_value);
  void ConvRelu(DnnlSubgraph& subgraph);
  void MatMulBinaryEltwise(DnnlSubgraph& subgraph);
  void RemoveMatMulIntegerZP(DnnlSubgraph& subgraph, const onnxruntime::GraphViewer& onnx_subgraph_viewer);
  void MatMulIntegerBinaryEltwise(DnnlSubgraph& subgraph);
  // Function used to identify and fuse post ops
  //
  // @param[in] subgraph the DnnlSubgrapy that we are searching for possible fusions
  // @param[in] node is the first node to check if it contains a binary or an elementwise op
  // @param[in/out] indicies list of all the indicies for the nodes that will be fused
  // @param[in/out] fused_node_inputs list of all the inputs that will be part of the fused node
  // @param[in/out] attr_node this node contains the attributes that will be passed onto the final fused node
  //
  // @return a pointer to the node after the last identified binary/elementwise fusion
  DnnlNode* FuseBinaryEltwisePostOps(DnnlSubgraph& subgraph, DnnlNode* node, std::vector<size_t>& indices, std::vector<DnnlTensor*>& fused_node_inputs, DnnlNode*& attr_node);
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
