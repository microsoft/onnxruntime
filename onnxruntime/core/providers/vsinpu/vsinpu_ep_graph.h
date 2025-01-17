/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#pragma once
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "builders/op_builder.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
struct GraphIOInfo {
  std::string name;
  bool is_initializer;
  std::shared_ptr<tim::vx::Tensor> tensor;
  TensorShape shape;
};

struct NodeIOInfo {
  std::shared_ptr<tim::vx::Operation> op_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

class GraphEP {
 public:
  explicit GraphEP(const GraphViewer& graph_viewer, const logging::Logger& logger);
  ~GraphEP() {}

  bool Prepare();

  static bool SupportedOp(const onnxruntime::GraphViewer& graph_viewer,
                          const NodeUnit& node_unit);

  // If a node is supported by VSINPU in a partition node group
  // `node_outputs_in_group` is the set of the output names of the nodes added to this group so far
  static bool IsNodeSupportedInGroup(const NodeUnit& node_unit, const GraphViewer& graph_viewer);

  const NodeUnit& GetNodeUnit(const Node* node) const;

  bool& GetCompiled() { return compiled_; }
  std::shared_ptr<tim::vx::Graph>& GetGraph() { return graph_; }
  std::vector<std::shared_ptr<tim::vx::Operation>>& GetOps() { return ops_; }
  std::map<std::string, std::shared_ptr<tim::vx::Tensor>>& GetTensors() {
    return tensors_;
  }

  std::vector<std::shared_ptr<GraphIOInfo>>& GetGraphInputs() {
    return graph_inputs_;
  }

  std::vector<std::shared_ptr<GraphIOInfo>>& GetGraphOutputs() {
    return graph_outputs_;
  }

  void UpdateTensorMap(const std::string& name, const std::shared_ptr<tim::vx::Tensor>& dst_tensor);

  std::shared_ptr<NodeIOInfo> ConstructNodeIO(const std::shared_ptr<tim::vx::Operation>& op,
                                              std::vector<NodeArg*> input_arg, std::vector<NodeArg*> output_arg);

  bool BindTensors(const std::shared_ptr<NodeIOInfo>& nodeio_info);

  std::shared_ptr<tim::vx::Tensor> MapTIMVXTensor(
      std::shared_ptr<tim::vx::Graph>& graph, const NodeUnitIODef nudef,
      const NodeUnit& nodeunit,
      const GraphViewer* graph_viewer, tim::vx::TensorAttribute attribute);

 private:
  std::shared_ptr<tim::vx::Context> context_;
  std::shared_ptr<tim::vx::Graph> graph_;
  std::map<std::string, std::shared_ptr<tim::vx::Tensor>> tensors_;
  std::vector<std::shared_ptr<tim::vx::Operation>> ops_;
  std::vector<std::shared_ptr<GraphIOInfo>> graph_inputs_;
  std::vector<std::shared_ptr<GraphIOInfo>> graph_outputs_;

  // Contains all quantized operators' input and the NodeUnit(s) using the input
  // In the form of {input_name, [NodeUnit(s) using the input]}
  std::unordered_map<std::string, std::vector<const NodeUnit*>> all_quantized_op_inputs_;
  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;

  // Holder for the NodeUnits in the graph, this will guarantee the NodeUnits is
  // valid throughout the lifetime of the ModelBuilder
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder_;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map_;
  bool compiled_;
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
