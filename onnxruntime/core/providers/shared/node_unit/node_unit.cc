// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_unit.h"
#include "core/graph/graph.h"

namespace onnxruntime {

NodeUnit::NodeUnit(const Node& node)
    : nodes_{&node},
      node_(node),
      type_(Type::SingleNode) {
  InitForNode();
}

const std::string& NodeUnit::Domain() const noexcept { return node_.Domain(); }
const std::string& NodeUnit::OpType() const noexcept { return node_.OpType(); }
const std::string& NodeUnit::Name() const noexcept { return node_.Name(); }
int NodeUnit::SinceVersion() const noexcept { return node_.SinceVersion(); }
NodeIndex NodeUnit::Index() const noexcept { return node_.Index(); }
const Path& NodeUnit::ModelPath() const noexcept { return node_.ModelPath(); }
ProviderType NodeUnit::GetExecutionProviderType() const noexcept { return node_.GetExecutionProviderType(); }

void NodeUnit::InitForNode() {
  const auto& input_defs = node_.InputDefs();
  const auto& output_defs = node_.OutputDefs();
  // The 1st step is to hookup the NodeUnit with the NNAPI builder interface
  // So we are not handling quantization here now
  // TODO, enable quantization
  // auto qlinear_type = GetQLinearOpType(node_);
  // if (qlinear_type == QLinearOpType::Unknown) {
  // Not a Qlinear op, add all inputs/outputs
  auto add_all_io = [](std::vector<IODef>& defs,
                       const ConstPointerContainer<std::vector<NodeArg*>>& node_defs) {
    defs.reserve(node_defs.size());

    for (const auto def : node_defs) {
      defs.push_back(NodeUnit::IODef{*def, std::nullopt});
    }
  };
  add_all_io(input_defs_, input_defs);
  add_all_io(output_defs_, output_defs);
}

}  // namespace onnxruntime
