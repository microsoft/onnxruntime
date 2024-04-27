// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_fusions.h"

#include <vector>
#include "core/framework/node_unit.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

namespace onnxruntime {
namespace qnn {

Status TryFusions(/*out*/ std::vector<const NodeUnit*>& fused_nodes,
                  QnnModelWrapper& qnn_model_wrapper,
                  const NodeUnit& starting_node,
                  const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                  const logging::Logger& logger,
                  bool do_op_validation) {
  ORT_RETURN_IF_ERROR(TryHandleConvertSequence(fused_nodes, qnn_model_wrapper, starting_node, node_unit_map, logger, do_op_validation));

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
