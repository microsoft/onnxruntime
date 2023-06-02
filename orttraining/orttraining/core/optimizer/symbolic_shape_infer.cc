// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/symbolic_shape_infer.h"

#include <limits>

#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

Status SymbolicShapeInferer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger)
    const {
  // LOGS(logger, WARNING) << "Enter SymbolicShapeInferer::ApplyImpl";
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Reshape", {1, 5, 13, 14}, kOnnxDomain)) {
      continue;
    }

    // LOGS(logger, WARNING) << "Enter SymbolicShapeInferer::ApplyImpl Loop for Reshape " << node->Name();

    auto reshape_out_shape = node->OutputDefs()[0]->Shape();
    if (reshape_out_shape) {
      bool found_emtpy_dim_param = false;
      for (auto& dim : reshape_out_shape->dim()) {
        if (utils::HasDimValue(dim)) {
          // LOGS(logger, WARNING) << "Found 1111111111 empty dim_value in Reshape " << node->Name() << "[" << dim.dim_value() << "]";
        } else if (utils::HasDimParam(dim)) {
          // LOGS(logger, WARNING) << "Found 2222222222 empty dim_param in Reshape " << node->Name() << "[" << dim.dim_param() << "]";
          if (dim.dim_param() == "") {
            found_emtpy_dim_param = true;
            break;
          }
        } else {
          found_emtpy_dim_param = true;
          // LOGS(logger, WARNING) << "Found 3333333333 empty dim_value and dim_param in Reshape " << node->Name();
        }
      }

      if (!found_emtpy_dim_param) {
        // LOGS(logger, WARNING) << "Exit 9999 SymbolicShapeInferer::ApplyImpl reshape_out_shape->dim_size(): " << reshape_out_shape->dim_size();
        continue;
      }
    }

    auto data_shape = node->MutableInputDefs()[0]->Shape();
    // auto reshape_out_shape = node.MutableOutputDefs()[0]->Shape();
    if (data_shape == nullptr) {
      // LOGS(logger, WARNING) << "Skip Reshape node " + node->Name() + " due to undefined shape.";
      continue;
    }

    const auto data_rank = data_shape->dim_size();
    if (data_rank != 3) {
      // LOGS(logger, WARNING) << "Skip Reshape node " + node->Name() + " due to data rank != 3.";
      continue;
    }

    if (!graph_utils::IsConstantInitializer(graph, node->InputDefs()[1]->Name(), /* check_outer_scope */ false)) {
      // LOGS(logger, WARNING) << "Skip Reshape node " + node->Name() + " due to target shape is non-constant initializer.";
      continue;
    }

    bool are_first_two_dims_concrete = utils::HasDimValue(data_shape->dim(0)) && utils::HasDimValue(data_shape->dim(1));
    int64_t merged_dims_value = are_first_two_dims_concrete
                                    ? data_shape->dim(0).dim_value() * data_shape->dim(1).dim_value()
                                    : -1;

    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *node->InputDefs()[1], new_shape_const_values, true);
    if (new_shape_const_values.size() != 2 ||
        !(new_shape_const_values[0] == -1 || new_shape_const_values[0] == merged_dims_value)) {
      // LOGS(logger, WARNING) << " Skip Reshape node " + node->Name() + " due to target shape is not merging first two dims.";
      continue;
    }

    if (!utils::HasDimValue(data_shape->dim(2))) {
      // LOGS(logger, WARNING) << " Skip Reshape node " + node->Name() + " due to the last dim size is not concrete value.";
      continue;
    }

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    std::ostringstream oss;
    bool found_invalid = false;
    for (int i = 0; i < 2; ++i) {
      if (i == 1) {
        oss << "*";
      }
      auto& dim = data_shape->dim(i);
      if (dim.has_dim_value()) {
        oss << dim.dim_value();
      } else if (dim.has_dim_param()) {
        oss << dim.dim_param();
      } else {
        // std::cout << "node->Name(): " << node->Name() << ", node->OpType(): " << node->OpType() << std::endl;
        found_invalid = true;
        break;
        // ORT_THROW("Invalid new_dim found");
      }
    }

    if (found_invalid) {
      continue;
    }

    output_shape.add_dim()->set_dim_param(oss.str());

    auto& dim = data_shape->dim(2);
    if (dim.has_dim_value()) {
      output_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      output_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found");
    }

    node->MutableOutputDefs()[0]->SetShape(output_shape);
    modified = true;

    LOGS(logger, WARNING) << " Reshape node " + node->Name() + " is merged.";
  }

  return Status::OK();
}

}  // namespace onnxruntime
