// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Represents a group of NodeUnits that QNN EP translates into a core QNN operator. Can represent a single NodeUnit
/// or a fusion of multiple NodeUnits (e.g., DQ* -> Conv -> Relu -> Q).
/// </summary>
class IQnnNodeGroup {
 public:
  virtual ~IQnnNodeGroup() = default;

  // Returns an OK status if this IQnnNodeGroup is supported by QNN.
  virtual Status IsSupported(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const = 0;

  // Adds this IQnnNodeGroup to the QNN model wrapper.
  virtual Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper, const logging::Logger& logger) const = 0;

  // Returns a list of NodeUnits contained by this IQnnNodeGroup.
  virtual gsl::span<const OrtNodeUnit* const> GetNodeUnits() const = 0;

  /// <summary>
  /// Returns the "target" NodeUnit of the group. This is important for topological ordering of IQnnNodeGroups.
  /// The target should be the first NodeUnit where all input paths (of the IQnnNodeGroup) converge.
  /// For example, "Conv" should be the target NodeUnit for the following IQnnNodeGroup with 6 NodeUnits.
  ///    input0 -> DQ -> Conv -> Relu -> Q
  ///                     ^
  ///                     |
  ///    input1 -> DQ ----+
  /// </summary>
  /// <returns>Target NodeUnit in IQnnNodeGroup</returns>
  virtual const OrtNodeUnit* GetTargetNodeUnit() const = 0;

  // Returns a string representation of the IQnnNodeGroup's type.
  virtual std::string_view Type() const = 0;
};

/// <summary>
/// Traverses the ONNX graph to create IQnnNodeGroup objects, each containing one or more NodeUnits.
/// The returned IQnnNodeGroup objects are sorted in topological order.
/// </summary>
/// <param name="qnn_node_groups">Output vector into which the resulting IQnnNodeGroup objects are stored.</param>
/// <param name="qnn_model_wrapper">Contains reference to the ONNX GraphViewer and used for validaton on QNN</param>
/// <param name="node_to_node_unit">Maps a Node* to a NodeUnit*</param>
/// <param name="num_node_units">The number of NodeUnits in the ONNX graph.</param>
/// <param name="logger">Logger</param>
/// <returns>Status with potential error</returns>
Status GetQnnNodeGroups(/*out*/ std::vector<std::unique_ptr<IQnnNodeGroup>>& qnn_node_groups,
                        QnnModelWrapper& qnn_model_wrapper,
                        const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
                        size_t num_node_units,
                        const logging::Logger& logger);
}  // namespace qnn
}  // namespace onnxruntime
