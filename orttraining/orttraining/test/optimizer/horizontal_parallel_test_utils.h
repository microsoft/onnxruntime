// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace horizontal_parallel_test_utils {

// Merge the nodes/initializers of graph_to_merge on rank into graph, which is usually on rank 0.
Status MergeGraph(Graph& graph, Graph& graph_to_merge, int rank, std::vector<Node*>& megatronGs);

Status MergeGraphsOnAllWorkers(std::vector<Graph*>& graphs, Graph& combine_graph);

void VerifyOutputs(const Tensor& expected_tensor, const Tensor& actual_tensor, bool use_threshold_compare,
                   float atol = 1e-8, float rtol = 1e-5, float threshold = 1e-3);

void VerifyOutputs(const std::vector<float>& expected, const std::vector<float>& actual,
                   bool use_threshold_compare, float atol = 1e-8, float rtol = 1e-5, float threshold = 1e-3);

Status GetDataAndShapeFromTensorProto(const Graph& graph, const NodeArg* input_arg,
                                      std::vector<float>& data, std::vector<int64_t>& shape);

}  // namespace horizontal_parallel_test_utils
}  // namespace onnxruntime
