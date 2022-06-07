// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "test/providers/provider_test_utils.h"
#include "horizontal_parallel_test_utils.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::test;

namespace onnxruntime {
namespace horizontal_parallel_test_utils {

Status MergeGraph(Graph& graph, Graph& graph_to_merge, int rank, std::vector<Node*>& megatronGs) {
  // Merge graph_to_merge's initializers into graph.
  const auto& init_tensors = graph_to_merge.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    const ONNX_NAMESPACE::TensorProto* tmp_tensor;
    // For those initializers already existing, we assume every rank should have same value.
    if (!graph.GetInitializedTensor(tensor.first, tmp_tensor)) {
      graph.AddInitializedTensor(*(tensor.second));
    }
  }

  const GraphViewer graph_viewer(graph_to_merge);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  const std::vector<const NodeArg*>& graph_outputs = graph_to_merge.GetOutputs();
  const std::string postfix = "_rank_" + std::to_string(rank);
  std::vector<const NodeArg*> new_outputs = graph.GetOutputs();
  for (auto node_index : node_topology_list) {
    auto& node = *graph_to_merge.GetNode(node_index);
    std::vector<NodeArg*> input_defs{};
    std::vector<NodeArg*> output_defs{};

    for (auto& input_def : node.MutableInputDefs()) {
      auto name = input_def->Name();
      if (!graph_utils::IsGraphInput(graph_to_merge, input_def) &&
          !graph_utils::IsInitializer(graph_to_merge, input_def->Name(), false)) {
        // We keep input and initializer's name unchanged.
        name = input_def->Name() + postfix;
      }

      auto type_info = *input_def->TypeAsProto();
      // The input/initializer args will be created only for rank 0.
      auto& input_arg = graph.GetOrCreateNodeArg(name, &type_info);
      input_defs.push_back(&input_arg);
    }

    for (auto& output_def : node.MutableOutputDefs()) {
      auto type_info = *output_def->TypeAsProto();
      auto& output_arg = graph.GetOrCreateNodeArg(output_def->Name() + postfix, &type_info);
      output_defs.push_back(&output_arg);

      if (std::find(graph_outputs.begin(), graph_outputs.end(), output_def) != graph_outputs.end()) {
        new_outputs.push_back(&output_arg);
      }
    }

    auto op_type = node.OpType();
    auto domain = node.Domain();
    if (op_type.compare("MegatronF") == 0) {
      op_type = "Identity";
      domain = "";
    }

    auto& new_node = graph.AddNode(node.Name() + postfix,
                                   op_type,
                                   node.Description(),
                                   input_defs,
                                   output_defs, &node.GetAttributes(), domain);

    if (op_type.compare("MegatronG") == 0) {
      megatronGs.push_back(&new_node);
    }
  }

  graph.SetOutputs(new_outputs);
  return graph.Resolve();
}

Status MergeGraphsOnAllWorkers(std::vector<Graph*>& graphs, Graph& combine_graph) {
  auto total_rank = graphs.size();
  std::vector<std::vector<Node*>> megatronGs(total_rank, std::vector<Node*>());
  for (auto i = 0u; i < total_rank; i++) {
    auto merge_ret = horizontal_parallel_test_utils::MergeGraph(combine_graph, *graphs[i], i, megatronGs[i]);
    ORT_ENFORCE(merge_ret.IsOK());
    ORT_ENFORCE(megatronGs[i].size() == megatronGs[0].size());
  }

  std::vector<onnxruntime::NodeIndex> nodes_to_remove;
  // Merge megatron g at the same index for different ranks
  for (auto g_index = 0u; g_index < megatronGs[0].size(); g_index++) {
    // Merge the "g_index"th MegatronG on each rank into one Sum node.
    std::vector<NodeArg*> input_defs{};
    auto type_info = *megatronGs[0][g_index]->MutableOutputDefs()[0]->TypeAsProto();
    auto& input_arg = combine_graph.GetOrCreateNodeArg("sum_" + std::to_string(g_index), &type_info);
    std::vector<NodeArg*> output_defs{&input_arg};

    for (auto rank_index = 0u; rank_index < total_rank; rank_index++) {
      input_defs.push_back(megatronGs[rank_index][g_index]->MutableInputDefs()[0]);
    }
    auto& sum_node = combine_graph.AddNode(combine_graph.GenerateNodeName("Sum_For_MegatronG"),
                                           "Sum",
                                           "Sum For MegatronG",
                                           input_defs,
                                           output_defs);
    sum_node.SetExecutionProviderType(megatronGs[0][g_index]->GetExecutionProviderType());

    for (auto rank_index = 0u; rank_index < total_rank; rank_index++) {
      graph_utils::ReplaceDownstreamNodeInput(combine_graph, *megatronGs[rank_index][g_index], 0, sum_node, 0);
      nodes_to_remove.push_back(megatronGs[rank_index][g_index]->Index());
    }
  }

  std::sort(nodes_to_remove.begin(), nodes_to_remove.end());
  for (const auto& node_index : nodes_to_remove) {
    combine_graph.RemoveNode(node_index);
  }

  return combine_graph.Resolve();
}

void VerifyOutputs(const Tensor& expected_tensor, const Tensor& actual_tensor, bool use_threshold_compare,
                   float atol, float rtol, float threshold) {
  ASSERT_EQ(expected_tensor.Shape(), actual_tensor.Shape());
  auto size = expected_tensor.Shape().Size();
  if (expected_tensor.GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const std::vector<float> expected(expected_tensor.template Data<float>(), expected_tensor.template Data<float>() + size);
    const std::vector<float> actual(actual_tensor.template Data<float>(), actual_tensor.template Data<float>() + size);
    VerifyOutputs(expected, actual, use_threshold_compare, atol, rtol, threshold);
  }
#ifdef USE_CUDA
  else if (expected_tensor.GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    auto* expected = expected_tensor.template Data<MLFloat16>();
    auto* actual = actual_tensor.template Data<MLFloat16>();
  
    std::vector<float> f_expected(size);
    std::vector<float> f_actual(size);
    ConvertMLFloat16ToFloat(expected, f_expected.data(), static_cast<int>(size));
    ConvertMLFloat16ToFloat(actual, f_actual.data(), static_cast<int>(size));
    VerifyOutputs(f_expected, f_actual, use_threshold_compare, math::halfToFloat(math::floatToHalf(atol)),
                  math::halfToFloat(math::floatToHalf(rtol)), math::halfToFloat(math::floatToHalf(threshold)));
  }
#endif
}

void VerifyOutputs(const std::vector<float>& expected, const std::vector<float>& actual,
                   bool use_threshold_compare, float atol, float rtol, float threshold) {
  auto size = expected.size();
  ORT_ENFORCE(size == actual.size());
  for (auto i = 0u; i < size; ++i) {
    const auto expected_value = expected[i], actual_value = actual[i];
    if (std::isnan(expected_value)) {
      ASSERT_TRUE(std::isnan(actual_value)) << "value mismatch at index " << i << "; expected is NaN, actual is not NaN";
    } else if (std::isinf(expected_value)) {
      ASSERT_EQ(expected_value, actual_value) << "value mismatch at index " << i;
    } else {
      double diff = fabs(expected_value - actual_value);
      if (use_threshold_compare) {
        ASSERT_TRUE(diff <= threshold) << "value mismatch at index "
                                       << i << "; diff: " << diff << ", threshold: " << threshold;
      } else {
        ASSERT_TRUE(diff <= (atol + rtol * fabs(expected_value))) << "value mismatch at index "
                                                                  << i << "; expected: " << expected_value << ", actual: " << actual_value;
      }
    }
  }
}

Status GetDataAndShapeFromTensorProto(const Graph& graph, const NodeArg* input_arg,
                                      std::vector<float>& data, std::vector<int64_t>& shape) {
  const ONNX_NAMESPACE::TensorShapeProto* tensor_shape = input_arg->Shape();
  size_t element_count = 1;
  int32_t rank = tensor_shape->dim_size();
  for (auto i = 0; i < rank; i++) {
    ORT_ENFORCE(utils::HasDimValue(tensor_shape->dim(i)));
    auto dim_value = tensor_shape->dim(i).dim_value();
    element_count *= dim_value;
    shape.push_back(dim_value);
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  graph.GetInitializedTensor(input_arg->Name(), tensor_proto);
  auto init_const = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* data_float = init_const->data<float>();
  data.insert(data.end(), data_float, data_float + element_count);

  return Status::OK();
}

}  // namespace horizontal_parallel_test_utils
}  // namespace onnxruntime
