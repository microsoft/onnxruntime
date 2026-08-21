// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/fallback_cpu_capability.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/op_kernel.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr const char* kTestEpType = "TestFallbackEp";

class TestKernelLookup : public IExecutionProvider::IKernelLookup {
 public:
  void Add(const std::string& op_type, std::unique_ptr<KernelDef> kernel_def) {
    kernels_.emplace(op_type, KernelCreateInfo(std::move(kernel_def), nullptr));
  }

  const KernelCreateInfo* LookUpKernel(const Node& node) const override {
    auto it = kernels_.find(node.OpType());
    return it != kernels_.end() ? &it->second : nullptr;
  }

 private:
  std::unordered_map<std::string, KernelCreateInfo> kernels_;
};

// data --> Shape --> shape (CPU output) --> Reshape --> reshaped
//   \                                          ^
//    \_________________________________________/
//
// Reshape consumes a CPU tensor produced by Shape, which makes it a fallback candidate,
// and it also consumes `data` directly. Whether it is preferred on CPU therefore depends
// solely on the element type of `data`.
struct ShapeSubgraph {
  std::shared_ptr<Model> model;
  NodeIndex reshape_node{};
};

void BuildShapeSubgraph(int32_t elem_type, ShapeSubgraph& subgraph) {
  const std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 23}};
  auto model = std::make_shared<Model>("fallback_cpu_capability_test", false, ModelMetaData(), PathString(),
                                       IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                       std::vector<ONNX_NAMESPACE::FunctionProto>(),
                                       DefaultLoggingManager().DefaultLogger());
  Graph& graph = model->MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(elem_type);
  data_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);

  auto& data = graph.GetOrCreateNodeArg("data", &data_type);
  auto& shape = graph.GetOrCreateNodeArg("shape", nullptr);
  auto& reshaped = graph.GetOrCreateNodeArg("reshaped", nullptr);

  graph.AddNode("shape_node", "Shape", "", {&data}, {&shape});
  auto& reshape_node = graph.AddNode("reshape_node", "Reshape", "", {&data, &shape}, {&reshaped});
  ASSERT_STATUS_OK(graph.Resolve());

  subgraph.reshape_node = reshape_node.Index();
  subgraph.model = std::move(model);
}

std::unordered_set<NodeIndex> GetCpuPreferredNodesFor(const ShapeSubgraph& subgraph) {
  const GraphViewer graph_viewer(subgraph.model->MainGraph());

  TestKernelLookup kernel_lookup;
  kernel_lookup.Add("Shape", KernelDefBuilder()
                                 .SetName("Shape")
                                 .Provider(kTestEpType)
                                 .SinceVersion(1)
                                 .OutputMemoryType(OrtMemTypeCPUOutput, 0)
                                 .Build());
  kernel_lookup.Add("Reshape",
                    KernelDefBuilder().SetName("Reshape").Provider(kTestEpType).SinceVersion(1).Build());

  const auto& tentative_nodes = graph_viewer.GetNodesInTopologicalOrder();

  return GetCpuPreferredNodes(graph_viewer, kernel_lookup, tentative_nodes,
                              DefaultLoggingManager().DefaultLogger());
}

}  // namespace

TEST(FallbackCpuCapabilityTest, FullPrecisionInputIsPreferredOnCpu) {
  ShapeSubgraph subgraph;
  ASSERT_NO_FATAL_FAILURE(BuildShapeSubgraph(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, subgraph));

  EXPECT_EQ(GetCpuPreferredNodesFor(subgraph).count(subgraph.reshape_node), 1u);
}

TEST(FallbackCpuCapabilityTest, ReducedPrecisionInputIsNotPreferredOnCpu) {
  constexpr int32_t kReducedPrecisionElemTypes[] = {
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
      ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN,
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ,
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2,
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ,
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT4E2M1,
  };

  for (const int32_t elem_type : kReducedPrecisionElemTypes) {
    SCOPED_TRACE("elem_type=" + std::to_string(elem_type));

    ShapeSubgraph subgraph;
    ASSERT_NO_FATAL_FAILURE(BuildShapeSubgraph(elem_type, subgraph));

    EXPECT_EQ(GetCpuPreferredNodesFor(subgraph).count(subgraph.reshape_node), 0u);
  }
}

}  // namespace test
}  // namespace onnxruntime
