// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/utils.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/platform/env.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "asserts.h"
#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(OptimizerTest, Basic) {
  Model model("OptimizerBasic", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
  const logging::Logger& logger = DefaultLoggingManager().DefaultLogger();
  auto& graph = model.MainGraph();

  constexpr int tensor_dim = 10;
  constexpr int input_num = 2;
  TensorProto initializer_tensor[input_num];
  std::vector<std::unique_ptr<NodeArg>> inputs(input_num);
  std::vector<std::unique_ptr<NodeArg>> outputs(1);
  InitializedTensorSet initialized_tensor_set;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(tensor_dim);

  for (int i = 0; i < input_num; i++) {
    string name("input_" + std::to_string(i));
    inputs[i] = std::make_unique<NodeArg>(name, &tensor_int32);

    initializer_tensor[i].set_name(inputs[i]->Name());
    initializer_tensor[i].add_dims(tensor_dim);
    initializer_tensor[i].set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    for (int j = 0; j < tensor_dim; j++) {
      initializer_tensor[i].add_int32_data((i + 1) * j);
    }
    initialized_tensor_set[name] = &initializer_tensor[i];
  }
  outputs[0] = std::make_unique<NodeArg>("out", &tensor_int32);

  std::vector<NodeArg*> tmp_inputs{inputs[0].get(), inputs[1].get()};
  std::vector<NodeArg*> tmp_outputs{outputs[0].get()};
  graph.AddNode("a", "Add", "a", tmp_inputs, tmp_outputs);
  ASSERT_STATUS_OK(graph.Resolve());

  std::vector<const Node*> nodes;
  for (auto& node : graph.Nodes()) {
    nodes.push_back(&node);
  }

  auto cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
#if !defined(DISABLE_SPARSE_TENSORS)
  OptimizerExecutionFrame::Info info(
      nodes, initialized_tensor_set, graph.ModelPath(), *cpu_execution_provider.get(),
      [&graph](const std::string& name) -> bool {
        return graph.IsSparseInitializer(name);
      },
      logger);
#else
  OptimizerExecutionFrame::Info info(
      nodes, initialized_tensor_set, graph.ModelPath(), *cpu_execution_provider.get(),
      [](std::string const&) { return false; },
      logger);
#endif  //! defined(DISABLE_SPARSE_TENSORS)

  std::vector<int> fetch_mlvalue_idxs{info.GetMLValueIndex("out")};
  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

  const ConfigOptions empty_config_options;

  for (auto& node : graph.Nodes()) {
    auto kernel = info.CreateKernel(&node, empty_config_options);

    // kernel can only be a nullptr if a CPU kernel implementation has been removed,
    // if that is the case, OpKernelContext instance construction will throw in the next step
    // and fail the test
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 6387)
#endif
    OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, nullptr, logger);
#ifdef _WIN32
#pragma warning(pop)
#endif

    auto st = kernel->Compute(&op_kernel_context);
    ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(frame.GetOutputs(fetches));
    auto& tensor = fetches[0].Get<Tensor>();
    const std::vector<int32_t> found(tensor.Data<int32_t>(), tensor.Data<int32_t>() + tensor_dim);
    std::vector<int32_t> expected;
    for (int j = 0; j < tensor_dim; j++) {
      expected.push_back(3 * j);
    }
    ASSERT_EQ(expected, found);
  }
}

namespace {
class OptimizerOpaqueType final : public NonTensorTypeBase {
 public:
  OptimizerOpaqueType() : NonTensorTypeBase(sizeof(int)) {
    data_types_internal::AssignOpaqueDomainName("com.microsoft.test", "OptimizerOpaque", MutableTypeProto());
  }

  bool IsCompatible(const TypeProto& type_proto) const override {
    return IsOpaqueCompatible(type_proto);
  }

  DeleteFunc GetDeleteFunc() const override {
    return nullptr;
  }

  CreateFunc GetCreateFunc() const override {
    return nullptr;
  }

  void CreateOrtValue(OrtValue& output) const override {
    output.Init(new int(42), this, [](void* value) { delete static_cast<int*>(value); });
  }
};

// Adds a scalar NodeArg of the given element type, optionally backed by an initializer with the
// supplied dims. When dims is empty no initializer is created, so the NodeArg has no tensor behind it.
NodeArg& AddScalarTypedArg(Graph& graph, const std::string& name, TensorProto_DataType elem_type,
                           bool add_initializer, const std::vector<int64_t>& dims) {
  TypeProto scalar_type;
  scalar_type.mutable_tensor_type()->set_elem_type(elem_type);
  // An empty shape (no dims) is how a rank-0 tensor type is expressed.
  scalar_type.mutable_tensor_type()->mutable_shape();

  // Create the NodeArg with the scalar type before registering the initializer.
  // AddInitializedTensor() only fills in a NodeArg itself when one doesn't already exist, using a
  // shapeless TypeProto; creating it here first keeps the rank-0 shape so IsScalar() sees a scalar.
  NodeArg& node_arg = graph.GetOrCreateNodeArg(name, &scalar_type);

  if (add_initializer) {
    TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(elem_type);
    for (int64_t dim : dims) {
      tensor_proto.add_dims(dim);
    }
    graph.AddInitializedTensor(tensor_proto);
  }

  return node_arg;
}
}  // namespace

TEST(OptimizerTest, AllocateOpaqueOutputThroughType) {
  OptimizerOpaqueType opaque_type;
  DataTypeImpl::RegisterDataType(&opaque_type);

  {
    Model model("OptimizerOpaqueOutput", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {},
                DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    auto& output = graph.GetOrCreateNodeArg("opaque_output", opaque_type.GetTypeProto());
    std::vector<NodeArg*> outputs{&output};
    auto& node = graph.AddNode("opaque", "OpaqueOutput", "", {}, outputs);
    std::vector<const Node*> nodes{&node};

    CPUExecutionProvider cpu_execution_provider{CPUExecutionProviderInfo{}};
    OptimizerExecutionFrame::Info info(
        nodes, InitializedTensorSet{}, graph.ModelPath(), cpu_execution_provider,
        [](const std::string&) { return false; }, DefaultLoggingManager().DefaultLogger());
    OptimizerExecutionFrame frame(info, {info.GetMLValueIndex(output.Name())});

    OrtValue* value = nullptr;
    ASSERT_STATUS_OK(frame.GetOrCreateNodeOutputMLValue(static_cast<int>(node.Index()), 0, nullptr, value, node));
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*static_cast<const int*>(value->DataRaw()), 42);
  }

  DataTypeImpl::UnregisterDataType(&opaque_type);
}

// Fusion helpers call these utilities on NodeArgs taken straight from a matched subgraph. A model
// is free to leave such an input without a (constant) initializer, or to declare an initializer that
// holds no elements, so neither the lookup result nor the element count can be assumed.
TEST(OptimizerTest, ScalarInitializerLookupHandlesMissingAndEmptyTensors) {
  const logging::Logger& logger = DefaultLoggingManager().DefaultLogger();
  Model model("ScalarInitializerLookup", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 13}}, {}, logger);
  Graph& graph = model.MainGraph();

  // No initializer at all behind the NodeArg.
  NodeArg& missing_int = AddScalarTypedArg(graph, "missing_int", TensorProto_DataType_INT64, false, {});
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, missing_int, int64_t{0}, true));
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, missing_int, int64_t{0}, false));

  NodeArg& missing_float = AddScalarTypedArg(graph, "missing_float", TensorProto_DataType_FLOAT, false, {});
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, missing_float, 0.0f, true));
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, missing_float, 0.0f, false));

  float scalar_value = 1.0f;
  EXPECT_FALSE(optimizer_utils::GetScalarInitializerValue<float>(graph, missing_float, scalar_value, true));
  EXPECT_FALSE(optimizer_utils::GetScalarInitializerValue<float>(graph, missing_float, scalar_value, false));

  // Initializer present, but it declares zero elements while the NodeArg type says scalar.
  // Confirm the NodeArg is actually seen as a scalar first, otherwise the checks below would
  // trivially pass via the unrelated "not a scalar" rejection instead of the element-count guard.
  NodeArg& empty_int = AddScalarTypedArg(graph, "empty_int", TensorProto_DataType_INT64, true, {0});
  ASSERT_TRUE(optimizer_utils::IsScalar(empty_int));
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, empty_int, int64_t{0}, true));
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, empty_int, int64_t{0}, false));

  NodeArg& empty_float = AddScalarTypedArg(graph, "empty_float", TensorProto_DataType_FLOAT, true, {0});
  ASSERT_TRUE(optimizer_utils::IsScalar(empty_float));
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, empty_float, 0.0f, true));
  EXPECT_FALSE(optimizer_utils::IsInitializerWithExpectedValue(graph, empty_float, 0.0f, false));
  EXPECT_FALSE(optimizer_utils::GetScalarInitializerValue<float>(graph, empty_float, scalar_value, true));
  EXPECT_FALSE(optimizer_utils::GetScalarInitializerValue<float>(graph, empty_float, scalar_value, false));
}

}  // namespace test
}  // namespace onnxruntime
