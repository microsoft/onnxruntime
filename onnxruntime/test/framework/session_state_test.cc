// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "asserts.h"
#include "core/framework/execution_providers.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/util/thread_utils.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "core/optimizer/layout_transformation/layout_transformation.h"

using namespace ONNX_NAMESPACE;
using namespace std;
namespace onnxruntime {

namespace test {
class TestOpKernel : public OpKernel {
 public:
  TestOpKernel(const OpKernelInfo& p) : OpKernel(p) {
  }
  Status Compute(OpKernelContext* context) const override {
    ORT_UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const override {
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(done);
    return Status::OK();
  }
};
class SessionStateAddGetKernelTest : public testing::TestWithParam<int> {};

TEST_P(SessionStateAddGetKernelTest, AddGetKernelTest) {
  OrtThreadPoolParams to;
  to.thread_pool_size = GetParam();
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
  ONNX_OPERATOR_SCHEMA(Variable)
      .SetDoc("Input variable.")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ExecutionProviders execution_providers;
  auto tmp_cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  auto* cpu_execution_provider = tmp_cpu_execution_provider.get();
  ASSERT_STATUS_OK(execution_providers.Add(kCpuExecutionProvider, std::move(tmp_cpu_execution_provider)));

  DataTransferManager dtm;
  profiling::Profiler profiler;

  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;

  SessionState s(graph, execution_providers, tp.get(), nullptr, dtm,
                 DefaultLoggingManager().DefaultLogger(), profiler, sess_options);

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  onnxruntime::NodeArg output_arg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  onnxruntime::Node& node = graph.AddNode("node_1", "Variable", "node 1.", inputs, outputs);
  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  auto kernel_def = KernelDefBuilder().SetName("Variable").Provider(kCpuExecutionProvider).SinceVersion(1, 10).Build();

  OpKernelInfo p_info(node, *kernel_def, *cpu_execution_provider, s.GetConstantInitializedTensors(),
                      s.GetOrtValueNameIdxMap(), s.GetDataTransferMgr());
  unique_ptr<TestOpKernel> p_kernel;
  p_kernel.reset(new TestOpKernel(p_info));
  size_t orig_num_outputs = p_kernel->Node().OutputDefs().size();
  std::cout << "node_idx: " << node.Index() << std::endl;

  KernelRegistryManager kernel_registry_manager;
  status = kernel_registry_manager.RegisterKernels(execution_providers);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  node.SetExecutionProviderType(kCpuExecutionProvider);
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ASSERT_STATUS_OK(kernel_registry->Register(
      KernelCreateInfo(std::move(kernel_def),
                       [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
                         out = std::make_unique<TestOpKernel>(info);
                         return Status::OK();
                       })));
  kernel_registry_manager.RegisterKernelRegistry(kernel_registry);
  ASSERT_STATUS_OK(s.FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager));

  auto test_kernel = s.GetKernel(node.Index());
  std::cout << "orig: " << orig_num_outputs << " new: " << test_kernel->Node().OutputDefs().size() << std::endl;
  EXPECT_EQ(orig_num_outputs, test_kernel->Node().OutputDefs().size());
}

INSTANTIATE_TEST_SUITE_P(SessionStateTests, SessionStateAddGetKernelTest, testing::Values(0, 1));

class TestParam {
 public:
  int ir_version;
  bool enable_mem_pattern;
  int thread_count;
};

TestParam param_list[] = {
    {3, true, 0},
    {4, true, 0},
    {3, false, 0},
    {4, false, 0},
    {3, true, 1},
    {4, true, 1},
    {3, false, 1},
    {4, false, 1}};

class SessionStateTestP : public testing::TestWithParam<TestParam> {};
// Test that we separate out constant and non-constant initializers correctly
TEST_P(SessionStateTestP, TestInitializerProcessing) {
  const TestParam& param = GetParam();
  OrtThreadPoolParams to;
  to.thread_pool_size = to.thread_pool_size;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);

  std::basic_ostringstream<ORTCHAR_T> oss;
  oss << ORT_TSTR("testdata/optional_inputs_ir") << param.ir_version << ORT_TSTR(".onnx");
  Status status;
  std::shared_ptr<Model> model;
  ASSERT_TRUE((status = Model::Load(oss.str(), model, nullptr, DefaultLoggingManager().DefaultLogger())).IsOK())
      << status;
  Graph& graph = model->MainGraph();
  // take a copy as this gets cleared during session state initialization
  InitializedTensorSet initializers = graph.GetAllInitializedTensors();

  ExecutionProviders execution_providers;
  CPUExecutionProviderInfo epi{false};
  status =
      execution_providers.Add(onnxruntime::kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(epi));
  ASSERT_TRUE(status.IsOK()) << status;

  KernelRegistryManager krm;
  status = krm.RegisterKernels(execution_providers);
  ASSERT_TRUE(status.IsOK()) << status;

  DataTransferManager dtm;
  profiling::Profiler profiler;

  SessionOptions sess_options;
  sess_options.enable_mem_pattern = param.enable_mem_pattern;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;

  SessionState session_state(graph, execution_providers, tp.get(), nullptr, dtm,
                             DefaultLoggingManager().DefaultLogger(), profiler, sess_options);

  GraphPartitioner partitioner(krm, execution_providers);
  ASSERT_STATUS_OK(
      partitioner.Partition(
          graph, session_state.GetMutableFuncMgr(),
          [](Graph& graph, bool& modified, const IExecutionProvider& execution_provider,
             const layout_transformation::DebugGraphFn& debug_graph_fn) -> Status {
            AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
            return layout_transformation::TransformLayoutForEP(
                graph, modified, execution_provider, std::move(cpu_allocator), debug_graph_fn);
          },
          sess_options.config_options,
          DefaultLoggingManager().DefaultLogger()));

  ASSERT_STATUS_OK(session_state.FinalizeSessionState(oss.str(), krm));

  const auto& initialized_tensors = session_state.GetInitializedTensors();
  const auto& const_initialized_tensors = session_state.GetConstantInitializedTensors();

  ASSERT_EQ(initializers.size(), initialized_tensors.size())
      << "SessionState should have an entry for all initializers in Graph.";

  if (param.ir_version < 4) {
    ASSERT_EQ(initialized_tensors.size(), const_initialized_tensors.size())
        << "All initializers should be considered constant if IR version < 4.";
  } else {
    const auto& name_to_idx = session_state.GetOrtValueNameIdxMap();

    for (const auto& entry : initializers) {
      int idx;
      ASSERT_STATUS_OK(name_to_idx.GetIdx(entry.first, idx));

      bool found = initialized_tensors.find(idx) != initialized_tensors.cend();
      ASSERT_TRUE(found) << "Missing entry for " << entry.first << " in session state initialized tensors";

      if (graph_utils::IsConstantInitializer(graph, entry.first, false)) {
        found = const_initialized_tensors.find(idx) != const_initialized_tensors.cend();
        ASSERT_TRUE(found) << "Missing entry for " << entry.first << " in session state const initialized tensors";
      }
    }
  }
}

// Test that we allocate memory for an initializer from non-arena memory even if we provide an arena-based allocator
// if the relevant session option config flag is set
// For this test we need to enable the arena-based allocator which is not supported on x86 builds, so
// enable this test only on x64 builds
#if (defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64)) && !defined(USE_MIMALLOC)
TEST(SessionStateTest, TestInitializerMemoryAllocatedUsingNonArenaMemory) {
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  // Part 1: Feature turned ON (i.e.) allocate from non-arena memory
  {
    std::basic_ostringstream<ORTCHAR_T> oss;
    oss << ORT_TSTR("testdata/mul_1.onnx");
    Status status;
    std::shared_ptr<Model> model;
    ASSERT_TRUE((status = Model::Load(oss.str(), model, nullptr, DefaultLoggingManager().DefaultLogger())).IsOK())
        << status;
    Graph& graph = model->MainGraph();

    ExecutionProviders execution_providers;
    CPUExecutionProviderInfo epi{true};  // use an arena-based allocator for this EP
    status = execution_providers.Add(onnxruntime::kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(epi));
    ASSERT_TRUE(status.IsOK()) << status;

    KernelRegistryManager krm;
    status = krm.RegisterKernels(execution_providers);
    ASSERT_TRUE(status.IsOK()) << status;

    DataTransferManager dtm;
    profiling::Profiler profiler;

    SessionOptions sess_options;
    sess_options.enable_mem_pattern = false;
    sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    sess_options.use_deterministic_compute = false;
    sess_options.enable_mem_reuse = true;
    // disable allocating initialized tensor memory from the arena(by default it will be allocated by the arena)
    ASSERT_STATUS_OK(sess_options.config_options.AddConfigEntry(kOrtSessionOptionsUseDeviceAllocatorForInitializers,
                                                                "1"));

    SessionState session_state(graph, execution_providers, nullptr, nullptr, dtm,
                               DefaultLoggingManager().DefaultLogger(), profiler, sess_options);

    // Partition the graph
    GraphPartitioner partitioner(krm, execution_providers);
    ASSERT_STATUS_OK(partitioner.Partition(
        graph, session_state.GetMutableFuncMgr(),
        [&cpu_allocator](Graph& graph, bool& modified, const IExecutionProvider& execution_provider,
                         const layout_transformation::DebugGraphFn& debug_graph_fn) -> Status {
          return layout_transformation::TransformLayoutForEP(graph, modified, execution_provider,
                                                             cpu_allocator, debug_graph_fn);
        },
        sess_options.config_options,
        DefaultLoggingManager().DefaultLogger()));

    ASSERT_STATUS_OK(session_state.FinalizeSessionState(oss.str(), krm));

    // Fetch the CPU arena-allocator from the session state
    OrtMemoryInfo mem_info(CPU, OrtArenaAllocator);
    AllocatorPtr alloc = session_state.GetAllocator(mem_info);
    ASSERT_TRUE(alloc != nullptr);

    // Get stats for the CPU arena-based allocator
    AllocatorStats alloc_stats;
    static_cast<BFCArena*>(alloc.get())->GetStats(&alloc_stats);

    // Assert that we have made 1 Reserve() call (for allocating memory for the sole initializer in the model)
    ASSERT_EQ(alloc_stats.num_reserves, 1);
  }

  // Part 2: Feature turned OFF (i.e.) allocate from arena memory (default behavior)
  {
    std::basic_ostringstream<ORTCHAR_T> oss;
    oss << ORT_TSTR("testdata/mul_1.onnx");
    Status status;
    std::shared_ptr<Model> model;
    ASSERT_TRUE((status = Model::Load(oss.str(), model, nullptr, DefaultLoggingManager().DefaultLogger())).IsOK())
        << status;
    Graph& graph = model->MainGraph();

    ExecutionProviders execution_providers;
    CPUExecutionProviderInfo epi{true};  // use an arena-based allocator for this EP
    status = execution_providers.Add(onnxruntime::kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(epi));
    ASSERT_TRUE(status.IsOK()) << status;

    KernelRegistryManager krm;
    status = krm.RegisterKernels(execution_providers);
    ASSERT_TRUE(status.IsOK()) << status;

    DataTransferManager dtm;
    profiling::Profiler profiler;

    SessionOptions sess_options;
    sess_options.enable_mem_pattern = false;
    sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    sess_options.use_deterministic_compute = false;
    sess_options.enable_mem_reuse = true;

    SessionState session_state(graph, execution_providers, nullptr, nullptr, dtm,
                               DefaultLoggingManager().DefaultLogger(), profiler, sess_options);

    // Partition the graph
    GraphPartitioner partitioner(krm, execution_providers);
    ASSERT_STATUS_OK(partitioner.Partition(
        graph, session_state.GetMutableFuncMgr(),
        [&cpu_allocator](Graph& graph, bool& modified,
                         const IExecutionProvider& execution_provider,
                         const layout_transformation::DebugGraphFn& debug_graph_fn) -> Status {
          return layout_transformation::TransformLayoutForEP(
              graph, modified, execution_provider, cpu_allocator, debug_graph_fn);
        },
        sess_options.config_options,
        DefaultLoggingManager().DefaultLogger()));

    // Finalize the session state
    ASSERT_STATUS_OK(session_state.FinalizeSessionState(oss.str(), krm));

    // Fetch the CPU arena-allocator from the session state
    OrtMemoryInfo mem_info(CPU, OrtArenaAllocator);
    AllocatorPtr alloc = session_state.GetAllocator(mem_info);
    ASSERT_TRUE(alloc != nullptr);

    // Get stats for the CPU arena-based allocator
    AllocatorStats alloc_stats;
    static_cast<BFCArena*>(alloc.get())->GetStats(&alloc_stats);

    // Assert that we have made no Reserve() calls
    ASSERT_EQ(alloc_stats.num_reserves, 0);

    // Assert to ensure an allocation was made for the initializer through the arena allocator (Alloc() was invoked)
    ASSERT_EQ(alloc_stats.num_allocs, 1);
  }
}

#endif

INSTANTIATE_TEST_SUITE_P(SessionStateTests, SessionStateTestP, testing::ValuesIn(param_list));

#ifndef ENABLE_TRAINING_CORE
class PrePackingTestOpKernel : public OpKernel {
 public:
  PrePackingTestOpKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override {
    ORT_UNUSED_PARAMETER(context);
    return Status::OK();
  }

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override {
    ORT_UNUSED_PARAMETER(input_idx);

    weight_packed_ = std::move(prepacked_buffers[0]);
    used_shared_buffers = true;
    ++store_pre_packed_weight_calls_count;
    return Status::OK();
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) override {
    ORT_UNUSED_PARAMETER(tensor);
    ORT_UNUSED_PARAMETER(input_idx);

    size_t weight_packed_len = 8;
    weight_packed_ = IAllocator::MakeUniquePtr<void>(alloc, weight_packed_len, true);
    float* data_weights_packed = reinterpret_cast<float*>(weight_packed_.get());
    data_weights_packed[0] = 1.2345f;
    data_weights_packed[1] = data_weights_packed[0] * 2.f;

    if (prepacked_weights != nullptr) {
      prepacked_weights->buffers_.push_back(std::move(weight_packed_));
      prepacked_weights->buffer_sizes_.push_back(weight_packed_len);
    }

    is_packed = true;
    ++prepack_calls_count;
    return Status::OK();
  }

  int prepack_calls_count = 0;
  int store_pre_packed_weight_calls_count = 0;
  IAllocatorUniquePtr<void> weight_packed_;
};

static void CreateSimpleGraph(Graph& graph) {
  // node creation and placement
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  std::vector<onnxruntime::NodeArg*> inputs;
  onnxruntime::NodeArg input_0_arg("node_0_input_0", &type);
  onnxruntime::NodeArg input_1_arg("node_0_input_1", &type);
  inputs.push_back(&input_0_arg);
  inputs.push_back(&input_1_arg);

  std::vector<onnxruntime::NodeArg*> outputs;
  onnxruntime::NodeArg output_arg("node_0_output_0", &type);
  outputs.push_back(&output_arg);

  graph.AddNode("node_0", "PrePackingTest", "node 0", inputs, outputs);

  // add an initializer
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("node_0_input_1");
  graph.AddInitializedTensor(tensor);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
}

static const ONNX_NAMESPACE::GraphProto CreateSubgraph(bool then_branch) {
  Model model(then_branch ? "If_then" : "If_else", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  const std::string suffix = then_branch ? "0" : "1";

  // graph input has to have type and rank even though it's an outer scope value.
  TypeProto type_float;
  type_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // outer scope values
  auto& if_shared = graph.GetOrCreateNodeArg("if_shared", &type_float);
  auto& if_input = graph.GetOrCreateNodeArg("if_input_" + suffix, &type_float);

  // add so that we don't end up with it being considered a graph input
  graph.AddOuterScopeNodeArg("if_shared");
  graph.AddOuterScopeNodeArg("if_input_" + suffix);

  auto& if_out = graph.GetOrCreateNodeArg("if_output_" + suffix, &type_float);

  inputs = {&if_shared, &if_input};
  outputs = {&if_out};

  graph.AddNode("if_node_" + suffix, "PrePackingTest", "if node " + suffix, inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto& proto = graph.ToGraphProto();

  return proto;
}

static void CreateGraphWithSubgraph(Graph& graph) {
  TypeProto type_float;
  type_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  {
    std::vector<onnxruntime::NodeArg*> inputs;
    onnxruntime::NodeArg input_0_arg("if_input_0", &type_float);
    onnxruntime::NodeArg input_1_arg("if_input_1", &type_float);
    inputs.push_back(&input_0_arg);
    inputs.push_back(&input_1_arg);

    std::vector<onnxruntime::NodeArg*> outputs;
    onnxruntime::NodeArg output_arg("node_0_output_0", &type_float);
    outputs.push_back(&output_arg);

    graph.AddNode("node_0", "PrePackingTest", "node 0", inputs, outputs);
  }

  {
    TypeProto type_bool;
    type_bool.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    type_bool.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    onnxruntime::NodeArg bool_arg("bool_arg", &type_bool);

    std::vector<onnxruntime::NodeArg*> outputs;
    onnxruntime::NodeArg output_arg("output_arg", &type_float);
    outputs.push_back(&output_arg);

    auto& if_node = graph.AddNode("if", "If", "If node", {&bool_arg}, outputs);

    auto then_proto = CreateSubgraph(true);
    auto else_proto = CreateSubgraph(false);
    if_node.AddAttribute("then_branch", then_proto);
    if_node.AddAttribute("else_branch", else_proto);
  }

  // add an initializer
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("if_shared");
  graph.AddInitializedTensor(tensor);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
}

static void PlaceAllNodesToCPUEP(Graph& graph) {
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
    if (node.ContainsSubgraph()) {
      for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
        Graph* subgraph = entry.second;
        PlaceAllNodesToCPUEP(*subgraph);
      }
    }
  }
}

struct PrepackingTestParam {
  bool test_subgraph;
  bool test_prepacking;
};

class SessionStatePrepackingTest : public testing::TestWithParam<PrepackingTestParam> {};
TEST_P(SessionStatePrepackingTest, PrePackingTest) {
  PrepackingTestParam test_param = GetParam();

  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
  ONNX_OPERATOR_SCHEMA(PrePackingTest)
      .SetDoc("Faking Node for PrePacking")
      .Input(0, "Input_0", "input 0", "tensor(float)")
      .Input(1, "Input_1", "input 1", "tensor(float)")
      .Output(0, "output_0", "docstr for output_0.", "tensor(float)");

  ExecutionProviders execution_providers;
  auto cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  ASSERT_STATUS_OK(execution_providers.Add(kCpuExecutionProvider, std::move(cpu_execution_provider)));

  DataTransferManager dtm;
  profiling::Profiler profiler;

  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 11;
  Model model("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());

  // onnxruntime::Model model("graph_main", false, DefaultLoggingManager().DefaultLogger());
  if (test_param.test_subgraph) {
    CreateGraphWithSubgraph(model.MainGraph());
  } else {
    CreateSimpleGraph(model.MainGraph());
  }

  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;
  sess_options.config_options.configurations[kOrtSessionOptionsConfigDisablePrepacking] =
      test_param.test_prepacking ? "0" : "1";

  SessionState session_state(model.MainGraph(),
                             execution_providers,
                             tp.get(),
                             nullptr, /*inter_op_thread_pool*/
                             dtm,
                             DefaultLoggingManager().DefaultLogger(),
                             profiler,
                             sess_options);

  KernelRegistryManager kernel_registry_manager;
  Status status = kernel_registry_manager.RegisterKernels(execution_providers);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  auto kernel_def = KernelDefBuilder().SetName("PrePackingTest").Provider(kCpuExecutionProvider).SinceVersion(1).Build();
  ASSERT_STATUS_OK(kernel_registry->Register(
      KernelCreateInfo(std::move(kernel_def),
                       [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
                         out = std::make_unique<PrePackingTestOpKernel>(info);
                         return Status::OK();
                       })));
  kernel_registry_manager.RegisterKernelRegistry(kernel_registry);

  PlaceAllNodesToCPUEP(model.MainGraph());
  ASSERT_STATUS_OK(session_state.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                      kernel_registry_manager));

  const auto& const_initialized_tensors = session_state.GetConstantInitializedTensors();
  // check prepacking
  ASSERT_EQ(const_initialized_tensors.size(), size_t(test_param.test_prepacking ? 0 : 1));
}

class SessionStateTestSharedInitalizersWithPrePacking : public ::testing::Test {
 protected:
  ExecutionProviders execution_providers;
  std::unordered_map<std::string, int> domain_to_version;
  DataTransferManager dtm;
  profiling::Profiler profiler;
  KernelRegistryManager kernel_registry_manager;
  std::unique_ptr<concurrency::ThreadPool> tp;

  void SetUp() override {
    OrtThreadPoolParams to;
    tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
    ONNX_OPERATOR_SCHEMA(PrePackingTest)
        .SetDoc("Faking Node for PrePacking")
        .Input(0, "Input_0", "input 0", "tensor(float)")
        .Input(1, "Input_1", "input 1", "tensor(float)")
        .Output(0, "output_0", "docstr for output_0.", "tensor(float)");

    auto cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
    ASSERT_STATUS_OK(execution_providers.Add(kCpuExecutionProvider, std::move(cpu_execution_provider)));

    domain_to_version[kOnnxDomain] = 11;

    ASSERT_STATUS_OK(kernel_registry_manager.RegisterKernels(execution_providers));
    std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();

    auto kernel_def = KernelDefBuilder()
                          .SetName("PrePackingTest")
                          .Provider(kCpuExecutionProvider)
                          .SinceVersion(1)
                          .Build();

    ASSERT_STATUS_OK(kernel_registry->Register(
        KernelCreateInfo(std::move(kernel_def),
                         [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status { out =  std::make_unique<PrePackingTestOpKernel>(info); return Status::OK(); })));

    kernel_registry_manager.RegisterKernelRegistry(kernel_registry);
  }
};

// Pre-packing enabled + no shared initializers = no pre-packed weights caching
TEST_F(SessionStateTestSharedInitalizersWithPrePacking, test1) {
  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;
  // Enable pre-packing
  sess_options.config_options.configurations[kOrtSessionOptionsConfigDisablePrepacking] = "0";

  // First session/model
  Model model_1("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateSimpleGraph(model_1.MainGraph());
  PlaceAllNodesToCPUEP(model_1.MainGraph());
  SessionState session_state_1(model_1.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options);

  ASSERT_STATUS_OK(session_state_1.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  const auto* kernel = reinterpret_cast<const PrePackingTestOpKernel*>(session_state_1.GetKernel(0));

  // Assert that a pre-pack call was made and that no mechanism to store weight from shared container was invoked
  ASSERT_EQ(session_state_1.GetNumberOfPrepacksCounter(), static_cast<size_t>(1));
  ASSERT_EQ(kernel->prepack_calls_count, 1);
  ASSERT_EQ(kernel->store_pre_packed_weight_calls_count, 0);

  // Second session/model
  Model model_2("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateSimpleGraph(model_2.MainGraph());
  PlaceAllNodesToCPUEP(model_2.MainGraph());
  SessionState session_state_2(model_2.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options);

  ASSERT_STATUS_OK(session_state_2.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  kernel = reinterpret_cast<const PrePackingTestOpKernel*>(session_state_2.GetKernel(0));

  // Assert that a pre-pack call was made and that no mechanism to store weight from shared container was invoked
  ASSERT_EQ(session_state_2.GetNumberOfPrepacksCounter(), static_cast<size_t>(1));
  ASSERT_EQ(kernel->prepack_calls_count, 1);
  ASSERT_EQ(kernel->store_pre_packed_weight_calls_count, 0);
}

// Pre-packing enabled + shared initializers + no pre-packed weights container = no pre-packed weights caching
TEST_F(SessionStateTestSharedInitalizersWithPrePacking, test2) {
  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;
  // Enable pre-packing
  sess_options.config_options.configurations[kOrtSessionOptionsConfigDisablePrepacking] = "0";

  // Enable shared initializer
  OrtMemoryInfo mem_info(CPU, OrtDeviceAllocator);
  std::vector<float> float_data(1, 1);
  auto value = std::make_unique<OrtValue>();
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),
                       TensorShape(std::vector<int64_t>{1}), reinterpret_cast<void*>(float_data.data()),
                       mem_info, *value);

  ASSERT_STATUS_OK(sess_options.AddInitializer("node_0_input_1", value.get()));

  // First session/model
  Model model_1("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateSimpleGraph(model_1.MainGraph());
  PlaceAllNodesToCPUEP(model_1.MainGraph());
  SessionState session_state_1(model_1.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options);

  ASSERT_STATUS_OK(session_state_1.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  const auto* kernel = reinterpret_cast<const PrePackingTestOpKernel*>(session_state_1.GetKernel(0));

  // Assert that a pre-pack call was made and that no mechanism to store weight from shared container was invoked
  ASSERT_EQ(session_state_1.GetNumberOfPrepacksCounter(), static_cast<size_t>(1));
  ASSERT_EQ(kernel->prepack_calls_count, 1);
  ASSERT_EQ(kernel->store_pre_packed_weight_calls_count, 0);

  // Second session/model
  Model model_2("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateSimpleGraph(model_2.MainGraph());
  PlaceAllNodesToCPUEP(model_2.MainGraph());
  SessionState session_state_2(model_2.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options);

  ASSERT_STATUS_OK(session_state_2.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  kernel = reinterpret_cast<const PrePackingTestOpKernel*>(session_state_2.GetKernel(0));

  // Assert that a pre-pack call was made and that no mechanism to store weight from shared container was invoked
  ASSERT_EQ(session_state_2.GetNumberOfPrepacksCounter(), static_cast<size_t>(1));
  ASSERT_EQ(kernel->prepack_calls_count, 1);
  ASSERT_EQ(kernel->store_pre_packed_weight_calls_count, 0);
}

// Pre-packing enabled + shared initializers + pre-packed weights container = pre-packed weights caching enabled
TEST_F(SessionStateTestSharedInitalizersWithPrePacking, test3) {
  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;
  // Enable pre-packing
  sess_options.config_options.configurations[kOrtSessionOptionsConfigDisablePrepacking] = "0";

  // Enable shared initializer
  OrtMemoryInfo mem_info(CPU, OrtDeviceAllocator);
  std::vector<float> float_data(1, 1);
  auto value = std::make_unique<OrtValue>();
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape(std::vector<int64_t>{1}),
                       reinterpret_cast<void*>(float_data.data()), mem_info, *value);

  ASSERT_STATUS_OK(sess_options.AddInitializer("node_0_input_1", value.get()));

  // Enable pre-packed weights container
  PrepackedWeightsContainer prepacked_weights_container;

  // First session/model
  Model model_1("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateSimpleGraph(model_1.MainGraph());
  PlaceAllNodesToCPUEP(model_1.MainGraph());
  SessionState session_state_1(model_1.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options,
                               &prepacked_weights_container);

  ASSERT_STATUS_OK(session_state_1.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  const auto* kernel = reinterpret_cast<const PrePackingTestOpKernel*>(session_state_1.GetKernel(0));
  // Assert that a pre-pack call was made
  ASSERT_EQ(session_state_1.GetNumberOfPrepacksCounter(), static_cast<size_t>(1));
  ASSERT_EQ(kernel->prepack_calls_count, 1);
  // Assert that we made a call to store pre-packed weight from a shared container
  ASSERT_EQ(kernel->store_pre_packed_weight_calls_count, 1);
  // The weight to be "stored" is the same weight that we got by invoking PrePack() in the step above.
  // Hence, assert that it wasn't a "cached" pre-packed weight (i.e.) pre-packed weight
  // from another instance of the same op_type consuming the same constant initializer.
  ASSERT_EQ(session_state_1.GetUsedSharedPrePackedWeightCounter(), static_cast<size_t>(0));

  // Second session/model
  Model model_2("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateSimpleGraph(model_2.MainGraph());
  PlaceAllNodesToCPUEP(model_2.MainGraph());
  SessionState session_state_2(model_2.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options,
                               &prepacked_weights_container);

  ASSERT_STATUS_OK(session_state_2.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  // Assert that a pre-pack call was made
  ASSERT_EQ(session_state_2.GetNumberOfPrepacksCounter(), static_cast<size_t>(1));
  ASSERT_EQ(kernel->prepack_calls_count, 1);
  // Assert that we made a call to store pre-packed weight from a shared container
  ASSERT_EQ(kernel->store_pre_packed_weight_calls_count, 1);
  // The weight to be "stored" is a "cached" weight (i.e.) a pre-packed weight
  // from another instance of the same op_type consuming the same constant initializer.
  // Assert this.
  ASSERT_EQ(session_state_2.GetUsedSharedPrePackedWeightCounter(), static_cast<size_t>(1));
}

// Pre-packing enabled + shared initializers +
// pre-packed weights container + subgraphs =
// caching enabled in pre-packed weights used in subgraphs
TEST_F(SessionStateTestSharedInitalizersWithPrePacking, test4) {
  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;
  // Enable pre-packing
  sess_options.config_options.configurations[kOrtSessionOptionsConfigDisablePrepacking] = "0";

  // Enable shared initializer
  OrtMemoryInfo mem_info(CPU, OrtDeviceAllocator);
  std::vector<float> float_data(1, 1);
  auto value = std::make_unique<OrtValue>();
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape(std::vector<int64_t>{1}),
                       reinterpret_cast<void*>(float_data.data()), mem_info, *value);

  ASSERT_STATUS_OK(sess_options.AddInitializer("if_shared", value.get()));

  // Enable pre-packed weights container
  PrepackedWeightsContainer prepacked_weights_container;

  // First session/model
  Model model_1("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateGraphWithSubgraph(model_1.MainGraph());
  PlaceAllNodesToCPUEP(model_1.MainGraph());
  SessionState session_state_1(model_1.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options,
                               &prepacked_weights_container);

  ASSERT_STATUS_OK(session_state_1.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  // At the main graph level, there should be no pre-packing calls as there are
  // no initializers (shared or otherwise) consumed by any nodes in the main graph
  ASSERT_EQ(session_state_1.GetNumberOfPrepacksCounter(), static_cast<size_t>(0));

  auto if_index_1 = 1;
  if (session_state_1.GetKernel(0)->Node().OpType() == "If") {
    if_index_1 = 0;
  }

  const auto& subgraph_session_states = session_state_1.GetSubgraphSessionStateMap();
  const auto& if_node_session_states = subgraph_session_states.at(if_index_1);
  const auto& session_state_1_then_branch_session_state = *if_node_session_states.at("then_branch");
  const auto& session_state_1_else_branch_session_state = *if_node_session_states.at("else_branch");

  auto if_node_branches_prepack_counter_1 =
      session_state_1_then_branch_session_state.GetNumberOfPrepacksCounter() +
      session_state_1_else_branch_session_state.GetNumberOfPrepacksCounter();

  // We should be seeing 2 pre-pack calls in the "If" node (one in each subgraph)
  ASSERT_EQ(if_node_branches_prepack_counter_1, static_cast<size_t>(2));

  auto if_node_branches_shared_prepack_counter_1 =
      session_state_1_then_branch_session_state.GetUsedSharedPrePackedWeightCounter() +
      session_state_1_else_branch_session_state.GetUsedSharedPrePackedWeightCounter();

  // We should only be seeing 1 shared pre-pack weights usage in the "If" node
  // Either the "then branch" or "else branch" will be using the shared version
  // depending on which branch writes to the shared container
  ASSERT_EQ(if_node_branches_shared_prepack_counter_1, static_cast<size_t>(1));

  // Second session/model
  Model model_2("graph_main", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                DefaultLoggingManager().DefaultLogger());

  CreateGraphWithSubgraph(model_2.MainGraph());
  PlaceAllNodesToCPUEP(model_2.MainGraph());
  SessionState session_state_2(model_2.MainGraph(),
                               execution_providers,
                               tp.get(),
                               nullptr, /*inter_op_thread_pool*/
                               dtm,
                               DefaultLoggingManager().DefaultLogger(),
                               profiler,
                               sess_options,
                               &prepacked_weights_container);

  ASSERT_STATUS_OK(session_state_2.FinalizeSessionState(std::basic_string<PATH_CHAR_TYPE>(),
                                                        kernel_registry_manager));

  // At the main graph level, there should be no pre-packing calls as there are
  // no initializers (shared or otherwise) consumed by any nodes in the main graph
  ASSERT_EQ(session_state_2.GetNumberOfPrepacksCounter(), static_cast<size_t>(0));

  auto if_index_2 = 1;
  if (session_state_2.GetKernel(0)->Node().OpType() == "If") {
    if_index_2 = 0;
  }

  const auto& subgraph_session_states_2 = session_state_2.GetSubgraphSessionStateMap();
  const auto& if_node_session_states_2 = subgraph_session_states_2.at(if_index_2);
  const auto& session_state_2_then_branch_session_state = *if_node_session_states_2.at("then_branch");
  const auto& session_state_2_else_branch_session_state = *if_node_session_states_2.at("else_branch");

  auto if_node_branches_prepack_counter_2 =
      session_state_2_then_branch_session_state.GetNumberOfPrepacksCounter() +
      session_state_2_else_branch_session_state.GetNumberOfPrepacksCounter();

  // We should be seeing 2 pre-pack calls in the "If" node (one in each subgraph)
  ASSERT_EQ(if_node_branches_prepack_counter_2, static_cast<size_t>(2));

  auto if_node_branches_shared_prepack_counter_2 =
      session_state_2_then_branch_session_state.GetUsedSharedPrePackedWeightCounter() +
      session_state_2_else_branch_session_state.GetUsedSharedPrePackedWeightCounter();

  // We should be seeing 2 shared pre-pack weights calls in the "If" node
  // Both branches will be using the shared version coming from the first model.
  ASSERT_EQ(if_node_branches_shared_prepack_counter_2, static_cast<size_t>(2));
}

INSTANTIATE_TEST_SUITE_P(SessionStateTests,
                         SessionStatePrepackingTest,
                         testing::Values(PrepackingTestParam{false, false},
                                         PrepackingTestParam{false, true},
                                         PrepackingTestParam{true, false},
                                         PrepackingTestParam{true, true}));
#endif

}  // namespace test
}  // namespace onnxruntime
