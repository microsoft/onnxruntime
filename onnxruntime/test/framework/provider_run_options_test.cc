#include <core/framework/op_kernel_context_internal.h>
#include <core/session/inference_session.h>
#include "core/framework/compute_capability.h"
#include <test/framework/dummy_allocator.h>
#include <test/framework/test_utils.h>
#include "core/framework/utils.h"
#include <test/providers/provider_test_utils.h>
#include <gtest/gtest.h>

using namespace std;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

namespace test {

struct ProviderContextMock {
  const char* val;
};

class DummyProvider : public IExecutionProvider {
 public:
  static constexpr const char* kDummyProviderType = "DummyProvider";
  DummyProvider() : IExecutionProvider{kDummyProviderType}, run_context(nullptr) {
    InsertAllocator(std::make_unique<DummyAllocator>());
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
    return kernel_registry;
  };

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override {
    std::vector<std::unique_ptr<ComputeCapability>> result;
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    for (auto& node : graph.Nodes()) {
      sub_graph->nodes.push_back(node.Index());
    }
    auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
    meta_def->name = "MatMul";
    meta_def->domain = "ProviderRunOptionMock";
    meta_def->inputs = {"X"};
    meta_def->outputs = {"Z"};
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    sub_graph->SetMetaDef(meta_def);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    return result;
  }

  common::Status Compile(const std::vector<onnxruntime::Node*>&,
                         std::vector<NodeComputeInfo>& node_compute_funcs) {
    NodeComputeInfo compute_info;

    compute_info.create_state_func = [=](ComputeContext*, FunctionState*) {
      return 0;
    };

    // Create compute function
    compute_info.compute_func = [this](FunctionState, const OrtCustomOpApi*, OrtKernelContext* context) {
      auto internal_context = reinterpret_cast<OpKernelContextInternal*>(context);
      auto extra_options = internal_context->GetRunOptions().extra_options;
      run_context = utils::GetProviderRunOptions(extra_options, kDummyProviderType);
      return Status::OK();
    };

    // Release function state
    compute_info.release_state_func = [](FunctionState) {
    };

    node_compute_funcs.push_back(compute_info);
    return Status::OK();
  }

  const char* GetContext() const {
    auto context = reinterpret_cast<ProviderContextMock*>(this->run_context);
    return context->val;
  }

 private:
  void* run_context;
};

TEST(ProviderRunOptionsTest, BasicTest) {
  SessionOptions so;
  so.session_logid = "ProviderRunOptionsTest.BasicTest";

  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  unordered_map<string, void*> provider_run_options_map;

  // Insert different provider specific contexts and the correct one reaches to the dummy provider
  ProviderContextMock dummy_provider_context = {"Hi I'm a DummyProvider context"};
  provider_run_options_map.insert(
      std::make_pair(DummyProvider::kDummyProviderType, &dummy_provider_context));
  string cpu_context = "Hi I'm a CpuExecutionProvider context";
  provider_run_options_map.insert(
      std::make_pair(kCpuExecutionProvider, &cpu_context));
  run_options.extra_options = provider_run_options_map;
    
  auto logger = DefaultLoggingManager().CreateLogger("GraphTest");
  onnxruntime::Model model("graph_1", false, *logger);
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  std::vector<onnxruntime::NodeArg*> input_defs;
  auto& input_arg = graph.GetOrCreateNodeArg("X", &float_tensor);
  input_defs.push_back(&input_arg);

  std::vector<onnxruntime::NodeArg*> output_defs;
  auto& output_arg = graph.GetOrCreateNodeArg("Z", &float_tensor);
  output_defs.push_back(&output_arg);

  // Create a simple model
  graph.AddNode("node1", "Clip", "Clip", input_defs, output_defs, nullptr, onnxruntime::kOnnxDomain);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "provider_run_options_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  InferenceSession session_object{so, GetEnvironment()};
  auto dummy_provider = std::make_unique<DummyProvider>();
  auto* p_dummy_provider = dummy_provider.get();
  session_object.RegisterExecutionProvider(std::move(dummy_provider));

  ASSERT_TRUE(session_object.Load(model_file_name).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  std::vector<int64_t> dims_mul_x = {1};
  std::vector<float> values_mul_x = {1.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::vector<std::string> output_names;
  output_names.push_back("Z");

  std::vector<OrtValue> fetches;

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  ASSERT_NE(p_dummy_provider, nullptr);
  ASSERT_EQ(dummy_provider_context.val, p_dummy_provider->GetContext());
}
}  // namespace test
}  // namespace onnxruntime