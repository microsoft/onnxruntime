// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/stateful_execution_provider.h"
#include "core/framework/stateful_provider.h"
#include "core/graph/graph_viewer.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/stateful_execution_test_utils.h"

namespace onnxruntime::test {
namespace {
constexpr const char* kTestProvider = "StatefulTestExecutionProvider";
constexpr const char* kUnextendedProvider = "StatefulUnextendedExecutionProvider";
constexpr uint64_t kAll = (uint64_t{1} << 7) - 1;

struct Probe {
  uint64_t operations = kAll;
  uint32_t version = 1;
  uint64_t sampler_types = (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
                           (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  int device_id = 0;
  uint32_t memory_types = 1;
  int fail_operation = -1;
  OrtErrorCode error = ORT_FAIL;
  bool bad_result = false;
  bool delegate_greedy = false;
  bool double_reference = false;
  bool allocate_directly = false;
  size_t destroyed = 0;
  std::array<size_t, 7> calls{};
};

struct ProviderContext {
  std::atomic<size_t> references{1};
  std::shared_ptr<Probe> probe;
  explicit ProviderContext(std::shared_ptr<Probe> value) : probe(std::move(value)) {}
};

void ORT_API_CALL RetainContext(void* pointer) noexcept {
  static_cast<ProviderContext*>(pointer)->references.fetch_add(1);
}

void ORT_API_CALL ReleaseContext(void* pointer) noexcept {
  auto* context = static_cast<ProviderContext*>(pointer);
  if (context->references.fetch_sub(1) == 1) {
    ++context->probe->destroyed;
    delete context;
  }
}

OrtStatus* ORT_API_CALL CacheOperation(void* pointer, const OrtStatefulCacheOperationRequest* request,
                                       OrtStatefulCacheOperationResult* out) noexcept {
  auto& probe = *static_cast<ProviderContext*>(pointer)->probe;
  ++probe.calls[request->operation];
  if (probe.fail_operation == request->operation) {
    return request->api->CreateStatus(probe.error, "injected stateful cache failure");
  }
  if (probe.allocate_directly && request->operation == ORT_STATEFUL_CACHE_INITIALIZE) {
    struct Call {
      const OrtStatefulCacheOperationRequest& request;
      OrtStatefulCacheOperationResult& out;
    } call{*request, *out};
    return StatefulCpuBoundary(&call, [](void* pointer) -> Status {
      auto& call = *static_cast<Call*>(pointer);
      const auto& input = call.request;
      TensorShapeVector shape(input.spec->shape, input.spec->shape + input.spec->rank);
      shape[input.spec->sequence_axis] = input.requested_capacity;
      OrtValue* storage = nullptr;
      ORT_RETURN_IF_ERROR(ToStatusAndRelease(input.api->CreateTensorAsOrtValue(
          input.allocator, shape.data(), shape.size(), input.spec->element_type, &storage)));
      call.out = {sizeof(call.out), storage, 0, input.requested_capacity,
                  input.requested_capacity ? 1u : 0u, 0, 0};
      return Status::OK();
    });
  }
  auto* status = request->cpu_reference(request->cpu_context, out);
  if (status) {
    return status;
  }
  if (probe.double_reference) {
    return request->cpu_reference(request->cpu_context, out);
  }
  if (probe.bad_result) {
    ++out->length;
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL GreedyOperation(void* pointer, const OrtStatefulGreedyRequest* request,
                                        int64_t* token) noexcept {
  auto& probe = *static_cast<ProviderContext*>(pointer)->probe;
  ++probe.calls[ORT_STATEFUL_SAMPLE_GREEDY];
  if (probe.fail_operation == ORT_STATEFUL_SAMPLE_GREEDY) {
    return request->api->CreateStatus(probe.error, "injected greedy failure");
  }
  if (probe.bad_result) {
    *token = -1;
    return nullptr;
  }
  if (probe.delegate_greedy) {
    auto* status = request->cpu_reference(request->cpu_context, token);
    return status || !probe.double_reference ? status : request->cpu_reference(request->cpu_context, token);
  }
  struct Call {
    const OrtStatefulGreedyRequest& request;
    int64_t& token;
  } call{*request, *token};
  return StatefulCpuBoundary(&call, [](void* argument) -> Status {
    auto& call = *static_cast<Call*>(argument);
    const auto& tensor = call.request.logits->Get<Tensor>();
    const auto& shape = tensor.Shape();
    const int64_t width = shape[shape.NumDimensions() - 1];
    const int64_t row = call.request.has_position    ? call.request.position
                        : shape.NumDimensions() == 3 ? shape[1] - 1
                                                     : 0;
    auto value = [&](int64_t i) {
      return tensor.IsDataType<float>() ? tensor.Data<float>()[i] : static_cast<float>(tensor.Data<MLFloat16>()[i]);
    };
    for (int64_t i = 0; i < shape.Size(); ++i) {
      ORT_RETURN_IF(std::isnan(value(i)), "test provider rejects NaN");
    }
    int64_t best = 0;
    for (int64_t i = 1; i < width; ++i) {
      if (value(row * width + i) > value(row * width + best)) {
        best = i;
      }
    }
    call.token = best;
    return Status::OK();
  });
}

class StatefulTestProvider final : public IExecutionProvider {
 public:
  explicit StatefulTestProvider(std::shared_ptr<Probe> probe)
      : IExecutionProvider(kTestProvider), context_(new ProviderContext(std::move(probe))) {
    descriptor_ = *GetCpuStatefulProviderDescriptor();
    descriptor_.capabilities.operations = context_->probe->operations;
    descriptor_.capabilities.version = context_->probe->version;
    descriptor_.capabilities.sampler_element_types = context_->probe->sampler_types;
    descriptor_.capabilities.device_id = context_->probe->device_id;
    descriptor_.capabilities.memory_types = context_->probe->memory_types;
    descriptor_.context = context_;
    descriptor_.CacheOperation = (descriptor_.capabilities.operations & 63) ? &CacheOperation : nullptr;
    descriptor_.Greedy = (descriptor_.capabilities.operations & 64) ? &GreedyOperation : nullptr;
    descriptor_.Retain = &RetainContext;
    descriptor_.Release = &ReleaseContext;
  }
  ~StatefulTestProvider() override { ReleaseContext(context_); }
  const OrtStatefulProviderDescriptor* GetStatefulProvider() const noexcept override { return &descriptor_; }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StatefulTestProvider);

 private:
  ProviderContext* context_;
  OrtStatefulProviderDescriptor descriptor_{};
};

class UnextendedProvider final : public IExecutionProvider {
 public:
  UnextendedProvider() : IExecutionProvider(kUnextendedProvider) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(UnextendedProvider);
};

template <typename T>
Ort::Value Tensor(std::initializer_list<int64_t> shape, std::initializer_list<T> values) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto value = Ort::Value::CreateTensor<T>(allocator, shape.begin(), shape.size());
  if (values.size()) {
    std::copy(values.begin(), values.end(), value.template GetTensorMutableData<T>());
  }
  return value;
}

OrtStatefulProviderSelection Selection(OrtStatefulProviderMode mode = ORT_STATEFUL_PROVIDER_REQUIRE,
                                       const char* name = kTestProvider) {
  return {sizeof(OrtStatefulProviderSelection), mode,
          mode == ORT_STATEFUL_PROVIDER_CPU ? nullptr : name, OrtMemoryInfoDeviceType_CPU, 0};
}

Ort::StatefulValue Output(const Ort::StatefulStepResult& result, const char* name,
                          OrtStatefulOutputGroup group = ORT_STATEFUL_SELECTED_OUTPUTS) {
  for (const auto& value : result.GetOutputs(group)) {
    if (value.name == name) {
      return value.value;
    }
  }
  ORT_CXX_API_THROW(std::string("Missing output: ") + name, ORT_FAIL);
}

class StatefulProviderTest : public testing::TestWithParam<bool> {
 protected:
  Ort::Session Session(std::shared_ptr<Probe> probe, ONNX_NAMESPACE::ModelProto model = stateful::MakeFeedbackModel()) {
    SessionOptions options;
    options.intra_op_param.thread_pool_size = 1;
    options.inter_op_param.thread_pool_size = 1;
    options.graph_optimization_level = TransformerLevel::Default;
    if (GetParam()) {
      ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    }
    auto session = std::make_unique<InferenceSession>(options, GetEnvironment());
    ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::make_unique<StatefulTestProvider>(std::move(probe))));
    ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::make_unique<UnextendedProvider>()));
    if (GetParam()) {
      ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    }
    const auto bytes = model.SerializeAsString();
    ORT_THROW_IF_ERROR(session->Load(bytes.data(), static_cast<int>(bytes.size())));
    ORT_THROW_IF_ERROR(session->Initialize());
    if (GetParam()) {
      for (const auto& node : session->GetSessionState().GetGraphViewer().Nodes()) {
        EXPECT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
      }
    }
    return Ort::Session{reinterpret_cast<OrtSession*>(session.release())};
  }
};

TEST_P(StatefulProviderTest, CapabilitySelectionSamplerAndMissingOverrides) {
  auto probe = std::make_shared<Probe>();
  auto session = Session(probe);
  auto caps = Ort::QueryStatefulProviderCapabilities(session, kTestProvider);
  EXPECT_EQ(caps.operations, kAll);
  EXPECT_EQ(Ort::QueryStatefulProviderCapabilities(session, kUnextendedProvider).operations, 0u);
  EXPECT_EQ(Ort::QueryStatefulProviderCapabilities(session, kCpuExecutionProvider).operations, kAll);
  EXPECT_THROW(Ort::QueryStatefulProviderCapabilities(session, "missing"), Ort::Exception);
  OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(config);
  auto logits = Tensor<float>({1, 2, 3}, {1.f, 8.f, 2.f, 3.f, 4.f, 4.f});
  EXPECT_EQ(sampler.GetDispatchInfo().origin, ORT_STATEFUL_DISPATCH_NOT_INVOKED);
  EXPECT_EQ(sampler.Sample(logits), 1);
  EXPECT_EQ(sampler.GetDispatchInfo().origin, ORT_STATEFUL_DISPATCH_CPU);
  sampler.SetProvider(Selection(), session);
  EXPECT_EQ(sampler.Sample(logits), 1);
  auto info = sampler.GetDispatchInfo();
  EXPECT_EQ(info.provider_name, kTestProvider);
  EXPECT_EQ(info.provider_calls, 1u);
  EXPECT_EQ(info.cpu_calls, 0u);
  EXPECT_EQ(info.cpu_delegations, 0u);
  config.has_logits_position = 1;
  config.logits_position = 0;
  sampler.SetConfig(config).SetProvider(Selection(), session);
  EXPECT_EQ(sampler.Sample(logits), 1);
  auto half = Tensor<Ort::Float16_t>(
      {1, 3}, {Ort::Float16_t(-1.f), Ort::Float16_t(3.f), Ort::Float16_t(3.f)});
  EXPECT_EQ(sampler.Sample(half), 1);
  auto nan = Tensor<float>({1, 2, 2}, {std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, 2.f});
  EXPECT_THROW(sampler.Sample(nan), Ort::Exception);
  probe->delegate_greedy = true;
  EXPECT_EQ(sampler.Sample(logits), 1);
  EXPECT_EQ(sampler.GetDispatchInfo().cpu_delegations, 1u);
  probe->double_reference = true;
  EXPECT_THROW(sampler.Sample(logits), Ort::Exception);
  probe->double_reference = false;
  probe->delegate_greedy = false;
  probe->fail_operation = ORT_STATEFUL_SAMPLE_GREEDY;
  probe->error = ORT_NOT_IMPLEMENTED;
  sampler.SetProvider(Selection(ORT_STATEFUL_PROVIDER_PREFER), session);
  EXPECT_THROW(sampler.Sample(logits), Ort::Exception);
  EXPECT_EQ(sampler.GetDispatchInfo().cpu_calls, 0u);
  EXPECT_EQ(sampler.GetDispatchInfo().failures, 1u);
  probe->fail_operation = -1;
  probe->bad_result = true;
  EXPECT_THROW(sampler.Sample(logits), Ort::Exception);
  probe->bad_result = false;
  auto preferred = Selection(ORT_STATEFUL_PROVIDER_PREFER, kUnextendedProvider);
  sampler.SetProvider(preferred, session);
  EXPECT_EQ(sampler.Sample(logits), 1);
  info = sampler.GetDispatchInfo();
  EXPECT_EQ(info.origin, ORT_STATEFUL_DISPATCH_CPU_FALLBACK);
  EXPECT_EQ(info.provider_name, kCpuExecutionProvider);
  EXPECT_EQ(info.requested_provider_name, kUnextendedProvider);
  auto automatic = Selection();
  automatic.device_id = -1;
  sampler.SetProvider(automatic, session);
  EXPECT_EQ(sampler.GetDispatchInfo().device_id, 0);
  automatic.device_id = 1;
  EXPECT_THROW(sampler.SetProvider(automatic, session), Ort::Exception);
  auto ambiguous_probe = std::make_shared<Probe>();
  ambiguous_probe->device_id = -1;
  auto ambiguous_session = Session(ambiguous_probe);
  automatic.device_id = -1;
  EXPECT_THROW(sampler.SetProvider(automatic, ambiguous_session), Ort::Exception);
  EXPECT_THROW(sampler.SetProvider(Selection(ORT_STATEFUL_PROVIDER_REQUIRE, kUnextendedProvider), session),
               Ort::Exception);
  sampler.SetProvider(Selection(ORT_STATEFUL_PROVIDER_REQUIRE, kCpuExecutionProvider), session);
  EXPECT_EQ(sampler.Sample(logits), 1);
  EXPECT_EQ(sampler.GetDispatchInfo().cpu_delegations, 1u);
  session = Ort::Session(nullptr);
  EXPECT_EQ(sampler.Sample(logits), 1);
  EXPECT_EQ(info.requested_provider_name, kUnextendedProvider);
}

TEST_P(StatefulProviderTest, PartialCacheFallbackFailureRecoveryAndRetainedContext) {
  auto probe = std::make_shared<Probe>();
  probe->operations = uint64_t{1} << ORT_STATEFUL_CACHE_APPEND;
  auto session = Session(probe);
  const int64_t dimensions[4]{1, 1, 0, 1};
  const OrtStatefulKVCacheSpec spec{"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dimensions, 4, 2, 1};
  const OrtStatefulKVCacheConfig config{sizeof(config), &spec, 1};
  Ort::KVCacheOptions options(config);
  options.SetProvider(Selection(ORT_STATEFUL_PROVIDER_PREFER), session);
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::KVCacheManager manager(options, allocator);
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_INITIALIZE).origin, ORT_STATEFUL_DISPATCH_CPU_FALLBACK);
  auto full = Tensor<float>({1, 1, 1, 1}, {3.f});
  manager.Update("key", full, ORT_KV_CACHE_SHARE);
  auto retained = manager.GetView("key").GetInfo();
  auto delta = Tensor<float>({1, 1, 1, 1}, {4.f});
  manager.Update("key", delta, ORT_KV_CACHE_APPEND);
  retained = manager.GetView("key").GetInfo();
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND).provider_calls, 1u);
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND).cpu_delegations, 1u);
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_SHARE).cpu_calls, 1u);
  probe->double_reference = true;
  EXPECT_THROW(manager.Update("key", delta, ORT_KV_CACHE_APPEND), Ort::Exception);
  manager.Reset();
  probe->double_reference = false;
  probe->fail_operation = ORT_STATEFUL_CACHE_APPEND;
  probe->error = ORT_NOT_IMPLEMENTED;
  EXPECT_THROW(manager.Update("key", delta, ORT_KV_CACHE_APPEND), Ort::Exception);
  EXPECT_TRUE(manager.RequiresReset());
  EXPECT_THROW(manager.Update("key", delta, ORT_KV_CACHE_APPEND), Ort::Exception);
  EXPECT_THROW(manager.GetView("key"), Ort::Exception);
  EXPECT_THROW(manager.Reserve("key", 9), Ort::Exception);
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND).cpu_calls, 0u);
  manager.Reset();
  EXPECT_FALSE(manager.RequiresReset());
  probe->fail_operation = -1;
  probe->bad_result = true;
  EXPECT_THROW(manager.Update("key", delta, ORT_KV_CACHE_APPEND), Ort::Exception);
  EXPECT_TRUE(manager.RequiresReset());
  manager.Reset();
  probe->bad_result = false;
  manager.Update("key", delta, ORT_KV_CACHE_COPY);
  manager.Reserve("key", 7);
  EXPECT_EQ(manager.GetView("key").GetInfo().capacity, 7);
  auto required = Selection();
  options.SetProvider(required, session);
  EXPECT_THROW(Ort::KVCacheManager(options, allocator), Ort::Exception);
  options.SetConfig(config);
  EXPECT_EQ(Ort::KVCacheManager(options, allocator).GetDispatchInfo(ORT_STATEFUL_CACHE_INITIALIZE).origin,
            ORT_STATEFUL_DISPATCH_CPU);
  options.SetProvider(Selection(ORT_STATEFUL_PROVIDER_REQUIRE, kCpuExecutionProvider), session);
  EXPECT_EQ(Ort::KVCacheManager(options, allocator).GetDispatchInfo(ORT_STATEFUL_CACHE_INITIALIZE).cpu_delegations, 1u);
  options.SetProvider(Selection(ORT_STATEFUL_PROVIDER_CPU));
  manager = Ort::KVCacheManager(nullptr);
  options = Ort::KVCacheOptions(nullptr);
  session = Ort::Session(nullptr);
  EXPECT_EQ(probe->destroyed, 0u);
  // The retained tensor keeps the originating provider context alive.
  EXPECT_EQ(retained.value.GetTensorData<float>()[0], 3.f);
  retained.value = Ort::StatefulValue(nullptr);
  EXPECT_EQ(probe->destroyed, 1u);
}

ONNX_NAMESPACE::ModelProto GenerationModel() {
  auto model = stateful::MakeDecoderModel();
  auto& graph = *model.mutable_graph();
  graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  graph.mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  stateful::SetTensorType(*graph.mutable_input(0), "token", ONNX_NAMESPACE::TensorProto_DataType_INT64, {1, -1});
  stateful::SetTensorType(*graph.mutable_output(0), "logits", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  for (auto& initializer : *graph.mutable_initializer()) {
    if (initializer.name() == "key_shape") {
      initializer.clear_int64_data();
      for (int64_t value : {1, 1, -1, 1}) {
        initializer.add_int64_data(value);
      }
    }
  }
  return model;
}

TEST_P(StatefulProviderTest, GenerationChainIndependentPoliciesAndResetFailure) {
  for (int64_t budget : {0, 2}) {
    auto probe = std::make_shared<Probe>();
    probe->allocate_directly = true;
    probe->memory_types = 3;  // Metadata-only GPU request below must fail before any callback.
    auto session = Session(probe, GenerationModel());
    const int64_t dimensions[4]{1, 1, 0, 1};
    const OrtStatefulKVCacheBinding cache{
        {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dimensions, 4, 2, 2},
        "past_key",
        "present_key",
        ORT_KV_CACHE_SHARE};
    const char* names[1]{"token"};
    const char* outputs[1]{"logits"};
    const OrtStatefulExitCondition prefill_exits[2]{
        {ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0},
        {ORT_STATEFUL_MAX_GENERATED_TOKENS, budget, nullptr, nullptr, 0}};
    const OrtStatefulExitCondition decode_exit{ORT_STATEFUL_MAX_GENERATED_TOKENS, budget, nullptr, nullptr, 0};
    OrtStatefulGenerationConfig generation{sizeof(generation), ORT_STATEFUL_PREFILL,
                                           "token", "logits", nullptr, nullptr, 0, 0};
    OrtStatefulLoopConfig config{};
    config.struct_size = sizeof(config);
    config.finite_input_names = names;
    config.num_finite_inputs = 1;
    config.output_names = outputs;
    config.num_outputs = 1;
    config.exit_conditions = prefill_exits;
    config.num_exit_conditions = 2;
    config.kv_caches = &cache;
    config.num_kv_caches = 1;
    config.generation = &generation;
    Ort::SessionLoopOptions prefill_options(config);
    prefill_options.SetProvider(ORT_STATEFUL_FEATURE_CACHE, Selection());
    prefill_options.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, Selection());
    Ort::SessionLoop prefill(session, prefill_options);
    auto gpu = Selection();
    gpu.device_type = OrtMemoryInfoDeviceType_GPU;
    Ort::SessionLoopOptions unsupported(config);
    unsupported.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, gpu);
    EXPECT_THROW(Ort::SessionLoop(session, unsupported), Ort::Exception);
    auto standalone_prompt = Tensor<int64_t>({1, 1}, {0});
    const OrtStatefulNamedValue standalone_value{"token", standalone_prompt};
    const OrtStatefulInputBatch batch{&standalone_value, 1};
    OrtStatefulInputs standalone_inputs{};
    standalone_inputs.struct_size = sizeof(standalone_inputs);
    standalone_inputs.batches = &batch;
    standalone_inputs.num_batches = 1;
    auto standalone = prefill.CreateExecution(standalone_inputs);
    standalone.Run();
    EXPECT_EQ(standalone.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY).provider_calls,
              budget ? 1u : 0u);
    EXPECT_EQ(standalone.GetDispatchInfo(ORT_STATEFUL_FEATURE_CACHE, ORT_STATEFUL_CACHE_INITIALIZE).provider_calls, 1u);
    generation.phase = ORT_STATEFUL_DECODE;
    config.num_finite_inputs = 0;
    config.exit_conditions = &decode_exit;
    config.num_exit_conditions = 1;
    Ort::SessionLoop decode(session, Ort::SessionLoopOptions(config));
    const OrtStatefulInputBinding input_binding{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
    OrtStatefulChainComponent components[2]{};
    components[0].name = "prefill";
    components[0].loop = prefill;
    components[0].inputs = &input_binding;
    components[0].num_inputs = 1;
    components[1].name = "decode";
    components[1].loop = decode;
    const OrtStatefulChainOutput output{"tokens", 0, nullptr, ORT_STATEFUL_GENERATED_TOKEN_IDS};
    const OrtStatefulChainConfig chain_config{sizeof(chain_config), components, 2, &output, 1};
    Ort::SessionChainOptions chain_options(chain_config);
    EXPECT_THROW(Ort::SessionChain{chain_options}, Ort::Exception);
    chain_options.SetComponentProvider(1, ORT_STATEFUL_FEATURE_CACHE, Selection());
    chain_options.SetComponentProvider(1, ORT_STATEFUL_FEATURE_SAMPLER, Selection(ORT_STATEFUL_PROVIDER_CPU));
    Ort::SessionChain chain(chain_options);
    auto prompt = Tensor<int64_t>({1, 1}, {0});
    const OrtStatefulNamedValue input{"prompt", prompt};
    OrtStatefulInputs inputs{};
    inputs.struct_size = sizeof(inputs);
    inputs.inputs = &input;
    inputs.num_inputs = 1;
    auto execution = chain.CreateExecution(inputs);
    auto result = execution.Run();
    auto tokens = Output(result, "tokens");
    EXPECT_EQ(tokens.GetTensorTypeAndShapeInfo().GetShape(), std::vector<int64_t>{budget});
    if (budget) {
      EXPECT_EQ(tokens.GetTensorData<int64_t>()[0], 1);
      EXPECT_EQ(tokens.GetTensorData<int64_t>()[1], 2);
    }
    EXPECT_EQ(probe->calls[ORT_STATEFUL_SAMPLE_GREEDY], budget ? 2u : 0u);
    EXPECT_EQ(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 1).cpu_calls,
              budget ? 1u : 0u);
    EXPECT_EQ(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_CACHE, ORT_STATEFUL_CACHE_SHARE).provider_calls,
              static_cast<size_t>(budget));
    EXPECT_THROW(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_CACHE, ORT_STATEFUL_SAMPLE_GREEDY), Ort::Exception);
    EXPECT_THROW(execution.GetStagingInfo(2), Ort::Exception);
    auto staging = execution.GetStagingInfo(0);
    EXPECT_EQ(staging.cpu_snapshot_copy_bytes, staging.cpu_output_bytes);
    EXPECT_EQ(staging.cpu_output_bytes != 0, budget != 0);
    probe->fail_operation = ORT_STATEFUL_CACHE_RESET;
    EXPECT_THROW(execution.Reset(), Ort::Exception);
    EXPECT_TRUE(execution.RequiresReset());
    EXPECT_THROW(execution.Step(), Ort::Exception);
    probe->fail_operation = -1;
    execution.Reset();
    EXPECT_FALSE(execution.RequiresReset());
    EXPECT_EQ(tokens.GetTensorTypeAndShapeInfo().GetElementCount(), static_cast<size_t>(budget));
  }
}

TEST_P(StatefulProviderTest, FeedbackStagingAndArgumentValidation) {
  auto session = Session(std::make_shared<Probe>());
  const char* finite[1]{"input_value"};
  const char* outputs[1]{"present_value"};
  const OrtStatefulFeedbackBinding feedback{"past_value", "present_value"};
  const OrtStatefulExitCondition exit{ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0};
  OrtStatefulLoopConfig config{};
  config.struct_size = sizeof(config);
  config.finite_input_names = finite;
  config.num_finite_inputs = 1;
  config.feedback = &feedback;
  config.num_feedback = 1;
  config.output_names = outputs;
  config.num_outputs = 1;
  config.exit_conditions = &exit;
  config.num_exit_conditions = 1;
  Ort::SessionLoopOptions options(config);
  options.SetProvider(ORT_STATEFUL_FEATURE_CACHE, Selection());
  EXPECT_THROW(Ort::SessionLoop(session, options), Ort::Exception);
  options.SetConfig(config);
  Ort::SessionLoop loop(session, options);
  auto seed = Tensor<float>({}, {0.f}), one = Tensor<float>({}, {1.f});
  const OrtStatefulNamedValue initial{"past_value", seed}, input{"input_value", one};
  const OrtStatefulInputBatch batches[2]{{&input, 1}, {&input, 1}};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &initial;
  inputs.num_inputs = 1;
  inputs.batches = batches;
  inputs.num_batches = 2;
  auto execution = loop.CreateExecution(inputs);
  auto result = execution.Run();
  EXPECT_EQ(Output(result, "present_value").GetTensorData<float>()[0], 2.f);
  const auto staging = execution.GetStagingInfo();
  EXPECT_EQ(staging.cpu_input_request_bytes, 20u);
  EXPECT_EQ(staging.cpu_output_bytes, 8u);
  EXPECT_EQ(staging.cpu_snapshot_copy_bytes, 8u);
  execution.Reset();
  EXPECT_EQ(execution.GetStagingInfo().cpu_input_request_bytes, 20u);
  EXPECT_THROW(execution.GetStagingInfo(1), Ort::Exception);
  EXPECT_THROW(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_CACHE, ORT_STATEFUL_CACHE_COPY), Ort::Exception);
  const auto& api = Ort::GetStatefulApi();
  OrtStatefulProviderCapabilities untouched{};
  untouched.operations = 91;
  auto* status = api.QueryStatefulProviderCapabilities(session, kTestProvider, 0, &untouched);
  ASSERT_NE(status, nullptr);
  Ort::GetApi().ReleaseStatus(status);
  EXPECT_EQ(untouched.operations, 91u);
  OrtStatefulStagingInfo short_info{};
  short_info.cpu_output_bytes = 99;
  status = api.StatefulExecutionGetStagingInfo(execution, 0, 0, &short_info);
  ASSERT_NE(status, nullptr);
  Ort::GetApi().ReleaseStatus(status);
  EXPECT_EQ(short_info.cpu_output_bytes, 99u);
  auto bad = Selection();
  bad.struct_size = 0;
  EXPECT_THROW(options.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, bad), Ort::Exception);
  bad = Selection();
  bad.provider_name = "missing";
  OrtStatefulSamplerConfig sampling{sizeof(sampling), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions sampler(sampling);
  EXPECT_THROW(sampler.SetProvider(bad, session), Ort::Exception);
  EXPECT_THROW(sampler.SetProvider(Selection()), Ort::Exception);
  bad = Selection(ORT_STATEFUL_PROVIDER_CPU);
  bad.provider_name = kTestProvider;
  EXPECT_THROW(sampler.SetProvider(bad), Ort::Exception);
  if (GetParam()) {
    EXPECT_EQ(Ort::QueryStatefulProviderCapabilities(session, kCudaExecutionProvider).operations, 0u);
    EXPECT_THROW(sampler.SetProvider(Selection(ORT_STATEFUL_PROVIDER_REQUIRE, kCudaExecutionProvider), session),
                 Ort::Exception);
  }
}

TEST(StatefulProviderDescriptorTest, RejectsInconsistentVersionedDescriptors) {
  const auto original = *GetCpuStatefulProviderDescriptor();
  EXPECT_STATUS_OK(StatefulProviderBinding::ValidateDescriptor(original));
  for (int kind = 0; kind < 10; ++kind) {
    auto value = original;
    switch (kind) {
      case 0:
        value.struct_size = 0;
        break;
      case 1:
        value.capabilities.struct_size = 0;
        break;
      case 2:
        value.capabilities.version = 2;
        break;
      case 3:
        value.capabilities.operations |= uint64_t{1} << 50;
        break;
      case 4:
        value.CacheOperation = nullptr;
        break;
      case 5:
        value.Greedy = nullptr;
        break;
      case 6:
        value.capabilities.cache_element_types = 0;
        break;
      case 7:
        value.capabilities.sampler_element_types = 0;
        break;
      case 8:
        value.capabilities.memory_types = 0;
        break;
      case 9:
        value.context = &value;
        break;
    }
    EXPECT_FALSE(StatefulProviderBinding::ValidateDescriptor(value).IsOK()) << kind;
  }
}

TEST(StatefulProviderApiTest, ValidatesNewRequiredPointersAndSelectionFields) {
  const auto& api = Ort::GetStatefulApi();
  auto error = [](OrtStatus* status) {
    EXPECT_NE(status, nullptr);
    if (status) {
      EXPECT_NE(std::string_view(Ort::GetApi().GetErrorMessage(status)).size(), 0u);
      Ort::GetApi().ReleaseStatus(status);
    }
  };
  error(api.SessionLoopOptionsSetProvider(nullptr, ORT_STATEFUL_FEATURE_CACHE, nullptr));
  error(api.SessionChainOptionsSetComponentProvider(nullptr, 0, ORT_STATEFUL_FEATURE_CACHE, nullptr));
  error(api.KVCacheOptionsSetProvider(nullptr, nullptr, nullptr));
  error(api.SamplerOptionsSetProvider(nullptr, nullptr, nullptr));
  error(api.QueryStatefulProviderCapabilities(nullptr, nullptr, 0, nullptr));
  error(api.StatefulExecutionGetDispatchInfo(nullptr, ORT_STATEFUL_FEATURE_CACHE, 0,
                                             ORT_STATEFUL_CACHE_COPY, 0, nullptr));
  error(api.KVCacheManagerGetDispatchInfo(nullptr, ORT_STATEFUL_CACHE_COPY, 0, nullptr));
  error(api.SamplerOptionsGetDispatchInfo(nullptr, 0, nullptr));
  error(api.KVCacheManagerRequiresReset(nullptr, nullptr));
  error(api.StatefulExecutionGetStagingInfo(nullptr, 0, 0, nullptr));
  OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  Ort::SamplerOptions options(config);
  options.SetProvider(Selection(ORT_STATEFUL_PROVIDER_CPU));
  auto owned = Tensor<float>({1}, {7.f});
  auto retained = StatefulProviderBinding{}.RetainValue(*static_cast<OrtValue*>(owned));
  owned = Ort::Value(nullptr);
  EXPECT_EQ(retained.Get<onnxruntime::Tensor>().Data<float>()[0], 7.f);
  for (int kind = 0; kind < 7; ++kind) {
    auto value = Selection(ORT_STATEFUL_PROVIDER_CPU);
    switch (kind) {
      case 0:
        value.mode = static_cast<OrtStatefulProviderMode>(99);
        break;
      case 1:
        value.device_type = static_cast<OrtMemoryInfoDeviceType>(99);
        break;
      case 2:
        value.device_id = -2;
        break;
      case 3:
        value.device_id = 1;
        break;
      case 4:
        value.device_type = OrtMemoryInfoDeviceType_GPU;
        break;
      case 5:
        value.provider_name = "unused";
        break;
      case 6:
        value.mode = ORT_STATEFUL_PROVIDER_REQUIRE;
        break;
    }
    EXPECT_THROW(options.SetProvider(value), Ort::Exception) << kind;
  }
  error(api.SamplerOptionsSetProvider(options, nullptr, nullptr));
  auto logits = Tensor<float>({1, 2}, {1.f, 2.f});
  EXPECT_EQ(options.Sample(logits), 1);
  OrtStatefulDispatchInfo info{};
  info.cpu_calls = 99;
  error(api.SamplerOptionsGetDispatchInfo(options, 0, &info));
  EXPECT_EQ(info.cpu_calls, 99u);
  EXPECT_THROW(options.Sample(Tensor<int64_t>({1, 2}, {1, 2})), Ort::Exception);
  EXPECT_THROW(options.Sample(Tensor<float>({2}, {1.f, 2.f})), Ort::Exception);
  EXPECT_THROW(options.Sample(Tensor<float>({1, 0}, {})), Ort::Exception);
  config.has_logits_position = 1;
  config.logits_position = 3;
  options.SetConfig(config);
  EXPECT_THROW(options.Sample(logits), Ort::Exception);
  options.SetConfig({sizeof(config), ORT_STATEFUL_GREEDY, 0, 0});
  Ort::MemoryInfo gpu("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  float data[2]{1.f, 2.f};
  const int64_t shape[2]{1, 2};
  auto device = Ort::Value::CreateTensor<float>(gpu, data, 2, shape, 2);
  try {
    options.Sample(device);
    FAIL() << "CPU dispatch must not copy device logits";
  } catch (const Ort::Exception& ex) {
    EXPECT_EQ(ex.GetOrtErrorCode(), ORT_NOT_IMPLEMENTED);
  }
}

INSTANTIATE_TEST_SUITE_P(Cpu, StatefulProviderTest, testing::Values(false));
#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(Cuda, StatefulProviderTest, testing::Values(true));
#endif
}  // namespace
}  // namespace onnxruntime::test
