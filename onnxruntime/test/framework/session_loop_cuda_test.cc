// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <type_traits>

#include <cuda_runtime_api.h>
#include "gtest/gtest.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "core/session/session_chain.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/stateful_execution_test_utils.h"

namespace onnxruntime::test {
namespace {

class ObservedCacheSession : public InferenceSession {
 public:
  using InferenceSession::InferenceSession;
  std::function<Status(IOBinding&)> before, after;
  size_t calls = 0;
  Status Run(const RunOptions& options, IOBinding& io) override {
    ++calls;
    if (before) ORT_RETURN_IF_ERROR(before(io));
    ORT_RETURN_IF_ERROR(InferenceSession::Run(options, io));
    return after ? after(io) : Status::OK();
  }
};

class CudaCacheIntegrationTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaGetDevice(&previous_device_), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
  void TearDown() override { EXPECT_EQ(cudaSetDevice(previous_device_), cudaSuccess); }

  std::unique_ptr<ObservedCacheSession> Session(bool gpu = true, cudaStream_t stream = nullptr,
                                                ONNX_NAMESPACE::ModelProto model = stateful::MakeDynamicDecoderModel(),
                                                int device_id = 0) {
    SessionOptions options;
    options.intra_op_param.thread_pool_size = 1;
    options.inter_op_param.thread_pool_size = 1;
    options.graph_optimization_level = TransformerLevel::Default;
    auto session = std::make_unique<ObservedCacheSession>(options, GetEnvironment());
    if (gpu) {
      OrtCUDAProviderOptionsV2 cuda{};
      cuda.device_id = device_id;
      cuda.user_compute_stream = stream;
      cuda.has_user_compute_stream = stream != nullptr;
      ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(CudaExecutionProviderWithOptions(&cuda)));
    }
    const auto data = model.SerializeAsString();
    ORT_THROW_IF_ERROR(session->Load(data.data(), static_cast<int>(data.size())));
    ORT_THROW_IF_ERROR(session->Initialize());
    for (const auto& node : session->GetSessionState().GetGraphViewer().Nodes()) {
      EXPECT_EQ(node.GetExecutionProviderType(), gpu ? kCudaExecutionProvider : kCpuExecutionProvider);
    }
    return session;
  }

  OrtValue Tokens(std::initializer_list<int64_t> tokens = {0, 1}) {
    OrtValue value;
    CreateMLValue<int64_t>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                           {1, static_cast<int64_t>(tokens.size())}, tokens, &value);
    return value;
  }

  SessionLoopOptions Options(bool gpu = true, KVCacheUpdateKind kind = KVCacheUpdateKind::Share) {
    SessionLoopOptions options;
    options.finite_input_names = {"token"};
    options.output_names = {"logits", "present_key"};
    options.exit_conditions = {{LoopExitKind::EndOfInputs}};
    options.kv_caches = {{{"key", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, 0, 1}, 2, 0},
                          "past_key",
                          kind == KVCacheUpdateKind::Append ? "new_key" : "present_key",
                          kind}};
    if (gpu) {
      options.cache_provider = {ORT_STATEFUL_PROVIDER_REQUIRE, kCudaExecutionProvider, OrtMemoryInfoDeviceType_GPU, 0};
    }
    return options;
  }

  SessionLoopOptions GenerationOptions(bool cache_gpu, bool sampler_gpu, bool decode = false,
                                       int64_t budget = 3) {
    auto options = Options(cache_gpu);
    options.generation = TokenGenerationOptions{
        decode ? TokenGenerationPhase::Decode : TokenGenerationPhase::Prefill,
        "token", "logits", "", "", std::nullopt};
    options.exit_conditions = {{LoopExitKind::MaxGeneratedTokens, budget}};
    if (decode) {
      options.finite_input_names.clear();
    } else {
      options.exit_conditions.push_back({LoopExitKind::EndOfInputs});
    }
    if (sampler_gpu) {
      options.sampler_provider = {
          ORT_STATEFUL_PROVIDER_PREFER, kCudaExecutionProvider, OrtMemoryInfoDeviceType_GPU, 0};
    }
    return options;
  }

  std::unique_ptr<SessionChainExecution> Generation(InferenceSession& prefill, InferenceSession& decode,
                                                    bool cache_gpu = true) {
    auto first = Loop(prefill, GenerationOptions(cache_gpu, true));
    auto second = Loop(decode, GenerationOptions(cache_gpu, true, true));
    SessionChainOptions options{
        {{"prefill", first, {{"token", {ChainSourceKind::ExternalInput, "prompt"}}}, {}, {}},
         {"decode", second, {}, {}, {}}},
        {{"tokens", 0, "", ChainOutputKind::GeneratedTokenIds}}};
    std::shared_ptr<const SessionChain> chain;
    ORT_THROW_IF_ERROR(SessionChain::Create(std::move(options), chain));
    std::unique_ptr<SessionChainExecution> execution;
    ORT_THROW_IF_ERROR(chain->CreateExecution({{{"prompt", Tokens()}}, {}, {}}, execution));
    return execution;
  }

  std::shared_ptr<const SessionLoop> Loop(InferenceSession& session, SessionLoopOptions options) {
    std::shared_ptr<const SessionLoop> loop;
    ORT_THROW_IF_ERROR(SessionLoop::Create(session, std::move(options), loop));
    return loop;
  }

  std::unique_ptr<SessionLoopExecution> Execution(InferenceSession& session, SessionLoopOptions options) {
    auto loop = Loop(session, std::move(options));
    std::unique_ptr<SessionLoopExecution> execution;
    ORT_THROW_IF_ERROR(loop->CreateExecution({{}, {{{"token", Tokens()}}}}, execution));
    return execution;
  }

  static InlinedVector<float> Read(const OrtValue& value) {
    const auto& tensor = value.Get<Tensor>();
    InlinedVector<float> data(static_cast<size_t>(tensor.Shape().Size()));
    if (!data.empty()) {
      if (tensor.Location().device.Type() == OrtDevice::GPU) {
        EXPECT_EQ(cudaMemcpy(data.data(), tensor.DataRaw(), tensor.SizeInBytes(), cudaMemcpyDeviceToHost), cudaSuccess);
      } else {
        std::copy_n(tensor.Data<float>(), data.size(), data.data());
      }
    }
    return data;
  }
  int previous_device_ = 0;
};

TEST_F(CudaCacheIntegrationTest, NativeFactoryHonorsExplicitOrdinalAndStream) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  const int device = count > 1 ? 1 : 0;
  ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
  cudaStream_t raw = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking), cudaSuccess);
  auto release = [](cudaStream_t stream) { EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess); };
  std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(release)> stream(raw, release);
  auto session = Session(true, raw, stateful::MakeDynamicDecoderModel(), device);
  auto options = Options();
  options.cache_provider.device_id = device;
  auto execution = Execution(*session, options);
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.outputs[1].Get<Tensor>().Location().device.Id(), device);
  EXPECT_EQ(Read(result.outputs[1]), (InlinedVector<float>{0.f, 1.f}));
  ASSERT_EQ(cudaStreamQuery(raw), cudaSuccess);
  OrtCUDAProviderOptionsV2 invalid{};
  invalid.device_id = count;
  EXPECT_ANY_THROW(CudaExecutionProviderWithOptions(&invalid));
  RecordProperty("native_explicit_second_device_tested", count > 1);
}

TEST_F(CudaCacheIntegrationTest, NativeCachePortsResultsStateAndRetainedLifetimes) {
  for (const auto kind : {KVCacheUpdateKind::Share, KVCacheUpdateKind::Copy, KVCacheUpdateKind::Append}) {
    auto model = stateful::MakeDynamicDecoderModel();
    if (kind == KVCacheUpdateKind::Append) {
      stateful::SetTensorType(*model.mutable_graph()->add_output(), "new_key",
                              ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});
    }
    auto session = Session(true, nullptr, model);
    session->before = [&](IOBinding& io) {
      const auto index = std::find(io.GetInputNames().begin(), io.GetInputNames().end(), "past_key") -
                         io.GetInputNames().begin();
      EXPECT_EQ(io.GetInputs()[index].Get<Tensor>().Location().device.Type(), OrtDevice::GPU);
      return Status::OK();
    };
    session->after = [&](IOBinding& io) {
      for (size_t i = 0; i < io.GetOutputNames().size(); ++i) {
        const auto& name = io.GetOutputNames()[i];
        if (name == (kind == KVCacheUpdateKind::Append ? "new_key" : "present_key")) {
          EXPECT_EQ(io.GetOutputs()[i].Get<Tensor>().Location().device.Type(), OrtDevice::GPU);
        }
      }
      return Status::OK();
    };
    auto options = Options(true, kind);
    if (kind == KVCacheUpdateKind::Append) options.output_names = {"logits", "new_key"};
    auto execution = Execution(*session, options);
    auto independent = Execution(*session, options);
    SessionLoopStepResult result;
    ASSERT_STATUS_OK(execution->Run(result));
    ASSERT_EQ(result.outputs.size(), 2u);
    EXPECT_EQ(result.outputs[0].Get<Tensor>().Location().device.Type(), OrtDevice::CPU);
    EXPECT_EQ(result.outputs[1].Get<Tensor>().Location().device.Type(), OrtDevice::GPU);
    EXPECT_EQ(Read(result.outputs[1]), (InlinedVector<float>{0.f, 1.f}));
    LoopFeeds state;
    ASSERT_STATUS_OK(execution->GetState(state));
    EXPECT_EQ(state.at("key").Get<Tensor>().Location().device.Type(), OrtDevice::GPU);
    EXPECT_EQ(Read(state.at("key")), (InlinedVector<float>{0.f, 1.f}));
    if (kind == KVCacheUpdateKind::Share) {
      EXPECT_EQ(state.at("key").Get<Tensor>().DataRaw(), result.outputs[1].Get<Tensor>().DataRaw());
    } else {
      EXPECT_NE(state.at("key").Get<Tensor>().DataRaw(), result.outputs[1].Get<Tensor>().DataRaw());
    }
    KVCacheView empty;
    ASSERT_STATUS_OK(independent->Cache()->Get("key", empty));
    EXPECT_EQ(empty.length, 0);
    const auto staging = execution->StagingInfo();
    EXPECT_EQ(staging.cpu_output_bytes, 32u);
    EXPECT_EQ(staging.cpu_snapshot_copy_bytes, 32u);
    ASSERT_STATUS_OK(execution->Reset());
    ASSERT_STATUS_OK(execution->Run(result));
    execution.reset();
    independent.reset();
    session.reset();
    EXPECT_EQ(Read(state.at("key")), (InlinedVector<float>{0.f, 1.f}));
    EXPECT_EQ(Read(result.outputs[1]), (InlinedVector<float>{0.f, 1.f}));
  }
}

TEST_F(CudaCacheIntegrationTest, ModelErrorCancellationAndResetPreservePriorResults) {
  auto session = Session();
  auto execution = Execution(*session, Options());
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  const auto retained = result.outputs[1];
  ASSERT_STATUS_OK(execution->Reset());
  session->after = [&](IOBinding&) {
    execution->Cancel();
    return Status::OK();
  };
  EXPECT_FALSE(execution->Run(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  ASSERT_STATUS_OK(execution->Reset());
  session->after = [](IOBinding&) { return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "controlled model failure"); };
  EXPECT_FALSE(execution->Run(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  session->after = {};
  ASSERT_STATUS_OK(execution->Reset());
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(Read(retained), (InlinedVector<float>{0.f, 1.f}));
}

TEST_F(CudaCacheIntegrationTest, DeviceFeedbackValidatesInitialStorageAndPreservesSnapshots) {
  auto model = stateful::MakeDynamicDecoderModel();
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(
      *graph.add_input(), "mirror", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});
  auto& concat = *graph.mutable_node(graph.node_size() - 1);
  concat.add_input("mirror");
  auto session = Session(true, nullptr, model);
  auto options = Options();
  options.feedback = {{"mirror", "present_key"}};
  auto loop = Loop(*session, options);
  OrtValue empty;
  const auto& allocators = session->GetSessionState().GetAllocators();
  const auto allocator = std::find_if(allocators.begin(), allocators.end(), [](const auto& item) {
                           return item.first.Type() == OrtDevice::GPU &&
                                  item.first.MemType() == OrtDevice::MemType::DEFAULT;
                         })->second;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1, 1, 0, 1}, allocator, empty);
  SessionLoopInputs inputs{{{"mirror", empty}}, {{{"token", Tokens()}}, {{"token", Tokens()}}}};
  std::unique_ptr<SessionLoopExecution> execution;
  ASSERT_STATUS_OK(loop->CreateExecution(inputs, execution));
  SessionLoopStepResult first, second;
  ASSERT_STATUS_OK(execution->Step(first));
  ASSERT_STATUS_OK(execution->Step(second));
  EXPECT_EQ(Read(first.outputs[1]), (InlinedVector<float>{0, 1}));
  EXPECT_EQ(Read(second.outputs[1]), (InlinedVector<float>{0, 1, 0, 1, 0, 1}));
  OrtValue host;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1, 1, 0, 1},
                       TestCPUExecutionProvider()->CreatePreferredAllocators()[0], host);
  inputs.inputs["mirror"] = host;
  EXPECT_FALSE(loop->CreateExecution(std::move(inputs), execution).IsOK());
}

TEST_F(CudaCacheIntegrationTest, ExplicitGpuPortsRejectHostConsumersAndUnavailableProviders) {
  auto cpu = Session(false);
  std::shared_ptr<const SessionLoop> loop;
  EXPECT_FALSE(SessionLoop::Create(*cpu, Options(), loop).IsOK());
  auto gpu = Session();
  auto options = Options();
  options.cache_provider.device_id = 1;
  EXPECT_FALSE(SessionLoop::Create(*gpu, options, loop).IsOK());
  options = Options();
  // Inject a declared CPU consumer; no GPU cache may be silently staged to it.
  const OrtDevice cpu_device;
  const auto& state = gpu->GetSessionState();
  auto& ports = const_cast<SessionState::NameNodeInfoMapType&>(state.GetInputNodeInfoMap());
  const auto* original = ports.at("past_key").front().device;
  ports.at("past_key").front().device = &cpu_device;
  EXPECT_FALSE(SessionLoop::Create(*gpu, options, loop).IsOK());
  ports.at("past_key").front().device = original;
}

TEST_F(CudaCacheIntegrationTest, SynchronousTaskStreamWithoutGlobalDeviceWait) {
  cudaStream_t compute = nullptr, unrelated = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&unrelated, cudaStreamNonBlocking), cudaSuccess);
  {
    auto session = Session(true, compute);
    auto options = Options();
    options.output_names = {"present_key"};
    auto loop = Loop(*session, options);
    const auto& allocators = session->GetSessionState().GetAllocators();
    const auto allocator = std::find_if(allocators.begin(), allocators.end(), [](const auto& item) {
                             const auto& device = item.first;
                             return device.Type() == OrtDevice::GPU && device.MemType() == OrtDevice::MemType::DEFAULT;
                           })->second;
    OrtValue tokens;
    Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape{1, 2}, allocator, tokens);
    OrtValue prepared[2];
    for (auto& output : prepared) {
      Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1, 1, 2, 1}, allocator, output);
    }
    session->before = [&](IOBinding& io) {
      return io.BindOutput("present_key", prepared[session->calls - 1]);
    };
    const int64_t ids[]{0, 1};
    ASSERT_EQ(cudaMemcpy(tokens.GetMutable<Tensor>()->MutableDataRaw(), ids, sizeof(ids),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    std::unique_ptr<SessionLoopExecution> execution;
    ASSERT_STATUS_OK(loop->CreateExecution({{}, {{{"token", tokens}}}}, execution));
    // Isolate device-cache work from ordinary synchronous host input/output transfers.
    std::promise<void> warmed, start;
    auto begin = start.get_future();
    auto finished = std::async(std::launch::async, [&] {
      SessionLoopStepResult step;
      const auto status = execution->Run(step);
      warmed.set_value();
      if (!status.IsOK()) return status;
      ORT_RETURN_IF_ERROR(execution->Reset());
      step = {};
      begin.wait();
      return execution->Run(step);
    });
    warmed.get_future().wait();
    std::promise<void> entered, release;
    auto ready = release.get_future();
    std::pair<std::promise<void>*, std::future<void>*> gate{&entered, &ready};
    ASSERT_EQ(cudaLaunchHostFunc(unrelated, [](void* pointer) {
      auto& gate = *static_cast<std::pair<std::promise<void>*, std::future<void>*>*>(pointer);
      gate.first->set_value();
      gate.second->wait(); }, &gate), cudaSuccess);
    entered.get_future().wait();
    start.set_value();
    const auto completion = finished.wait_for(std::chrono::seconds(5));
    release.set_value();
    EXPECT_EQ(completion, std::future_status::ready);
    ASSERT_STATUS_OK(finished.get());
    EXPECT_EQ(cudaStreamQuery(compute), cudaSuccess);
  }
  EXPECT_EQ(cudaStreamDestroy(unrelated), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(compute), cudaSuccess);
}

class CudaSamplingIntegrationTest : public CudaCacheIntegrationTest {};

TEST_F(CudaSamplingIntegrationTest, IndependentCacheAllocationAndGpuLogitsAreBoundAndRetained) {
  for (const bool cache_gpu : {false, true}) {
    auto prefill = Session(), decode = Session();
    auto check_outputs = [&](IOBinding& io) {
      for (size_t i = 0; i < io.GetOutputNames().size(); ++i) {
        const auto& name = io.GetOutputNames()[i];
        EXPECT_EQ(io.GetOutputs()[i].Get<Tensor>().Location().device.Type(),
                  name == "logits" || cache_gpu ? OrtDevice::GPU : OrtDevice::CPU);
      }
      return Status::OK();
    };
    prefill->after = check_outputs;
    decode->after = check_outputs;
    auto execution = Generation(*prefill, *decode, cache_gpu);
    SessionChainStepResult result;
    ASSERT_STATUS_OK(execution->Run(result));
    EXPECT_EQ(execution->Generation()->token_ids, (InlinedVector<int64_t>{2, 3, 0}));
    EXPECT_EQ(execution->Generation()->processed_tokens, 4);
    const auto logits = result.component_outputs.at("logits");
    const auto saved = Read(logits);
    KVCacheView view;
    ASSERT_STATUS_OK(execution->Cache()->Get("key", view));
    EXPECT_EQ(view.value.Get<Tensor>().Location().device.Type(), cache_gpu ? OrtDevice::GPU : OrtDevice::CPU);
    EXPECT_EQ(Read(view.value), (InlinedVector<float>{0, 1, 2, 3}));
    OrtStatefulStagingInfo staging{};
    ASSERT_STATUS_OK(execution->GetStagingInfo(0, staging));
    EXPECT_EQ(staging.cpu_output_bytes, cache_gpu ? 0u : 8u);
    ASSERT_STATUS_OK(execution->GetStagingInfo(1, staging));
    EXPECT_EQ(staging.cpu_output_bytes, cache_gpu ? 0u : 28u);
    for (size_t stage = 0; stage < 2; ++stage) {
      OrtStatefulDispatchInfo info{};
      ASSERT_STATUS_OK(execution->GetDispatchInfo(
          ORT_STATEFUL_FEATURE_SAMPLER, stage, ORT_STATEFUL_SAMPLE_GREEDY, info));
      EXPECT_EQ(info.policy, ORT_STATEFUL_PROVIDER_PREFER);
      EXPECT_EQ(info.provider_calls, stage == 0 ? 1u : 2u);
      EXPECT_EQ(info.cpu_calls, 0u);
      EXPECT_EQ(info.cpu_delegations, 0u);
    }
    ASSERT_STATUS_OK(execution->Reset());
    execution.reset();
    prefill.reset();
    decode.reset();
    EXPECT_EQ(Read(logits), saved);
  }
}

TEST_F(CudaSamplingIntegrationTest, GpuSamplerWithoutCacheAndUnselectedLogitsDoesNotReadBackLogits) {
  auto model = stateful::MakeDynamicDecoderModel();
  auto& graph = *model.mutable_graph();
  graph.mutable_input()->DeleteSubrange(1, 1);
  graph.mutable_output()->DeleteSubrange(1, 1);
  graph.mutable_node()->DeleteSubrange(1, graph.node_size() - 1);
  graph.mutable_initializer()->DeleteSubrange(1, graph.initializer_size() - 1);
  auto session = Session(true, nullptr, model);
  auto options = GenerationOptions(false, true, false, 1);
  options.kv_caches.clear();
  options.output_names = {"logits"};
  auto execution = Execution(*session, options);
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(execution->Cache(), nullptr);
  EXPECT_EQ(result.generated_token, 2);
  EXPECT_EQ(result.outputs[0].Get<Tensor>().Location().device.Type(), OrtDevice::GPU);
  EXPECT_EQ(execution->StagingInfo().cpu_output_bytes, 0u);
  auto cached = Session();
  auto hidden = GenerationOptions(true, true, false, 1);
  hidden.output_names = {"present_key"};
  cached->after = [](IOBinding& io) {
    EXPECT_NE(std::find(io.GetOutputNames().begin(), io.GetOutputNames().end(), "logits"), io.GetOutputNames().end());
    return Status::OK();
  };
  auto unselected = Execution(*cached, hidden);
  ASSERT_STATUS_OK(unselected->Run(result));
  ASSERT_EQ(result.outputs.size(), 1u);
  EXPECT_EQ(result.generated_token, 2);
  EXPECT_EQ(unselected->StagingInfo().cpu_output_bytes, 0u);
}

TEST_F(CudaSamplingIntegrationTest, GpuSamplingFailureCancellationAndResetSuppressDownstream) {
  auto prefill = Session(), decode = Session();
  auto execution = Generation(*prefill, *decode);
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  const auto retained = result.component_outputs.at("logits");
  const auto bytes = Read(retained);
  ASSERT_STATUS_OK(execution->Reset());
  prefill->after = [](IOBinding& io) {
    const auto found = std::find(io.GetOutputNames().begin(), io.GetOutputNames().end(), "logits");
    ORT_RETURN_IF(found == io.GetOutputNames().end(), "Missing controlled model logits output");
    const size_t index = static_cast<size_t>(found - io.GetOutputNames().begin());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ORT_RETURN_IF(cudaMemcpy(io.GetOutputs()[index].GetMutable<Tensor>()->MutableDataRaw(),
                             &nan, sizeof(nan), cudaMemcpyHostToDevice) != cudaSuccess,
                  "Failed to inject controlled model NaN");
    return Status::OK();
  };
  const auto downstream = decode->calls;
  EXPECT_FALSE(execution->Run(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(decode->calls, downstream);
  EXPECT_FALSE(execution->Step(result).IsOK());
  ASSERT_STATUS_OK(execution->Reset());
  prefill->after = [](IOBinding&) { return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "controlled model error"); };
  EXPECT_FALSE(execution->Run(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(decode->calls, downstream);
  ASSERT_STATUS_OK(execution->Reset());
  prefill->after = [&](IOBinding&) {
    execution->Cancel();
    return Status::OK();
  };
  EXPECT_FALSE(execution->Run(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(decode->calls, downstream);
  ASSERT_STATUS_OK(execution->Reset());
  prefill->after = {};
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(Read(retained), bytes);
  EXPECT_EQ(execution->Generation()->token_ids, (InlinedVector<int64_t>{2, 3, 0}));
}

TEST_F(CudaSamplingIntegrationTest, RolesPlacementAndUnexpectedOutputStorageRejectBeforeSampling) {
  EXPECT_EQ(StatefulProviderBinding{}.GetAllocator(), nullptr);
  auto session = Session();
  auto options = GenerationOptions(false, true, false, 1);
  const OrtDevice cpu;
  auto& outputs = const_cast<SessionState::NameNodeInfoMapType&>(session->GetSessionState().GetOutputNodeInfoMap());
  const auto* saved = outputs.at("logits").front().device;
  outputs.at("logits").front().device = &cpu;
  std::shared_ptr<const SessionLoop> loop;
  EXPECT_FALSE(SessionLoop::Create(*session, options, loop).IsOK());
  outputs.at("logits").front().device = saved;
  auto execution = Execution(*session, options);
  session->after = [](IOBinding& io) {
    const auto allocator = io.GetCPUAllocator(kCpuExecutionProvider);
    const auto shape = io.GetOutputs()[0].Get<Tensor>().Shape();
    OrtValue host;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), shape, allocator, host);
    io.GetOutputs()[0] = std::move(host);
    return Status::OK();
  };
  SessionLoopStepResult result;
  EXPECT_FALSE(execution->Run(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  OrtStatefulDispatchInfo info{};
  ASSERT_STATUS_OK(execution->GetDispatchInfo(
      ORT_STATEFUL_FEATURE_SAMPLER, 0, ORT_STATEFUL_SAMPLE_GREEDY, info));
  EXPECT_EQ(info.provider_calls, 0u);
  session->after = {};
  ASSERT_STATUS_OK(execution->Reset());
  ASSERT_STATUS_OK(execution->Run(result));
  auto model = stateful::MakeDynamicDecoderModel();
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "past_logits", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  stateful::SetTensorType(*graph.add_output(), "seen", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  stateful::AddNode(graph, "Identity", {"past_logits"}, "seen");
  auto aliased = Session(true, nullptr, model);
  options = GenerationOptions(false, true);
  options.kv_caches.push_back({{"scores", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 0, 4}, 1, 0},
                               "past_logits",
                               "logits",
                               KVCacheUpdateKind::Share});
  EXPECT_FALSE(SessionLoop::Create(*aliased, options, loop).IsOK());
  options.cache_provider = {
      ORT_STATEFUL_PROVIDER_REQUIRE, kCudaExecutionProvider, OrtMemoryInfoDeviceType_GPU, 0};
  ASSERT_STATUS_OK(SessionLoop::Create(*aliased, options, loop));
}

}  // namespace
}  // namespace onnxruntime::test

#endif
