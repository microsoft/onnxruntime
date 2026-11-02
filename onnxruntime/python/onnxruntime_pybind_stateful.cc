// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_state_common.h"

#include <pybind11/stl.h>

#if !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "core/common/path_string.h"
#include "core/framework/data_transfer_manager.h"
#include "core/session/abi_stateful_impl.h"
#include "core/session/session_chain.h"
#endif

namespace onnxruntime::python {
namespace py = pybind11;

#if !defined(ORT_MINIMAL_BUILD)
namespace {

template <typename T>
T Field(py::handle value, const char* name) {
  return py::cast<T>(value.attr(name));
}

std::string OptionalName(py::handle value, const char* name) {
  const auto item = value.attr(name);
  return item.is_none() ? std::string{} : item.cast<std::string>();
}

std::optional<int64_t> Position(py::handle value) {
  const auto position = value.attr("logits_position");
  return position.is_none() ? std::nullopt : std::make_optional(position.cast<int64_t>());
}

StatefulProviderSelection Provider(py::handle value) {
  const auto name = OptionalName(value, "provider_name");
  OrtStatefulProviderSelection config{};
  config.struct_size = sizeof(config);
  config.mode = static_cast<OrtStatefulProviderMode>(Field<int>(value, "mode"));
  config.provider_name = name.c_str();
  config.device_type = static_cast<OrtMemoryInfoDeviceType>(Field<int>(value, "device_type"));
  config.device_id = Field<int>(value, "device_id");
  StatefulProviderSelection result;
  OrtPybindThrowIfError(StatefulProviderSelection::Parse(&config, result));
  return result;
}

KVCacheSpec Cache(py::handle value) {
  KVCacheSpec result;
  result.name = Field<std::string>(value, "name");
  result.element_type = Field<int32_t>(value, "element_type");
  for (const auto& dimension : value.attr("shape")) result.shape.push_back(py::cast<int64_t>(dimension));
  result.sequence_axis = Field<int64_t>(value, "sequence_axis");
  result.initial_capacity = Field<int64_t>(value, "initial_capacity");
  return result;
}

InlinedVector<LoopKVCacheBinding, 1> Caches(py::handle value) {
  InlinedVector<LoopKVCacheBinding, 1> result;
  for (const auto& item : value.attr("kv_caches")) {
    result.push_back({Cache(item.attr("cache")), Field<std::string>(item, "input_name"),
                      Field<std::string>(item, "output_name"),
                      static_cast<KVCacheUpdateKind>(Field<int>(item, "update"))});
  }
  return result;
}

ChainValueSource Source(py::handle value) {
  const int64_t index = Field<int64_t>(value, "component_index");
  if (index < 0) throw InvalidArgument("component_index must be nonnegative");
  return {static_cast<ChainSourceKind>(Field<int>(value, "kind")),
          Field<std::string>(value, "name"), static_cast<size_t>(index)};
}

SessionLoopOptions LoopOptions(py::handle value) {
  SessionLoopOptions options;
  for (const auto& name : value.attr("output_names")) options.output_names.push_back(py::cast<std::string>(name));
  for (const auto& name : value.attr("finite_input_names")) {
    options.finite_input_names.push_back(py::cast<std::string>(name));
  }
  for (const auto& item : value.attr("exit_conditions")) {
    LoopExitCondition condition{};
    condition.kind = static_cast<LoopExitKind>(Field<int>(item, "kind"));
    condition.limit = Field<int64_t>(item, "limit");
    condition.output_name = OptionalName(item, "output_name");
    for (const auto& token : item.attr("eos_token_ids")) condition.eos_token_ids.push_back(py::cast<int64_t>(token));
    options.exit_conditions.push_back(std::move(condition));
  }
  for (const auto& item : value.attr("feedback")) {
    options.feedback.push_back({Field<std::string>(item, "input_name"), Field<std::string>(item, "output_name")});
  }
  options.generated_token_output = OptionalName(value, "generated_token_output");
  options.nonempty_input = OptionalName(value, "nonempty_input");
  options.kv_caches = Caches(value);
  options.cache_provider = Provider(value.attr("cache_provider"));
  options.sampler_provider = Provider(value.attr("sampler_provider"));
  const auto generation = value.attr("generation");
  if (!generation.is_none()) {
    const int phase = Field<int>(generation, "phase");
    if (phase != 1 && phase != 2) throw InvalidArgument("Unknown generation phase");
    TokenGenerationOptions config;
    config.phase = phase == 1 ? TokenGenerationPhase::Prefill : TokenGenerationPhase::Decode;
    config.input_ids = Field<std::string>(generation, "input_ids");
    config.logits_output = Field<std::string>(generation, "logits_output");
    config.attention_mask = OptionalName(generation, "attention_mask");
    config.position_ids = OptionalName(generation, "position_ids");
    config.logits_position = Position(generation);
    options.generation = std::move(config);
  }
  const auto ddim = value.attr("ddim");
  if (!ddim.is_none()) {
    DdimOptions config;
    const auto shape = Field<std::vector<int64_t>>(ddim, "latent_shape");
    if (shape.size() != config.latent_shape.size()) throw InvalidArgument("DDIM requires four latent dimensions");
    std::copy(shape.begin(), shape.end(), config.latent_shape.begin());
    config.profile = Field<std::string>(ddim, "profile");
    config.num_inference_steps = Field<int64_t>(ddim, "num_inference_steps");
    config.guidance_scale = Field<float>(ddim, "guidance_scale");
    config.guidance_mode = static_cast<DdimGuidanceMode>(Field<int>(ddim, "guidance_mode"));
    config.timestep_format = static_cast<DdimTimestepFormat>(Field<int>(ddim, "timestep_format"));
    config.latent_input = Field<std::string>(ddim, "latent_input");
    config.timestep_input = Field<std::string>(ddim, "timestep_input");
    config.conditioning_input = Field<std::string>(ddim, "conditioning_input");
    config.prediction_output = Field<std::string>(ddim, "prediction_output");
    config.initialization_input = Field<std::string>(ddim, "initialization_input");
    config.positive_conditioning_input = Field<std::string>(ddim, "positive_conditioning_input");
    config.negative_conditioning_input = Field<std::string>(ddim, "negative_conditioning_input");
    config.latent_output = Field<std::string>(ddim, "latent_output");
    options.ddim = std::move(config);
  }
  return options;
}

py::dict DispatchMetadata(const OrtStatefulDispatchInfo& info) {
  py::dict result;
  result["provider_name"] = info.provider_name;
  result["requested_provider_name"] = info.requested_provider_name;
  result["policy"] = static_cast<int>(info.policy);
  result["origin"] = static_cast<int>(info.origin);
  result["operation"] = static_cast<int>(info.operation);
  result["device_type"] = static_cast<int>(info.device_type);
  result["device_id"] = info.device_id;
  result["cpu_calls"] = info.cpu_calls;
  result["provider_calls"] = info.provider_calls;
  result["cpu_delegations"] = info.cpu_delegations;
  result["failures"] = info.failures;
  return result;
}

struct NativeValue {
  OrtValue value;
  InlinedVector<InferenceSession*> sessions;

  OrtValue Copy(bool to_cpu) const {
    const auto& source = value.Get<Tensor>();
    AllocatorPtr allocator;
    if (to_cpu || source.Location().device.UsesCpuMemory()) {
      allocator = GetAllocator();
    } else {
      for (const auto* session : sessions) {
        allocator = session->GetAllocator(source.Location());
        if (allocator) break;
      }
    }
    if (!allocator) throw NotImplemented("No retained Session allocator supports this stateful value copy");
    OrtValue result;
    {
      py::gil_scoped_release release;
      Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, result);
      auto& target = *result.GetMutable<Tensor>();
      if (source.SizeInBytes() != 0) {
        if (source.Location().device.UsesCpuMemory() && target.Location().device.UsesCpuMemory()) {
          OrtPybindThrowIfError(CPUDataTransfer{}.CopyTensor(source, target));
        } else {
          const DataTransferManager* transfer = nullptr;
          for (const auto* session : sessions) {
            const auto& candidate = session->GetDataTransferManager();
            if (candidate.GetDataTransfer(source.Location().device, target.Location().device)) {
              transfer = &candidate;
              break;
            }
          }
          if (!transfer) throw NotImplemented("No retained Session transfer supports this stateful value copy");
          OrtPybindThrowIfError(transfer->CopyTensor(source, target));
        }
      }
    }
    return result;
  }
};

OrtValue InputValue(py::handle value) {
  if (py::isinstance<NativeValue>(value)) return py::cast<const NativeValue&>(value).value;
  return py::cast<OrtValue>(value);
}

LoopFeeds Feeds(const py::dict& values) {
  LoopFeeds result;
  for (const auto& item : values) result.emplace(py::cast<std::string>(item.first), InputValue(item.second));
  return result;
}

struct NativeResult {
  NativeResult() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativeResult);
  OrtStatefulResultKind kind = ORT_STATEFUL_LOOP_RESULT;
  SessionLoopStepResult progress;
  int64_t invocations = 0;
  size_t components_completed = 0;
  std::string component_name;
  std::array<InlinedVector<std::pair<std::string, OrtValue>>, 3> collections;
  InlinedVector<InferenceSession*> sessions;

  py::dict Info() const {
    py::dict info;
    info["kind"] = static_cast<int>(kind);
    info["executed"] = progress.executed;
    info["completed"] = progress.completed;
    info["invocations"] = invocations;
    info["iterations"] = progress.iterations;
    info["generated_tokens"] = progress.generated_tokens;
    info["components_completed"] = components_completed;
    info["component_name"] = component_name;
    std::vector<int> reasons;
    for (const auto reason : progress.exit_reasons) reasons.push_back(static_cast<int>(reason));
    info["exit_reasons"] = std::move(reasons);
    info["generated_token"] = py::cast(progress.generated_token);
    return info;
  }

  py::dict Outputs(int group) const {
    if (group < 0 || group > 2) throw InvalidArgument("Unknown stateful output group");
    py::dict result;
    for (const auto& [name, value] : collections[static_cast<size_t>(group)]) {
      result[py::str(name)] = py::cast(NativeValue{value, sessions});
    }
    return result;
  }
};

struct NativeExecution {
  NativeExecution() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativeExecution);
  std::unique_ptr<SessionLoopExecution> loop;
  std::unique_ptr<SessionChainExecution> chain;
  InlinedVector<std::string> output_names;
  InlinedVector<InferenceSession*> sessions;

  std::unique_ptr<NativeResult> Advance(bool run) {
    auto result = std::make_unique<NativeResult>();
    result->sessions = sessions;
    if (loop) {
      {
        py::gil_scoped_release release;
        OrtPybindThrowIfError(run ? loop->Run(result->progress) : loop->Step(result->progress));
      }
      result->invocations = result->progress.invocations;
      for (size_t i = 0; i < result->progress.outputs.size(); ++i) {
        result->collections[0].emplace_back(output_names[i], result->progress.outputs[i]);
      }
      result->collections[1] = result->collections[0];
    } else {
      SessionChainStepResult progress;
      {
        py::gil_scoped_release release;
        OrtPybindThrowIfError(run ? chain->Run(progress) : chain->Step(progress));
      }
      result->kind = ORT_STATEFUL_CHAIN_RESULT;
      result->progress = std::move(progress.component_result);
      result->progress.executed = progress.executed;
      result->progress.completed = progress.completed;
      result->progress.generated_tokens = progress.generated_tokens;
      result->invocations = progress.invocations;
      result->components_completed = progress.components_completed;
      result->component_name = std::move(progress.component_name);
      for (auto& item : progress.outputs) result->collections[0].emplace_back(item.first, std::move(item.second));
      for (auto& item : progress.component_outputs) {
        result->collections[1].emplace_back(item.first, std::move(item.second));
      }
    }
    return result;
  }

  void Cancel() {
    if (loop)
      loop->Cancel();
    else
      chain->Cancel();
  }

  void Reset() {
    py::gil_scoped_release release;
    OrtPybindThrowIfError(loop ? loop->Reset() : chain->Reset());
  }

  bool RequiresReset() const {
    return loop ? loop->RequiresReset() : chain->RequiresReset();
  }

  std::unique_ptr<NativeResult> State() const {
    if (RequiresReset()) throw Fail("Stateful execution requires Reset");
    auto result = std::make_unique<NativeResult>();
    result->kind = ORT_STATEFUL_STATE_SNAPSHOT;
    result->sessions = sessions;
    LoopFeeds state;
    if (loop)
      OrtPybindThrowIfError(loop->GetState(state));
    else
      state = chain->State();
    for (auto& item : state) result->collections[2].emplace_back(item.first, std::move(item.second));
    return result;
  }

  py::dict Staging(size_t component) const {
    OrtStatefulStagingInfo info{};
    OrtPybindThrowIfError(loop ? loop->GetStagingInfo(component, info) : chain->GetStagingInfo(component, info));
    py::dict result;
    result["cpu_input_request_bytes"] = info.cpu_input_request_bytes;
    result["cpu_output_bytes"] = info.cpu_output_bytes;
    result["cpu_snapshot_copy_bytes"] = info.cpu_snapshot_copy_bytes;
    return result;
  }

  py::dict Dispatch(int feature, int operation, size_t component) const {
    OrtStatefulDispatchInfo info{};
    const auto kind = static_cast<OrtStatefulProviderFeature>(feature);
    const auto op = static_cast<OrtStatefulProviderOperation>(operation);
    OrtPybindThrowIfError(loop ? loop->GetDispatchInfo(kind, component, op, info)
                               : chain->GetDispatchInfo(kind, component, op, info));
    return DispatchMetadata(info);
  }
};

struct NativeLoop {
  NativeLoop() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativeLoop);
  std::shared_ptr<const SessionLoop> value;
  InlinedVector<InferenceSession*> sessions;

  static std::unique_ptr<NativeLoop> Create(PyInferenceSession& session, py::handle config) {
    auto result = std::make_unique<NativeLoop>();
    auto options = LoopOptions(config);
    auto* native = session.GetSessionHandle();
    result->sessions.push_back(native);
    {
      py::gil_scoped_release release;
      OrtPybindThrowIfError(SessionLoop::Create(*native, std::move(options), result->value));
    }
    return result;
  }

  std::unique_ptr<NativeExecution> CreateExecution(const py::dict& values, const py::tuple& batches) const {
    SessionLoopInputs inputs;
    inputs.inputs = Feeds(values);
    for (const auto& batch : batches) inputs.batches.push_back(Feeds(py::cast<py::dict>(batch)));
    auto result = std::make_unique<NativeExecution>();
    result->sessions = sessions;
    result->output_names = value->Options().output_names;
    {
      py::gil_scoped_release release;
      OrtPybindThrowIfError(value->CreateExecution(std::move(inputs), result->loop));
    }
    return result;
  }
};

struct NativeChain {
  NativeChain() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativeChain);
  InlinedVector<PySessionOptionsSnapshot, 1> retained_options;
  std::shared_ptr<const SessionChain> value;
  InlinedVector<InferenceSession*> sessions;

  static std::unique_ptr<NativeChain> Create(py::handle config, const py::tuple& executions) {
    auto result = std::make_unique<NativeChain>();
    const auto components = config.attr("components").cast<py::tuple>();
    if (components.size() != executions.size()) throw InvalidArgument("Chain execution count mismatch");
    SessionChainOptions options;
    for (size_t i = 0; i < components.size(); ++i) {
      const auto item = components[i];
      SessionChainComponent component;
      component.name = Field<std::string>(item, "name");
      if (py::isinstance<NativeLoop>(executions[i])) {
        const auto& loop = executions[i].cast<const NativeLoop&>();
        component.execution = loop.value;
        result->sessions.insert(result->sessions.end(), loop.sessions.begin(), loop.sessions.end());
      } else {
        auto* session = executions[i].cast<PyInferenceSession*>()->GetSessionHandle();
        component.execution = session;
        result->sessions.push_back(session);
      }
      for (const auto& binding : item.attr("inputs")) {
        component.inputs.push_back({Field<std::string>(binding, "input_name"), Source(binding.attr("source"))});
      }
      for (const auto& binding : item.attr("state_outputs")) {
        component.state_outputs.push_back({Field<std::string>(binding, "state_name"),
                                           Field<std::string>(binding, "output_name")});
      }
      component.kv_caches = Caches(item);
      if (!item.attr("cache_provider").is_none()) component.cache_provider = Provider(item.attr("cache_provider"));
      if (!item.attr("sampler_provider").is_none()) component.sampler_provider = Provider(item.attr("sampler_provider"));
      const auto bootstrap = item.attr("generation_bootstrap");
      if (!bootstrap.is_none()) {
        component.generation_bootstrap = ChainGenerationBootstrap{
            Source(bootstrap.attr("logits")), Source(bootstrap.attr("prompt")), Position(bootstrap)};
      }
      options.components.push_back(std::move(component));
    }
    for (const auto& output : config.attr("outputs")) {
      const int64_t index = Field<int64_t>(output, "component_index");
      if (index < 0) throw InvalidArgument("component_index must be nonnegative");
      options.outputs.push_back({Field<std::string>(output, "name"), static_cast<size_t>(index),
                                 OptionalName(output, "output_name"),
                                 static_cast<ChainOutputKind>(Field<int>(output, "kind"))});
    }
    {
      py::gil_scoped_release release;
      OrtPybindThrowIfError(SessionChain::Create(std::move(options), result->value));
    }
    return result;
  }

  std::unique_ptr<NativeExecution> CreateExecution(
      const py::dict& values, const py::dict& sequences, const py::dict& state) const {
    SessionChainInputs inputs;
    inputs.inputs = Feeds(values);
    inputs.initial_state = Feeds(state);
    for (const auto& item : sequences) {
      auto& destination = inputs.sequences[py::cast<std::string>(item.first)];
      for (const auto& tensor : py::cast<py::tuple>(item.second)) destination.push_back(InputValue(tensor));
    }
    auto result = std::make_unique<NativeExecution>();
    result->sessions = sessions;
    {
      py::gil_scoped_release release;
      OrtPybindThrowIfError(value->CreateExecution(std::move(inputs), result->chain));
    }
    return result;
  }
};

struct NativeCacheView {
  KVCacheView value;
  InlinedVector<InferenceSession*> sessions;
};

struct NativeCache {
  NativeCache() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativeCache);
  std::unique_ptr<KVCacheManager> value;
  InlinedVector<InferenceSession*> sessions;

  static std::unique_ptr<NativeCache> Create(
      const py::tuple& specs, PyInferenceSession* session, py::handle provider) {
    auto result = std::make_unique<NativeCache>();
    InlinedVector<KVCacheSpec> definitions;
    for (const auto& spec : specs) definitions.push_back(Cache(spec));
    auto selection = Provider(provider);
    auto* native = session ? session->GetSessionHandle() : nullptr;
    if (native) result->sessions.push_back(native);
    StatefulProviderBinding binding;
    OrtPybindThrowIfError(StatefulProviderBinding::Resolve(native, selection, binding));
    auto allocator = selection.device_type == OrtMemoryInfoDeviceType_CPU ? GetAllocator() : binding.GetAllocator();
    {
      py::gil_scoped_release release;
      OrtPybindThrowIfError(
          KVCacheManager::Create(std::move(definitions), std::move(allocator), result->value, std::move(binding)));
    }
    return result;
  }

  std::unique_ptr<NativeCacheView> View(const std::string& name) const {
    auto result = std::make_unique<NativeCacheView>();
    result->sessions = sessions;
    OrtPybindThrowIfError(value->Get(name, result->value));
    return result;
  }

  void Update(const std::string& name, py::handle input, int kind) {
    const auto tensor = InputValue(input);
    py::gil_scoped_release release;
    OrtPybindThrowIfError(value->Update(name, tensor, static_cast<KVCacheUpdateKind>(kind)));
  }

  void Reserve(const std::string& name, int64_t capacity) {
    py::gil_scoped_release release;
    OrtPybindThrowIfError(value->Reserve(name, capacity));
  }

  void Reset() {
    py::gil_scoped_release release;
    OrtPybindThrowIfError(value->Reset());
  }

  py::dict Stats() const {
    const auto& stats = value->Stats();
    py::dict result;
    result["allocations"] = stats.allocations;
    result["bytes_copied"] = stats.bytes_copied;
    result["shared_updates"] = stats.shared_updates;
    return result;
  }

  py::dict Dispatch(int operation) const {
    if (operation < 0 || operation >= ORT_STATEFUL_SAMPLE_GREEDY) {
      throw InvalidArgument("Operation does not belong to cache management");
    }
    OrtStatefulDispatchInfo info{};
    OrtPybindThrowIfError(value->Dispatch().GetInfo(static_cast<OrtStatefulProviderOperation>(operation), info));
    return DispatchMetadata(info);
  }
};

struct NativeSampler {
  NativeSampler() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativeSampler);
  std::optional<int64_t> position;
  std::unique_ptr<StatefulDispatch> dispatch;

  static std::unique_ptr<NativeSampler> Create(
      py::handle config, PyInferenceSession* session, py::handle provider) {
    if (Field<int>(config, "algorithm") != 0) throw InvalidArgument("Only greedy sampling is supported");
    auto result = std::make_unique<NativeSampler>();
    result->position = Position(config);
    StatefulProviderBinding binding;
    OrtPybindThrowIfError(
        StatefulProviderBinding::Resolve(session ? session->GetSessionHandle() : nullptr, Provider(provider), binding));
    result->dispatch = std::make_unique<StatefulDispatch>(std::move(binding));
    return result;
  }

  int64_t Sample(py::handle input) {
    const auto value = InputValue(input);
    int64_t token = 0;
    {
      py::gil_scoped_release release;
      OrtPybindThrowIfError(Sampler::Sample(value, position, *dispatch, token));
    }
    return token;
  }

  py::dict Dispatch() const {
    OrtStatefulDispatchInfo info{};
    OrtPybindThrowIfError(dispatch->GetInfo(ORT_STATEFUL_SAMPLE_GREEDY, info));
    return DispatchMetadata(info);
  }
};

template <typename Function>
decltype(auto) PackageCall(Function&& function) {
  try {
    return std::forward<Function>(function)();
  } catch (const Ort::Exception& error) {
    OrtPybindThrowIfError(common::Status(common::ONNXRUNTIME, static_cast<int>(error.GetOrtErrorCode()), error.what()));
    throw;
  }
}

const OrtModelPackageApi& PackageApi() {
  const auto* api = Ort::GetApi().GetModelPackageApi();
  if (!api) throw NotImplemented("Model Package APIs are unavailable in this build");
  return *api;
}

PySessionOptionsSnapshot OptionsSnapshot(PySessionOptions* options) {
  return CreatePySessionOptionsSnapshot(options ? *options : GetDefaultCPUSessionOptions());
}

struct NativeComponent;

struct NativePackage : std::enable_shared_from_this<NativePackage> {
  NativePackage() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativePackage);
  Ort::ModelPackageContext context{nullptr};

  static std::shared_ptr<NativePackage> Open(const std::string& path) {
    PackageApi();
    auto result = std::make_shared<NativePackage>();
    const auto native_path = ToPathString(path);
    {
      py::gil_scoped_release release;
      result->context = PackageCall([&] { return Ort::ModelPackageContext(native_path.c_str()); });
    }
    return result;
  }

  py::dict Info() const {
    return PackageCall([&] {
      py::dict result;
      result["schema_version_major"] = context.GetSchemaVersion();
      py::list components;
      for (const auto& name : context.GetComponentNames()) {
        py::dict component;
        component["name"] = name;
        py::list variants;
        for (const auto& variant_name : context.GetVariantNames(name.c_str())) {
          py::dict variant;
          variant["name"] = variant_name;
          variant["execution_provider"] = context.GetVariantEpName(name.c_str(), variant_name.c_str());
          variants.append(std::move(variant));
        }
        component["variants"] = std::move(variants);
        components.append(std::move(component));
      }
      result["components"] = std::move(components);
      return result;
    });
  }

  std::string Resolve(const std::string& reference, const std::string& base, bool must_exist) const {
    py::gil_scoped_release release;
    return PackageCall([&] { return context.ResolveStringRef(base.c_str(), reference.c_str(), must_exist); });
  }

  std::shared_ptr<NativeComponent> Select(const std::string& name, PySessionOptions* options);

  std::unique_ptr<NativeChain> Chain(
      const std::string& name, PySessionOptions* defaults, const py::dict& steps) const {
    auto result = std::make_unique<NativeChain>();
    result->retained_options.reserve(steps.size() + 1);
    result->retained_options.push_back(OptionsSnapshot(defaults));
    InlinedVector<std::string> names;
    names.reserve(steps.size());
    for (const auto& item : steps) {
      names.push_back(py::cast<std::string>(item.first));
      result->retained_options.push_back(OptionsSnapshot(py::cast<PySessionOptions*>(item.second)));
    }
    InlinedVector<OrtModelPackageStepSessionOptions> overrides;
    overrides.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      overrides.push_back({names[i].c_str(), &result->retained_options[i + 1].options});
    }
    const OrtModelPackageChainConfig config{sizeof(config), name.c_str(),
                                            defaults ? &result->retained_options[0].options : nullptr,
                                            overrides.data(), overrides.size()};
    Ort::SessionChain handle{nullptr};
    {
      py::gil_scoped_release release;
      OrtSessionChain* raw = nullptr;
      PackageCall([&] {
        Ort::ThrowOnError(PackageApi().CreateSessionChain(GetOrtEnv(), context, &config, &raw));
      });
      handle = Ort::SessionChain(raw);
      result->value = raw->value;
      const auto sessions = result->value->Sessions();
      result->sessions.assign(sessions.begin(), sessions.end());
    }
    return result;
  }
};

struct NativeComponent : std::enable_shared_from_this<NativeComponent> {
  explicit NativeComponent(const PySessionOptionsSnapshot& options) : selected_options(options) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NativeComponent);
  std::shared_ptr<NativePackage> package;
  PySessionOptionsSnapshot selected_options;
  Ort::ModelPackageComponentContext context{nullptr};

  py::dict Info() const {
    return PackageCall([&] {
      py::dict result;
      result["variant_name"] = context.GetSelectedVariantName();
      result["variant_folder"] = ToUTF8String(context.GetSelectedVariantFolderPath());
      return result;
    });
  }

  py::tuple Session(PySessionOptions* options) {
    const auto provided = OptionsSnapshot(options);
    auto snapshot = provided;
    std::unique_ptr<InferenceSession> session;
    {
      py::gil_scoped_release release;
      OrtSession* raw = nullptr;
      PackageCall([&] {
        Ort::ThrowOnError(PackageApi().CreateSession(GetOrtEnv(), context, options ? &snapshot.options : nullptr, &raw));
      });
      session.reset(reinterpret_cast<InferenceSession*>(raw));
    }
    const auto& actual = session->GetSessionOptions();
    snapshot.options.value = actual;
    snapshot.py_ep_selection_registration.reset();
    snapshot.py_ep_context_data_read_registration.reset();
    const auto retain_matching = [&](const PySessionOptionsSnapshot& candidate) {
      if (candidate.py_ep_selection_registration &&
          actual.ep_selection_policy.state == candidate.py_ep_selection_registration.get()) {
        snapshot.py_ep_selection_registration = candidate.py_ep_selection_registration;
      }
      if (candidate.py_ep_context_data_read_registration &&
          actual.ep_context_data_read_state == candidate.py_ep_context_data_read_registration.get()) {
        snapshot.py_ep_context_data_read_registration = candidate.py_ep_context_data_read_registration;
      }
    };
    retain_matching(selected_options);
    if (options) retain_matching(provided);
    const auto path = ToUTF8String(session->ModelLocation());
    auto adopted = std::make_unique<PyInferenceSession>(
        std::move(session), std::move(snapshot), shared_from_this());
    return py::make_tuple(py::cast(std::move(adopted)), path);
  }
};

std::shared_ptr<NativeComponent> NativePackage::Select(const std::string& name, PySessionOptions* options) {
  auto result = std::make_shared<NativeComponent>(OptionsSnapshot(options));
  result->package = shared_from_this();
  {
    py::gil_scoped_release release;
    OrtModelPackageOptions* raw_options = nullptr;
    PackageCall([&] {
      Ort::ThrowOnError(PackageApi().CreateModelPackageOptionsFromSessionOptions(
          GetOrtEnv(), &result->selected_options.options, &raw_options));
    });
    Ort::ModelPackageOptions selection(raw_options);
    result->context = PackageCall([&] { return context.SelectComponent(name.c_str(), selection); });
  }
  return result;
}

}  // namespace
#endif

void addStatefulMethods(py::module& parent) {
  auto module = parent.def_submodule("_stateful");
#if !defined(ORT_MINIMAL_BUILD)
  module.attr("available") = true;
  py::class_<NativeValue>(module, "Value")
      .def("copy", &NativeValue::Copy)
      .def("metadata", [](const NativeValue& value) {
        const auto& tensor = value.value.Get<Tensor>();
        py::dict result;
        const auto shape = tensor.Shape().GetDims();
        result["shape"] = std::vector<int64_t>(shape.begin(), shape.end());
        result["element_type"] = static_cast<int>(tensor.GetElementType());
        result["device_type"] = static_cast<int>(tensor.Location().device.Type());
        result["device_id"] = tensor.Location().device.Id();
        return result;
      });
  py::class_<NativeResult>(module, "Result")
      .def("info", &NativeResult::Info)
      .def("outputs", &NativeResult::Outputs);
  py::class_<NativeExecution>(module, "Execution")
      .def("advance", &NativeExecution::Advance)
      .def("cancel", &NativeExecution::Cancel)
      .def("reset", &NativeExecution::Reset)
      .def("requires_reset", &NativeExecution::RequiresReset)
      .def("state", &NativeExecution::State)
      .def("staging", &NativeExecution::Staging)
      .def("dispatch", &NativeExecution::Dispatch);
  py::class_<NativeLoop>(module, "Loop")
      .def_static("create", &NativeLoop::Create)
      .def("create_execution", &NativeLoop::CreateExecution);
  py::class_<NativeChain>(module, "Chain")
      .def_static("create", &NativeChain::Create)
      .def("create_execution", &NativeChain::CreateExecution);
  py::class_<NativeCacheView>(module, "CacheView")
      .def("info", [](const NativeCacheView& view) {
        return py::make_tuple(view.value.length, view.value.capacity, NativeValue{view.value.value, view.sessions});
      });
  py::class_<NativeCache>(module, "Cache")
      .def_static("create", &NativeCache::Create)
      .def("view", &NativeCache::View)
      .def("update", &NativeCache::Update)
      .def("reserve", &NativeCache::Reserve)
      .def("reset", &NativeCache::Reset)
      .def("requires_reset", [](const NativeCache& cache) { return cache.value->RequiresReset(); })
      .def("stats", &NativeCache::Stats)
      .def("dispatch", &NativeCache::Dispatch);
  py::class_<NativeSampler>(module, "Sampler")
      .def_static("create", &NativeSampler::Create)
      .def("sample", &NativeSampler::Sample)
      .def("dispatch", &NativeSampler::Dispatch);
  module.def("capabilities", [](PyInferenceSession& session, const std::string& name) {
    OrtStatefulProviderCapabilities info{};
    OrtPybindThrowIfError(StatefulProviderBinding::Query(*session.GetSessionHandle(), name.c_str(), info));
    py::dict result;
    result["version"] = info.version;
    result["operations"] = info.operations;
    result["cache_element_types"] = info.cache_element_types;
    result["sampler_element_types"] = info.sampler_element_types;
    result["memory_types"] = info.memory_types;
    result["device_id"] = info.device_id;
    return result;
  });
  py::class_<NativePackage, std::shared_ptr<NativePackage>>(module, "Package")
      .def_static("open", &NativePackage::Open)
      .def("info", &NativePackage::Info)
      .def("resolve", &NativePackage::Resolve)
      .def("select", &NativePackage::Select)
      .def("chain", &NativePackage::Chain);
  py::class_<NativeComponent, std::shared_ptr<NativeComponent>>(module, "Component")
      .def("info", &NativeComponent::Info)
      .def("session", &NativeComponent::Session);
#else
  module.attr("available") = false;
#endif
}

}  // namespace onnxruntime::python
