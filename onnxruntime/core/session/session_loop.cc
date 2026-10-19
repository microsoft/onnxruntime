// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/session_loop.h"

#include <algorithm>

#include "core/common/safeint.h"
#include "core/framework/data_transfer.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "core/framework/stateful_execution_utils.h"

namespace onnxruntime {
using namespace stateful_execution;

namespace {
Status CheckScalar(const NodeArg& arg, int element_type) {
  ORT_RETURN_IF_ERROR(CheckTensorType(arg));
  const auto& type = arg.TypeAsProto()->tensor_type();
  ORT_RETURN_IF_NOT(type.elem_type() == element_type &&
                        (!type.has_shape() || type.shape().dim_size() == 0),
                    "SessionLoop condition requires a scalar of the configured type: ", arg.Name());
  return Status::OK();
}

}  // namespace

SessionLoop::SessionLoop(InferenceSession& session, SessionLoopOptions options,
                         StatefulProviderBinding cache_provider, StatefulProviderBinding sampler_provider)
    : session_(session),
      options_(std::move(options)),
      fetch_names_(options_.output_names),
      cache_provider_(std::move(cache_provider)),
      sampler_provider_(std::move(sampler_provider)) {
  auto add_output = [&](const std::string& name) {
    if (FindOutput(fetch_names_, name) == fetch_names_.size()) {
      fetch_names_.push_back(name);
    }
  };
  for (const auto& binding : options_.feedback) {
    add_output(binding.output_name);
  }
  for (const auto& binding : options_.kv_caches) {
    add_output(binding.output_name);
  }
  if (!options_.generated_token_output.empty()) {
    add_output(options_.generated_token_output);
  }
  if (options_.generation) {
    add_output(options_.generation->logits_output);
  }
  for (const auto& condition : options_.exit_conditions) {
    if (condition.kind == LoopExitKind::BooleanOutput) {
      add_output(condition.output_name);
    }
  }
}

Status SessionLoop::Create(InferenceSession& session, SessionLoopOptions options,
                           std::shared_ptr<const SessionLoop>& out) {
  std::unique_ptr<IOBinding> binding;
  ORT_RETURN_IF_ERROR(session.NewIOBinding(&binding));
  ORT_RETURN_IF(options.exit_conditions.empty(), "SessionLoop requires at least one explicit exit condition");
  ORT_RETURN_IF(options.output_names.empty(), "SessionLoop requires at least one selected output");
  const auto& inputs = *session.GetModelInputs().second;
  const auto& outputs = *session.GetModelOutputs().second;
  for (const auto* input : inputs) {
    ORT_RETURN_IF_ERROR(CheckTensorType(*input));
  }
  InlinedHashSet<std::string> destinations;
  for (const auto& name : options.finite_input_names) {
    ORT_RETURN_IF_NOT(FindArgument(inputs, name) && destinations.insert(name).second,
                      "Invalid or duplicate finite input: ", name);
  }
  ORT_RETURN_IF(!options.nonempty_input.empty() &&
                    std::find(options.finite_input_names.begin(), options.finite_input_names.end(),
                              options.nonempty_input) == options.finite_input_names.end(),
                "Nonempty input must name a finite input");
  for (const auto& binding : options.feedback) {
    const auto* input = FindArgument(inputs, binding.input_name);
    const auto* output = FindArgument(outputs, binding.output_name);
    ORT_RETURN_IF_NOT(input && output && destinations.insert(binding.input_name).second,
                      "Invalid or conflicting feedback binding: ", binding.input_name);
    ORT_RETURN_IF_ERROR(CheckCompatibleTypes(*output, *input));
  }
  InlinedVector<KVCacheSpec> cache_specs;
  for (const auto& item : options.kv_caches) {
    cache_specs.push_back(item.cache);
  }
  ORT_RETURN_IF_ERROR(KVCacheManager::ValidateSpecs(cache_specs));
  for (const auto& item : options.kv_caches) {
    const auto* input = FindArgument(inputs, item.input_name);
    const auto* output = FindArgument(outputs, item.output_name);
    ORT_RETURN_IF_NOT(input && output && destinations.insert(item.input_name).second,
                      "Invalid or conflicting KV cache binding: ", item.cache.name);
    ORT_RETURN_IF(item.update_kind != KVCacheUpdateKind::Share && item.update_kind != KVCacheUpdateKind::Copy &&
                      item.update_kind != KVCacheUpdateKind::Append,
                  "Unknown KV cache binding update kind");
    ORT_RETURN_IF_ERROR(KVCacheManager::CheckBinding(item.cache, *input, true));
    ORT_RETURN_IF_ERROR(KVCacheManager::CheckBinding(item.cache, *output, false));
  }
  if (options.generation) {
    const auto& generation = *options.generation;
    ORT_RETURN_IF_ERROR(TokenGeneration::Validate(generation, inputs, outputs));
    ORT_RETURN_IF(!options.generated_token_output.empty(), "Native sampling cannot also use a model token output");
    if (generation.phase == TokenGenerationPhase::Prefill) {
      ORT_RETURN_IF(options.finite_input_names.size() != 1 || options.finite_input_names[0] != generation.input_ids,
                    "Prefill requires exactly one finite prompt input");
      ORT_RETURN_IF(std::none_of(options.exit_conditions.begin(), options.exit_conditions.end(),
                                 [](const auto& condition) { return condition.kind == LoopExitKind::EndOfInputs; }),
                    "Prefill requires an explicit EndOfInputs condition");
    } else {
      ORT_RETURN_IF(!options.finite_input_names.empty(), "Decode inputs are supplied by the native generation policy");
    }
    for (const auto& name : TokenGeneration::ManagedInputs(generation)) {
      ORT_RETURN_IF_NOT(destinations.insert(name).second, "Conflicting managed generation input: ", name);
    }
  }
  InlinedHashSet<std::string> selected;
  for (const auto& name : options.output_names) {
    const auto* output = FindArgument(outputs, name);
    ORT_RETURN_IF_NOT(output && selected.insert(name).second, "Invalid or duplicate selected output: ", name);
    ORT_RETURN_IF_ERROR(CheckTensorType(*output));
  }
  if (!options.generated_token_output.empty()) {
    const auto* output = FindArgument(outputs, options.generated_token_output);
    ORT_RETURN_IF_NOT(output, "Unknown generated-token output");
    ORT_RETURN_IF_ERROR(CheckScalar(*output, ONNX_NAMESPACE::TensorProto_DataType_INT64));
  }
  InlinedHashSet<LoopExitKind> kinds;
  for (const auto& condition : options.exit_conditions) {
    ORT_RETURN_IF_NOT(kinds.insert(condition.kind).second, "Duplicate SessionLoop exit condition");
    switch (condition.kind) {
      case LoopExitKind::EndOfInputs:
        ORT_RETURN_IF(options.finite_input_names.empty(), "EndOfInputs requires a finite input source");
        break;
      case LoopExitKind::MaxIterations:
      case LoopExitKind::MaxGeneratedTokens:
        ORT_RETURN_IF(condition.limit < 0, "SessionLoop limits must be nonnegative");
        if (condition.kind == LoopExitKind::MaxIterations) {
          break;
        }
        [[fallthrough]];
      case LoopExitKind::Eos:
        ORT_RETURN_IF(options.generated_token_output.empty() && !options.generation,
                      "Token exits require a generated-token output or native sampling policy");
        if (condition.kind == LoopExitKind::Eos) {
          ORT_RETURN_IF(condition.eos_token_ids.empty() ||
                            std::any_of(condition.eos_token_ids.begin(), condition.eos_token_ids.end(),
                                        [](int64_t id) { return id < 0; }),
                        "EOS requires nonnegative token IDs");
          if (options.generation) {
            const auto* logits = FindArgument(outputs, options.generation->logits_output);
            const auto* shape = logits->Shape();
            if (shape && shape->dim(shape->dim_size() - 1).has_dim_value()) {
              const int64_t width = shape->dim(shape->dim_size() - 1).dim_value();
              ORT_RETURN_IF(std::any_of(condition.eos_token_ids.begin(), condition.eos_token_ids.end(),
                                        [&](int64_t id) { return id >= width; }),
                            "EOS token ID is outside the logits vocabulary");
            }
          }
        }
        break;
      case LoopExitKind::BooleanOutput: {
        const auto* output = FindArgument(outputs, condition.output_name);
        ORT_RETURN_IF_NOT(output, "Unknown boolean exit output");
        ORT_RETURN_IF_ERROR(CheckScalar(*output, ONNX_NAMESPACE::TensorProto_DataType_BOOL));
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown SessionLoop exit kind");
    }
  }
  StatefulProviderBinding cache_provider, sampler_provider;
  ORT_RETURN_IF_ERROR(StatefulProviderBinding::Resolve(&session, options.cache_provider, cache_provider));
  ORT_RETURN_IF_ERROR(StatefulProviderBinding::Resolve(&session, options.sampler_provider, sampler_provider));
  ORT_RETURN_IF((options.kv_caches.empty() && options.cache_provider.mode != ORT_STATEFUL_PROVIDER_CPU) ||
                    (!options.generation && options.sampler_provider.mode != ORT_STATEFUL_PROVIDER_CPU),
                "Provider policy requires the corresponding managed cache/generation feature");
  StatefulDispatch cache_dispatch(cache_provider), sampler_dispatch(sampler_provider);
  bool provider_selected = false;
  for (const auto& item : options.kv_caches) {
    for (const auto operation : {ORT_STATEFUL_CACHE_INITIALIZE, ORT_STATEFUL_CACHE_RESET,
                                 static_cast<OrtStatefulProviderOperation>(static_cast<int>(item.update_kind) + 1)}) {
      ORT_RETURN_IF_ERROR(cache_dispatch.Select(operation, item.cache.element_type, true, provider_selected));
    }
  }
  if (options.generation) {
    const auto* logits = FindArgument(outputs, options.generation->logits_output);
    const auto element_type = logits->TypeAsProto()->tensor_type().elem_type();
    ORT_RETURN_IF_ERROR(
        sampler_dispatch.Select(ORT_STATEFUL_SAMPLE_GREEDY, element_type, true, provider_selected));
  }
  if (options.sampler_provider.device_type != OrtMemoryInfoDeviceType_CPU) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, NOT_IMPLEMENTED, "Device-native logits Session integration is unavailable");
  }
  auto loop = std::shared_ptr<SessionLoop>(
      new SessionLoop(session, std::move(options), std::move(cache_provider), std::move(sampler_provider)));
  if (loop->cache_provider_.Selection().device_type != OrtMemoryInfoDeviceType_CPU) {
    InlinedHashSet<std::string> device_inputs, device_outputs;
    for (const auto& item : loop->options_.kv_caches) {
      device_inputs.insert(item.input_name);
      device_outputs.insert(item.output_name);
    }
    ORT_RETURN_IF_ERROR(loop->ConfigureDeviceCache(
        loop->cache_provider_, std::move(device_inputs), std::move(device_outputs)));
  }
  out = std::move(loop);
  return Status::OK();
}

Status SessionLoop::ConfigureDeviceCache(const StatefulProviderBinding& provider,
                                         InlinedHashSet<std::string> inputs,
                                         InlinedHashSet<std::string> outputs) {
  const auto& selection = provider.Selection();
  ORT_RETURN_IF_NOT(selection.device_type == OrtMemoryInfoDeviceType_GPU &&
                        selection.provider_name == kCudaExecutionProvider,
                    "Device-cache Session integration requires the CUDA plugin");
  StatefulProviderBinding local;
  ORT_RETURN_IF_ERROR(StatefulProviderBinding::Resolve(&session_, selection, local));
  ORT_RETURN_IF_NOT(provider.CacheCompatible(local), "Incompatible device-cache Session provider");
  ORT_RETURN_IF(options_.generation && outputs.contains(options_.generation->logits_output),
                "Device cache output cannot also supply CPU sampler logits");
  const auto& state = session_.GetSessionState();
  for (const auto& [device, allocator] : state.GetAllocators()) {
    if (device.Type() == OrtDevice::GPU && device.Id() == selection.device_id &&
        device.MemType() == OrtDevice::MemType::DEFAULT) {
      device_cache_allocator_ = allocator;
      break;
    }
  }
  ORT_RETURN_IF_NOT(device_cache_allocator_, "Selected device-cache Session allocator is unavailable");
  const auto& device = device_cache_allocator_->Info().device;
  for (const auto& feedback : options_.feedback) {
    if (outputs.contains(feedback.output_name)) inputs.insert(feedback.input_name);
  }
  auto check_ports = [&](const auto& names, bool is_input) -> Status {
    for (const auto& name : names) {
      InlinedVector<SessionState::NodeInfo> nodes;
      ORT_RETURN_IF_ERROR(is_input ? state.GetInputNodeInfo(name, nodes) : state.GetOutputNodeInfo(name, nodes));
      ORT_RETURN_IF(nodes.empty() ||
                        std::any_of(nodes.begin(), nodes.end(), [&](const auto& node) {
                          return !node.device || *node.device != device;
                        }),
                    "Device-cache port requires the selected GPU without staging: ", name);
    }
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(check_ports(inputs, true));
  ORT_RETURN_IF_ERROR(check_ports(outputs, false));
  device_cache_provider_ = std::move(local);
  device_cache_inputs_ = std::move(inputs);
  device_cache_outputs_ = std::move(outputs);
  return Status::OK();
}

Status SessionLoop::WithDeviceCachePorts(const StatefulProviderBinding& provider,
                                         InlinedHashSet<std::string> inputs,
                                         InlinedHashSet<std::string> outputs,
                                         std::shared_ptr<const SessionLoop>& out) const {
  auto loop = std::shared_ptr<SessionLoop>(new SessionLoop(session_, options_, cache_provider_, sampler_provider_));
  ORT_RETURN_IF_ERROR(loop->ConfigureDeviceCache(provider, std::move(inputs), std::move(outputs)));
  out = std::move(loop);
  return Status::OK();
}

Status SessionLoop::ValidateInputs(const SessionLoopInputs& inputs) const {
  const auto& arguments = *session_.GetModelInputs().second;
  InlinedHashSet<std::string> finite(options_.finite_input_names.begin(), options_.finite_input_names.end());
  const auto managed = options_.generation ? TokenGeneration::ManagedInputs(*options_.generation)
                                           : InlinedVector<std::string>{};
  ORT_RETURN_IF(inputs.inputs.size() + finite.size() + options_.kv_caches.size() + managed.size() != arguments.size(),
                "SessionLoop input count mismatch");
  for (const auto& [name, value] : inputs.inputs) {
    const auto* argument = FindArgument(arguments, name);
    ORT_RETURN_IF_NOT(argument && !finite.contains(name), "Unknown or conflicting invariant input: ", name);
    ORT_RETURN_IF(std::find(managed.begin(), managed.end(), name) != managed.end(),
                  "Managed generation inputs cannot also be invariant feeds");
    ORT_RETURN_IF(std::any_of(options_.kv_caches.begin(), options_.kv_caches.end(),
                              [&](const auto& item) { return item.input_name == name; }),
                  "Managed KV cache inputs cannot also be supplied as invariant feeds");
    ORT_RETURN_IF_ERROR(CheckInput(*argument, value));
    if (device_cache_inputs_.contains(name)) {
      ORT_RETURN_IF(value.Get<Tensor>().Location().device != device_cache_allocator_->Info().device,
                    "Device-cache input requires matching GPU storage: ", name);
    }
  }
  ORT_RETURN_IF(finite.empty() && !inputs.batches.empty(), "Batches require finite input names");
  ORT_RETURN_IF(!options_.nonempty_input.empty() && inputs.batches.empty(), "Prefill input must not be empty");
  for (const auto& batch : inputs.batches) {
    ORT_RETURN_IF(batch.size() != finite.size(), "SessionLoop finite batch input count mismatch");
    for (const auto& [name, value] : batch) {
      ORT_RETURN_IF_NOT(finite.contains(name), "Unknown finite input: ", name);
      ORT_RETURN_IF_ERROR(CheckInput(*FindArgument(arguments, name), value));
      if (device_cache_inputs_.contains(name)) {
        ORT_RETURN_IF(value.Get<Tensor>().Location().device != device_cache_allocator_->Info().device,
                      "Device-cache finite input requires matching GPU storage: ", name);
      }
      ORT_RETURN_IF(name == options_.nonempty_input && value.Get<Tensor>().Shape().Size() == 0,
                    "Prefill input tensor must not be empty");
    }
  }
  if (options_.generation && options_.generation->phase == TokenGenerationPhase::Prefill) {
    ORT_RETURN_IF(inputs.batches.size() != 1, "Initial prefill supports one complete prompt batch");
    ORT_RETURN_IF_ERROR(TokenGeneration::ValidatePrompt(inputs.batches[0].at(options_.generation->input_ids)));
  }
  return Status::OK();
}

Status SessionLoop::CreateExecution(SessionLoopInputs inputs, std::unique_ptr<SessionLoopExecution>& out) const {
  ORT_RETURN_IF_ERROR(ValidateInputs(inputs));
  ORT_RETURN_IF(options_.generation && options_.generation->phase == TokenGenerationPhase::Decode,
                "Decode generation must follow prefill in the same chain context");
  std::shared_ptr<KVCacheManager> cache;
  if (!options_.kv_caches.empty()) {
    InlinedVector<KVCacheSpec> specs;
    for (const auto& item : options_.kv_caches) {
      specs.push_back(item.cache);
    }
    std::unique_ptr<IOBinding> binding;
    ORT_RETURN_IF_ERROR(session_.NewIOBinding(&binding));
    std::unique_ptr<KVCacheManager> manager;
    auto allocator =
        device_cache_allocator_ ? device_cache_allocator_ : binding->GetCPUAllocator(kCpuExecutionProvider);
    ORT_RETURN_IF_ERROR(KVCacheManager::Create(std::move(specs), std::move(allocator), manager, cache_provider_));
    cache = std::move(manager);
  }
  auto generation = options_.generation ? std::make_shared<TokenGenerationState>() : nullptr;
  auto sampler = options_.generation ? std::make_shared<StatefulDispatch>(sampler_provider_) : nullptr;
  out = std::unique_ptr<SessionLoopExecution>(
      new SessionLoopExecution(shared_from_this(), std::move(inputs), std::move(cache), std::move(generation),
                               std::move(sampler), true));
  return Status::OK();
}

Status SessionLoop::CreateExecutionWithCache(SessionLoopInputs inputs, std::shared_ptr<KVCacheManager> cache,
                                             std::shared_ptr<TokenGenerationState> generation,
                                             std::shared_ptr<StatefulDispatch> sampler,
                                             std::unique_ptr<SessionLoopExecution>& out) const {
  ORT_RETURN_IF_ERROR(ValidateInputs(inputs));
  out = std::unique_ptr<SessionLoopExecution>(
      new SessionLoopExecution(shared_from_this(), std::move(inputs), std::move(cache), std::move(generation),
                               std::move(sampler), false));
  return Status::OK();
}

SessionLoopExecution::SessionLoopExecution(std::shared_ptr<const SessionLoop> loop, SessionLoopInputs inputs,
                                           std::shared_ptr<KVCacheManager> cache,
                                           std::shared_ptr<TokenGenerationState> generation,
                                           std::shared_ptr<StatefulDispatch> sampler, bool owns_state)
    : loop_(std::move(loop)),
      initial_inputs_(std::move(inputs)),
      cache_(std::move(cache)),
      generation_(std::move(generation)),
      sampler_(std::move(sampler)),
      owns_state_(owns_state) {}

void SessionLoopExecution::EvaluateExits(const InlinedVector<OrtValue>& values) {
  progress_.exit_reasons.clear();
  const auto& options = loop_->options_;
  if (options.generation) {
    progress_.generated_tokens = SafeInt<int64_t>(generation_->token_ids.size());
  }
  for (const auto& condition : options.exit_conditions) {
    bool stop = false;
    switch (condition.kind) {
      case LoopExitKind::EndOfInputs:
        stop = static_cast<uint64_t>(progress_.iterations) >= initial_inputs_.batches.size();
        break;
      case LoopExitKind::MaxIterations:
        stop = progress_.iterations >= condition.limit;
        break;
      case LoopExitKind::MaxGeneratedTokens:
        stop = progress_.generated_tokens >= condition.limit;
        break;
      case LoopExitKind::Eos:
        if ((options.generation && !generation_->token_ids.empty()) || !values.empty()) {
          const int64_t token = options.generation
                                    ? generation_->token_ids.back()
                                    : *values[FindOutput(loop_->fetch_names_, options.generated_token_output)]
                                           .Get<Tensor>()
                                           .Data<int64_t>();
          stop = std::find(condition.eos_token_ids.begin(), condition.eos_token_ids.end(), token) !=
                 condition.eos_token_ids.end();
        }
        break;
      case LoopExitKind::BooleanOutput:
        stop = !values.empty() &&
               *values[FindOutput(loop_->fetch_names_, condition.output_name)].Get<Tensor>().Data<bool>();
        break;
    }
    if (stop) {
      progress_.exit_reasons.push_back(condition.kind);
    }
  }
  std::sort(progress_.exit_reasons.begin(), progress_.exit_reasons.end());
  progress_.completed = !progress_.exit_reasons.empty();
}

Status SessionLoopExecution::Step(SessionLoopStepResult& out) {
  ORT_RETURN_IF(requires_reset_, "SessionLoop execution requires Reset after a failure or cancellation");
  if (cancelled_.load()) {
    requires_reset_ = true;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SessionLoop execution cancelled; call Reset before reuse");
  }
  if (!progress_.completed) {
    EvaluateExits({});
  }
  if (progress_.completed) {
    out = progress_;
    return Status::OK();
  }
  ORT_RETURN_IF(!loop_->options_.finite_input_names.empty() &&
                    static_cast<uint64_t>(progress_.iterations) >= initial_inputs_.batches.size(),
                "SessionLoop finite inputs exhausted without EndOfInputs");

  // Any exception or error after this point conservatively invalidates mutable state.
  requires_reset_ = true;
  std::unique_ptr<IOBinding> binding;
  ORT_RETURN_IF_ERROR(loop_->session_.NewIOBinding(&binding));
  auto bind_input = [&](const std::string& name, const OrtValue& value) -> Status {
    ORT_RETURN_IF_ERROR(binding->BindInput(name, value));
    const auto& tensor = value.Get<Tensor>();
    if (tensor.Location().device.UsesCpuMemory()) {
      staging_.cpu_input_request_bytes = SafeInt<size_t>(staging_.cpu_input_request_bytes) + tensor.SizeInBytes();
    }
    return Status::OK();
  };
  for (const auto& [name, value] : initial_inputs_.inputs) {
    ORT_RETURN_IF_ERROR(bind_input(name, value));
  }
  if (!loop_->options_.finite_input_names.empty()) {
    for (const auto& [name, value] : initial_inputs_.batches[static_cast<size_t>(progress_.iterations)]) {
      ORT_RETURN_IF_ERROR(bind_input(name, value));
    }
  }
  for (size_t i = 0; i < feedback_.size(); ++i) {
    ORT_RETURN_IF_ERROR(bind_input(loop_->options_.feedback[i].input_name, feedback_[i]));
  }
  for (const auto& item : loop_->options_.kv_caches) {
    KVCacheView view;
    ORT_RETURN_IF_ERROR(cache_->Get(item.cache.name, view));
    ORT_RETURN_IF(loop_->options_.generation && view.length != generation_->processed_tokens,
                  "KV length does not match the generation context before invocation");
    ORT_RETURN_IF_ERROR(bind_input(item.input_name, view.value));
  }
  int64_t input_count = 0;
  LoopFeeds generated_feeds;
  if (loop_->options_.generation) {
    const auto& options = *loop_->options_.generation;
    const OrtValue* prompt = options.phase == TokenGenerationPhase::Prefill
                                 ? &initial_inputs_.batches[0].at(options.input_ids)
                                 : nullptr;
    ORT_RETURN_IF_ERROR(TokenGeneration::PrepareInputs(
        options, *generation_, prompt, binding->GetCPUAllocator(kCpuExecutionProvider), generated_feeds, input_count));
    for (const auto& [name, value] : generated_feeds) {
      ORT_RETURN_IF_ERROR(bind_input(name, value));
    }
  }
  for (const auto& name : loop_->fetch_names_) {
    ORT_RETURN_IF_ERROR(binding->BindOutput(
        name, loop_->device_cache_outputs_.contains(name)
                  ? loop_->device_cache_allocator_->Info().device
                  : OrtDevice{}));
  }
  if (!loop_->device_cache_provider_) {
    ORT_RETURN_IF_ERROR(binding->SynchronizeInputs());
  }
  RunOptions run_options;
  ORT_RETURN_IF_ERROR(loop_->session_.Run(run_options, *binding));
  if (!loop_->device_cache_provider_) {
    ORT_RETURN_IF_ERROR(binding->SynchronizeOutputs());
  }
  ORT_RETURN_IF(cancelled_.load(), "SessionLoop execution cancelled during a step; call Reset before reuse");

  InlinedVector<OrtValue> snapshots;
  snapshots.reserve(binding->GetOutputs().size());
  const auto allocator = binding->GetCPUAllocator(kCpuExecutionProvider);
  CPUDataTransfer transfer;
  for (const auto& value : binding->GetOutputs()) {
    const Tensor& source = value.Get<Tensor>();
    if (!source.Location().device.UsesCpuMemory()) {
      ORT_RETURN_IF_NOT(loop_->device_cache_provider_ &&
                            source.Location().device == loop_->device_cache_allocator_->Info().device,
                        "Unexpected device output in stateful execution");
      snapshots.push_back(loop_->device_cache_provider_->RetainValue(value, loop_->device_cache_allocator_));
      continue;
    }
    if (source.Location().device.UsesCpuMemory()) {
      staging_.cpu_output_bytes = SafeInt<size_t>(staging_.cpu_output_bytes) + source.SizeInBytes();
    }
    OrtValue snapshot;
    Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, snapshot);
    if (source.SizeInBytes() != 0) {
      ORT_RETURN_IF_ERROR(transfer.CopyTensor(source, *snapshot.GetMutable<Tensor>()));
      staging_.cpu_snapshot_copy_bytes = SafeInt<size_t>(staging_.cpu_snapshot_copy_bytes) + source.SizeInBytes();
    }
    snapshots.push_back(std::move(snapshot));
  }
  for (const auto& condition : loop_->options_.exit_conditions) {
    if (condition.kind == LoopExitKind::BooleanOutput) {
      ORT_RETURN_IF(snapshots[FindOutput(loop_->fetch_names_, condition.output_name)]
                            .Get<Tensor>()
                            .Shape()
                            .NumDimensions() != 0,
                    "Boolean exit output must be scalar");
    }
  }
  std::optional<int64_t> sampled_token;
  if (loop_->options_.generation) {
    const auto& options = *loop_->options_.generation;
    int64_t token = 0;
    ORT_RETURN_IF_ERROR(Sampler::Sample(snapshots[FindOutput(loop_->fetch_names_, options.logits_output)],
                                        options.logits_position, *sampler_, token));
    const auto& shape = snapshots[FindOutput(loop_->fetch_names_, options.logits_output)].Get<Tensor>().Shape();
    for (const auto& condition : loop_->options_.exit_conditions) {
      if (condition.kind == LoopExitKind::Eos) {
        ORT_RETURN_IF(std::any_of(condition.eos_token_ids.begin(), condition.eos_token_ids.end(),
                                  [&](int64_t id) { return id >= shape[shape.NumDimensions() - 1]; }),
                      "EOS token ID is outside the runtime logits vocabulary");
      }
    }
    sampled_token = token;
  }
  if (!loop_->options_.generated_token_output.empty()) {
    const auto& token =
        snapshots[FindOutput(loop_->fetch_names_, loop_->options_.generated_token_output)].Get<Tensor>();
    ORT_RETURN_IF(token.Shape().NumDimensions() != 0 || *token.Data<int64_t>() < 0,
                  "Generated-token output must be a nonnegative scalar");
    progress_.generated_tokens = SafeInt<int64_t>(progress_.generated_tokens) + 1;
    sampled_token = *token.Data<int64_t>();
  }
  for (const auto& item : loop_->options_.kv_caches) {
    ORT_RETURN_IF_ERROR(cache_->Update(item.cache.name, snapshots[FindOutput(loop_->fetch_names_, item.output_name)],
                                       item.update_kind));
    if (loop_->options_.generation) {
      KVCacheView view;
      ORT_RETURN_IF_ERROR(cache_->Get(item.cache.name, view));
      ORT_RETURN_IF(view.length != SafeInt<int64_t>(generation_->processed_tokens) + input_count,
                    "KV length does not match the consumed generation input");
    }
  }
  if (loop_->options_.generation) {
    generation_->processed_tokens = SafeInt<int64_t>(generation_->processed_tokens) + input_count;
    generation_->token_ids.push_back(*sampled_token);
    generation_->prefill_complete = true;
  }
  progress_.iterations = SafeInt<int64_t>(progress_.iterations) + 1;
  EvaluateExits(snapshots);
  feedback_.clear();
  for (const auto& item : loop_->options_.feedback) {
    feedback_.push_back(snapshots[FindOutput(loop_->fetch_names_, item.output_name)]);
  }
  SessionLoopStepResult result = progress_;
  result.executed = true;
  result.generated_token = sampled_token;
  for (const auto& name : loop_->options_.output_names) {
    result.outputs.push_back(snapshots[FindOutput(loop_->fetch_names_, name)]);
  }
  requires_reset_ = false;
  out = std::move(result);
  return Status::OK();
}

Status SessionLoopExecution::Run(SessionLoopStepResult& out) {
  SessionLoopStepResult result;
  do {
    ORT_RETURN_IF_ERROR(Step(result));
  } while (!result.completed);
  out = std::move(result);
  return Status::OK();
}

Status SessionLoopExecution::GetState(LoopFeeds& out) const {
  ORT_RETURN_IF(requires_reset_, "SessionLoop state requires Reset after a failed execution");
  LoopFeeds state;
  for (size_t i = 0; i < loop_->options_.feedback.size(); ++i) {
    const auto& name = loop_->options_.feedback[i].input_name;
    state.emplace(name, feedback_.empty() ? initial_inputs_.inputs.at(name) : feedback_[i]);
  }
  for (const auto& binding : loop_->options_.kv_caches) {
    KVCacheView view;
    ORT_RETURN_IF_ERROR(cache_->Get(binding.cache.name, view));
    ORT_RETURN_IF_NOT(state.emplace(binding.cache.name, std::move(view.value)).second,
                      "Feedback and KV names collide in loop state inspection: ", binding.cache.name);
  }
  out = std::move(state);
  return Status::OK();
}

Status SessionLoopExecution::GetDispatchInfo(
    OrtStatefulProviderFeature feature, size_t component,
    OrtStatefulProviderOperation operation, OrtStatefulDispatchInfo& out) const {
  ORT_RETURN_IF(component != 0, "Standalone loop component index must be zero");
  ORT_RETURN_IF(feature != ORT_STATEFUL_FEATURE_CACHE && feature != ORT_STATEFUL_FEATURE_SAMPLER,
                "Unknown stateful provider feature");
  ORT_RETURN_IF(feature == ORT_STATEFUL_FEATURE_CACHE ? operation == ORT_STATEFUL_SAMPLE_GREEDY
                                                      : operation != ORT_STATEFUL_SAMPLE_GREEDY,
                "Operation does not belong to the selected feature");
  ORT_RETURN_IF(feature == ORT_STATEFUL_FEATURE_CACHE ? !cache_ : !sampler_, "Stateful feature is not configured");
  return (feature == ORT_STATEFUL_FEATURE_CACHE ? cache_->Dispatch() : *sampler_).GetInfo(operation, out);
}

Status SessionLoopExecution::GetStagingInfo(size_t component, OrtStatefulStagingInfo& out) const {
  ORT_RETURN_IF(component != 0, "Standalone loop component index must be zero");
  out = staging_;
  return Status::OK();
}

Status SessionLoopExecution::Reset() {
  requires_reset_ = true;
  if (owns_state_) {
    if (cache_) {
      ORT_RETURN_IF_ERROR(cache_->Reset());
    }
    if (generation_) {
      generation_->Reset();
    }
  }
  feedback_.clear();
  progress_ = {};
  cancelled_.store(false);
  requires_reset_ = false;
  return Status::OK();
}

}  // namespace onnxruntime
