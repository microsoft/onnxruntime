// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/model_package/model_package_chain.h"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>

namespace onnxruntime {
namespace {

using Json = nlohmann::ordered_json;
using StepIndices = InlinedHashMap<std::string, size_t>;
using CacheSpecs = InlinedHashMap<std::string, KVCacheSpec>;

Status Object(const Json& value, std::initializer_list<std::string_view> keys) {
  ORT_RETURN_IF(!value.is_object(), "Expected an ORT configuration object");
  for (auto it = value.begin(); it != value.end(); ++it) {
    ORT_RETURN_IF(std::find(keys.begin(), keys.end(), it.key()) == keys.end(),
                  "Unknown ORT configuration field: ", it.key());
  }
  return Status::OK();
}

Status Name(const Json& value, std::string& out) {
  ORT_RETURN_IF(!value.is_string(), "Expected a string name");
  const auto& name = value.get_ref<const std::string&>();
  ORT_RETURN_IF(name.empty() || name.find('\0') != std::string::npos, "Names must be nonempty and contain no NUL");
  out = name;
  return Status::OK();
}

Status String(const Json& value, const char* key, std::string& out, bool required = true) {
  const auto it = value.find(key);
  if (it == value.end()) {
    ORT_RETURN_IF(required, "Missing required field: ", key);
    return Status::OK();
  }
  return Name(*it, out);
}

Status Integer(const Json& value, int64_t& out) {
  ORT_RETURN_IF(!value.is_number_integer() ||
                    (value.is_number_unsigned() &&
                     value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())),
                "Expected an int64 integer");
  out = value.get<int64_t>();
  return Status::OK();
}

Status IntField(const Json& value, const char* key, int64_t& out, bool required = true) {
  const auto it = value.find(key);
  if (it == value.end()) {
    ORT_RETURN_IF(required, "Missing required integer: ", key);
    return Status::OK();
  }
  return Integer(*it, out);
}

Status Array(const Json& value) {
  ORT_RETURN_IF(!value.is_array(), "Expected an array");
  return Status::OK();
}

Status Provider(const Json& value, StatefulProviderSelection& out) {
  ORT_RETURN_IF_ERROR(Object(value, {"mode", "provider_name", "device_type", "device_id"}));
  std::string mode, name, device;
  ORT_RETURN_IF_ERROR(String(value, "mode", mode));
  ORT_RETURN_IF_ERROR(String(value, "provider_name", name, mode != "cpu"));
  ORT_RETURN_IF_ERROR(String(value, "device_type", device, mode != "cpu"));
  int64_t id = 0;
  ORT_RETURN_IF_ERROR(IntField(value, "device_id", id, mode != "cpu"));
  ORT_RETURN_IF(id < 0 || id > std::numeric_limits<int>::max(), "Invalid provider device_id");
  OrtStatefulProviderSelection config{};
  config.struct_size = sizeof(config);
  config.provider_name = name.c_str();
  config.device_id = static_cast<int>(id);
  if (mode == "cpu")
    config.mode = ORT_STATEFUL_PROVIDER_CPU;
  else if (mode == "prefer_provider")
    config.mode = ORT_STATEFUL_PROVIDER_PREFER;
  else if (mode == "require_provider")
    config.mode = ORT_STATEFUL_PROVIDER_REQUIRE;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown provider mode");
  if (device.empty() || device == "cpu")
    config.device_type = OrtMemoryInfoDeviceType_CPU;
  else if (device == "gpu")
    config.device_type = OrtMemoryInfoDeviceType_GPU;
  else if (device == "npu")
    config.device_type = OrtMemoryInfoDeviceType_NPU;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown provider device_type");
  return StatefulProviderSelection::Parse(&config, out);
}

Status Source(const Json& value, const StepIndices& indices, size_t before, ChainValueSource& out,
              bool initial_logits = false) {
  std::string kind;
  ORT_RETURN_IF_ERROR(String(value, "kind", kind));
  if (kind == "model_output") {
    ORT_RETURN_IF_ERROR(initial_logits ? Object(value, {"kind", "step", "output", "prompt_input", "position"})
                                       : Object(value, {"kind", "step", "output"}));
    std::string step;
    ORT_RETURN_IF_ERROR(String(value, "step", step));
    const auto it = indices.find(step);
    ORT_RETURN_IF(it == indices.end() || it->second >= before, "Unknown or non-upstream step: ", step);
    out.kind = ChainSourceKind::ComponentOutput;
    out.component_index = it->second;
    return String(value, "output", out.name);
  }
  ORT_RETURN_IF(initial_logits, "Initial logits must be an upstream model_output");
  ORT_RETURN_IF_ERROR(Object(value, {"kind", "name"}));
  ORT_RETURN_IF_ERROR(String(value, "name", out.name));
  if (kind == "external_input")
    out.kind = ChainSourceKind::ExternalInput;
  else if (kind == "external_sequence")
    out.kind = ChainSourceKind::ExternalSequence;
  else if (kind == "state")
    out.kind = ChainSourceKind::State;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown input source kind");
  return Status::OK();
}

Status Specs(const Json& value, CacheSpecs& out) {
  ORT_RETURN_IF(!value.is_object(), "kv_caches specifications must be an object");
  for (auto it = value.begin(); it != value.end(); ++it) {
    ORT_RETURN_IF_ERROR(Object(*it, {"element_type", "shape", "sequence_axis", "initial_capacity"}));
    KVCacheSpec spec;
    ORT_RETURN_IF_ERROR(Name(Json(it.key()), spec.name));
    std::string type;
    ORT_RETURN_IF_ERROR(String(*it, "element_type", type));
    if (type == "float32")
      spec.element_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    else if (type == "float16")
      spec.element_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    else if (type == "bfloat16")
      spec.element_type = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    else
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown cache element_type");
    const auto shape = it->find("shape");
    ORT_RETURN_IF(shape == it->end(), "Missing cache shape");
    ORT_RETURN_IF_ERROR(Array(*shape));
    for (const auto& dimension : *shape) {
      int64_t size = 0;
      ORT_RETURN_IF_ERROR(Integer(dimension, size));
      spec.shape.push_back(size);
    }
    ORT_RETURN_IF_ERROR(IntField(*it, "sequence_axis", spec.sequence_axis));
    ORT_RETURN_IF_ERROR(IntField(*it, "initial_capacity", spec.initial_capacity));
    ORT_RETURN_IF_ERROR(KVCacheManager::ValidateSpecs(gsl::span<const KVCacheSpec>(&spec, 1)));
    out.emplace(spec.name, std::move(spec));
  }
  return Status::OK();
}

Status CacheBindings(const Json& value, const CacheSpecs& specs, InlinedVector<LoopKVCacheBinding, 1>& out) {
  ORT_RETURN_IF_ERROR(Array(value));
  for (const auto& item : value) {
    ORT_RETURN_IF_ERROR(Object(item, {"cache", "input", "output", "update"}));
    LoopKVCacheBinding binding;
    std::string name, update;
    ORT_RETURN_IF_ERROR(String(item, "cache", name));
    const auto spec = specs.find(name);
    ORT_RETURN_IF(spec == specs.end(), "Unknown managed cache: ", name);
    binding.cache = spec->second;
    ORT_RETURN_IF_ERROR(String(item, "input", binding.input_name));
    ORT_RETURN_IF_ERROR(String(item, "output", binding.output_name));
    ORT_RETURN_IF_ERROR(String(item, "update", update));
    if (update == "share")
      binding.update_kind = KVCacheUpdateKind::Share;
    else if (update == "copy")
      binding.update_kind = KVCacheUpdateKind::Copy;
    else if (update == "append")
      binding.update_kind = KVCacheUpdateKind::Append;
    else
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown cache update kind");
    out.push_back(std::move(binding));
  }
  return Status::OK();
}

Status Exits(const Json& value, InlinedVector<LoopExitCondition>& out) {
  ORT_RETURN_IF_ERROR(Array(value));
  ORT_RETURN_IF(value.empty(), "SessionLoop requires explicit exit_conditions");
  for (const auto& item : value) {
    std::string kind;
    ORT_RETURN_IF_ERROR(String(item, "kind", kind));
    LoopExitCondition condition{};
    if (kind == "end_of_inputs") {
      ORT_RETURN_IF_ERROR(Object(item, {"kind"}));
      condition.kind = LoopExitKind::EndOfInputs;
    } else if (kind == "max_iterations" || kind == "max_generated_tokens") {
      ORT_RETURN_IF_ERROR(Object(item, {"kind", "limit"}));
      condition.kind = kind == "max_iterations" ? LoopExitKind::MaxIterations : LoopExitKind::MaxGeneratedTokens;
      ORT_RETURN_IF_ERROR(IntField(item, "limit", condition.limit));
      ORT_RETURN_IF(condition.limit < 0, "Exit limits must be nonnegative");
    } else if (kind == "eos") {
      ORT_RETURN_IF_ERROR(Object(item, {"kind", "token_ids"}));
      condition.kind = LoopExitKind::Eos;
      const auto ids = item.find("token_ids");
      ORT_RETURN_IF(ids == item.end(), "EOS requires token_ids");
      ORT_RETURN_IF_ERROR(Array(*ids));
      for (const auto& id : *ids) {
        int64_t token = 0;
        ORT_RETURN_IF_ERROR(Integer(id, token));
        condition.eos_token_ids.push_back(token);
      }
    } else if (kind == "boolean_output") {
      ORT_RETURN_IF_ERROR(Object(item, {"kind", "output"}));
      condition.kind = LoopExitKind::BooleanOutput;
      ORT_RETURN_IF_ERROR(String(item, "output", condition.output_name));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown exit kind");
    }
    out.push_back(std::move(condition));
  }
  return Status::OK();
}

Status Generation(const Json& value, const StepIndices& indices, size_t index, ModelPackageChainDefinition& result) {
  ORT_RETURN_IF_ERROR(Object(value, {"phase", "input_ids", "logits_output", "attention_mask", "position_ids",
                                     "logits_position", "initial_logits"}));
  TokenGenerationOptions generation;
  std::string phase;
  ORT_RETURN_IF_ERROR(String(value, "phase", phase));
  if (phase == "prefill")
    generation.phase = TokenGenerationPhase::Prefill;
  else if (phase == "decode")
    generation.phase = TokenGenerationPhase::Decode;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown generation phase");
  ORT_RETURN_IF_ERROR(String(value, "input_ids", generation.input_ids));
  ORT_RETURN_IF_ERROR(String(value, "logits_output", generation.logits_output));
  ORT_RETURN_IF_ERROR(String(value, "attention_mask", generation.attention_mask, false));
  ORT_RETURN_IF_ERROR(String(value, "position_ids", generation.position_ids, false));
  if (const auto position = value.find("logits_position"); position != value.end()) {
    int64_t row = 0;
    ORT_RETURN_IF_ERROR(Integer(*position, row));
    ORT_RETURN_IF(row < 0, "Logits position must be nonnegative");
    generation.logits_position = row;
  }
  if (const auto initial = value.find("initial_logits"); initial != value.end()) {
    ORT_RETURN_IF(generation.phase != TokenGenerationPhase::Decode, "initial_logits requires decode generation");
    ChainGenerationBootstrap bootstrap;
    ORT_RETURN_IF_ERROR(Source(*initial, indices, index, bootstrap.logits, true));
    const auto source = bootstrap.logits.component_index;
    std::string prompt;
    ORT_RETURN_IF_ERROR(String(*initial, "prompt_input", prompt, false));
    if (prompt.empty() && result.loops[source]) prompt = result.loops[source]->nonempty_input;
    const auto& bindings = result.options.components[source].inputs;
    const auto found = std::find_if(bindings.begin(), bindings.end(),
                                    [&](const auto& binding) { return binding.input_name == prompt; });
    ORT_RETURN_IF(prompt.empty() || found == bindings.end(),
                  "initial_logits requires an explicit prompt input binding");
    bootstrap.prompt = found->source;
    if (const auto position = initial->find("position"); position != initial->end()) {
      int64_t row = 0;
      ORT_RETURN_IF_ERROR(Integer(*position, row));
      ORT_RETURN_IF(row < 0, "Initial logits position must be nonnegative");
      bootstrap.logits_position = row;
    }
    result.options.components[index].generation_bootstrap = std::move(bootstrap);
  }
  result.loops[index]->generation = std::move(generation);
  return Status::OK();
}

Status Step(const Json& value, const StepIndices& indices, const CacheSpecs& specs, size_t index,
            ModelPackageChainDefinition& result) {
  std::string kind;
  ORT_RETURN_IF_ERROR(String(value, "kind", kind));
  auto& component = result.options.components[index];
  const bool loop = kind == "session_loop";
  ORT_RETURN_IF(kind != "session" && !loop, "Unknown execution kind");
  ORT_RETURN_IF_ERROR(loop ? Object(value, {"kind", "inputs", "state_outputs", "loop"})
                           : Object(value, {"kind", "inputs", "state_outputs", "kv_caches", "cache_provider"}));
  if (const auto inputs = value.find("inputs"); inputs != value.end()) {
    ORT_RETURN_IF(!inputs->is_object(), "inputs must be a map");
    for (auto it = inputs->begin(); it != inputs->end(); ++it) {
      ChainInputBinding binding;
      ORT_RETURN_IF_ERROR(Name(Json(it.key()), binding.input_name));
      ORT_RETURN_IF_ERROR(Source(*it, indices, index, binding.source));
      component.inputs.push_back(std::move(binding));
    }
  }
  if (const auto states = value.find("state_outputs"); states != value.end()) {
    ORT_RETURN_IF(!states->is_object(), "state_outputs must be a map");
    for (auto it = states->begin(); it != states->end(); ++it) {
      ChainStateBinding binding;
      ORT_RETURN_IF_ERROR(Name(Json(it.key()), binding.state_name));
      ORT_RETURN_IF_ERROR(Name(*it, binding.output_name));
      component.state_outputs.push_back(std::move(binding));
    }
  }
  if (!loop) {
    if (const auto caches = value.find("kv_caches"); caches != value.end()) {
      ORT_RETURN_IF_ERROR(CacheBindings(*caches, specs, component.kv_caches));
    }
    if (const auto provider = value.find("cache_provider"); provider != value.end()) {
      StatefulProviderSelection selection;
      ORT_RETURN_IF_ERROR(Provider(*provider, selection));
      component.cache_provider = std::move(selection);
    }
    return Status::OK();
  }
  const auto config = value.find("loop");
  ORT_RETURN_IF(config == value.end(), "session_loop requires loop configuration");
  ORT_RETURN_IF_ERROR(
      Object(*config, {"finite_input_names", "nonempty_input", "feedback", "exit_conditions", "generated_token_output",
                       "generation", "sampler", "cache_provider", "kv_caches"}));
  result.loops[index].emplace();
  auto& options = *result.loops[index];
  if (const auto finite = config->find("finite_input_names"); finite != config->end()) {
    ORT_RETURN_IF_ERROR(Array(*finite));
    for (const auto& name : *finite) {
      std::string input;
      ORT_RETURN_IF_ERROR(Name(name, input));
      options.finite_input_names.push_back(std::move(input));
    }
  }
  ORT_RETURN_IF_ERROR(String(*config, "nonempty_input", options.nonempty_input, false));
  ORT_RETURN_IF_ERROR(String(*config, "generated_token_output", options.generated_token_output, false));
  if (const auto feedback = config->find("feedback"); feedback != config->end()) {
    ORT_RETURN_IF_ERROR(Array(*feedback));
    for (const auto& item : *feedback) {
      ORT_RETURN_IF_ERROR(Object(item, {"input", "output"}));
      LoopFeedbackBinding binding;
      ORT_RETURN_IF_ERROR(String(item, "input", binding.input_name));
      ORT_RETURN_IF_ERROR(String(item, "output", binding.output_name));
      options.feedback.push_back(std::move(binding));
    }
  }
  const auto exits = config->find("exit_conditions");
  ORT_RETURN_IF(exits == config->end(), "SessionLoop requires explicit exit_conditions");
  ORT_RETURN_IF_ERROR(Exits(*exits, options.exit_conditions));
  if (const auto caches = config->find("kv_caches"); caches != config->end()) {
    ORT_RETURN_IF_ERROR(CacheBindings(*caches, specs, options.kv_caches));
  }
  if (const auto provider = config->find("cache_provider"); provider != config->end()) {
    ORT_RETURN_IF_ERROR(Provider(*provider, options.cache_provider));
  }
  const auto generation = config->find("generation");
  const auto sampler = config->find("sampler");
  ORT_RETURN_IF((generation != config->end()) != (sampler != config->end()),
                "A generation policy and an explicit sampler must be configured together");
  if (generation != config->end()) {
    ORT_RETURN_IF(!options.generated_token_output.empty(), "Native generation conflicts with generated_token_output");
    ORT_RETURN_IF_ERROR(Generation(*generation, indices, index, result));
    ORT_RETURN_IF_ERROR(Object(*sampler, {"algorithm", "provider"}));
    std::string algorithm;
    ORT_RETURN_IF_ERROR(String(*sampler, "algorithm", algorithm));
    ORT_RETURN_IF(algorithm != "greedy", "Unsupported sampler algorithm");
    if (const auto provider = sampler->find("provider"); provider != sampler->end()) {
      ORT_RETURN_IF_ERROR(Provider(*provider, options.sampler_provider));
    }
  }
  return Status::OK();
}

Status OrtPayload(const Json& value, const Json*& out) {
  const auto info = value.find("executor_info");
  ORT_RETURN_IF(info == value.end() || !info->is_object(), "Missing executor_info");
  const auto ort = info->find("ort");
  ORT_RETURN_IF(ort == info->end() || !ort->is_object(), "Missing inline executor_info.ort");
  out = &*ort;
  return Status::OK();
}

}  // namespace

Status ParseModelPackageChain(std::string_view text, ModelPackageChainDefinition& out) {
  const auto chain = Json::parse(text, nullptr, false);
  ORT_RETURN_IF(!chain.is_object(), "Invalid chain JSON");
  const auto steps = chain.find("steps");
  ORT_RETURN_IF(steps == chain.end() || !steps->is_array() || steps->empty(), "Chain requires ordered steps");
  const Json* ort = nullptr;
  ORT_RETURN_IF_ERROR(OrtPayload(chain, ort));
  ORT_RETURN_IF_ERROR(Object(*ort, {"kv_caches", "outputs"}));
  CacheSpecs specs;
  if (const auto caches = ort->find("kv_caches"); caches != ort->end()) {
    ORT_RETURN_IF_ERROR(Specs(*caches, specs));
  }
  ModelPackageChainDefinition result;
  StepIndices indices;
  for (const auto& step : *steps) {
    SessionChainComponent component;
    std::string name;
    ORT_RETURN_IF_ERROR(String(step, "id", component.name));
    ORT_RETURN_IF_ERROR(String(step, "component", name));
    ORT_RETURN_IF(!indices.emplace(component.name, result.options.components.size()).second, "Duplicate step id");
    result.options.components.push_back(std::move(component));
    result.component_names.push_back(std::move(name));
    result.loops.emplace_back();
  }
  for (size_t i = 0; i < steps->size(); ++i) {
    ORT_RETURN_IF_ERROR(OrtPayload((*steps)[i], ort));
    const auto status = Step(*ort, indices, specs, i, result);
    if (!status.IsOK()) {
      return Status(status.Category(), status.Code(),
                    MakeString("step '", result.options.components[i].name, "': ", status.ErrorMessage()));
    }
  }
  ORT_RETURN_IF_ERROR(OrtPayload(chain, ort));
  const auto outputs = ort->find("outputs");
  ORT_RETURN_IF(outputs == ort->end() || !outputs->is_object() || outputs->empty(), "Chain requires named outputs");
  for (auto it = outputs->begin(); it != outputs->end(); ++it) {
    ChainOutputBinding output;
    ORT_RETURN_IF_ERROR(Name(Json(it.key()), output.name));
    std::string kind;
    ORT_RETURN_IF_ERROR(String(*it, "kind", kind));
    if (kind == "generated_token_ids") {
      ORT_RETURN_IF_ERROR(Object(*it, {"kind", "step"}));
      std::string step;
      ORT_RETURN_IF_ERROR(String(*it, "step", step));
      const auto found = indices.find(step);
      ORT_RETURN_IF(found == indices.end() || !result.loops[found->second] || !result.loops[found->second]->generation,
                    "Token output must name a generation step");
      output.kind = ChainOutputKind::GeneratedTokenIds;
      output.component_index = found->second;
    } else {
      ChainValueSource source;
      ORT_RETURN_IF_ERROR(Source(*it, indices, steps->size(), source));
      ORT_RETURN_IF(source.kind != ChainSourceKind::ComponentOutput, "Chain output must name a model output");
      output.component_index = source.component_index;
      output.output_name = std::move(source.name);
    }
    result.options.outputs.push_back(std::move(output));
  }
  out = std::move(result);
  return Status::OK();
}

}  // namespace onnxruntime
