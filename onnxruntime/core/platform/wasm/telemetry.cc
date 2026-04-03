// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/wasm/telemetry.h"
#include "onnxruntime_config.h"

#include <emscripten.h>

#include <cstdio>
#include <string>

namespace onnxruntime {

namespace {

class JsonBuilder {
  std::string buf_;
  bool first_ = true;

  void Sep() {
    if (!first_) buf_ += ',';
    first_ = false;
  }

  static std::string Escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
      switch (c) {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
            out += buf;
          } else {
            out += c;
          }
      }
    }
    return out;
  }

 public:
  JsonBuilder() { buf_ = '{'; }

  JsonBuilder& Add(const char* key, const std::string& val) {
    Sep();
    buf_ += '"';
    buf_ += key;
    buf_ += "\":\"";
    buf_ += Escape(val);
    buf_ += '"';
    return *this;
  }

  JsonBuilder& Add(const char* key, const char* val) {
    return Add(key, val ? std::string(val) : std::string());
  }

  JsonBuilder& Add(const char* key, int32_t val) {
    Sep();
    buf_ += '"';
    buf_ += key;
    buf_ += "\":";
    buf_ += std::to_string(val);
    return *this;
  }

  JsonBuilder& Add(const char* key, uint32_t val) {
    Sep();
    buf_ += '"';
    buf_ += key;
    buf_ += "\":";
    buf_ += std::to_string(val);
    return *this;
  }

  JsonBuilder& Add(const char* key, int64_t val) {
    Sep();
    buf_ += '"';
    buf_ += key;
    buf_ += "\":";
    buf_ += std::to_string(val);
    return *this;
  }

  JsonBuilder& Add(const char* key, bool val) {
    Sep();
    buf_ += '"';
    buf_ += key;
    buf_ += "\":";
    buf_ += val ? "true" : "false";
    return *this;
  }

  JsonBuilder& AddStringList(const char* key, const std::vector<std::string>& vec) {
    std::string joined;
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) joined += ',';
      joined += vec[i];
    }
    return Add(key, joined);
  }

  JsonBuilder& AddIntMap(const char* key, const std::unordered_map<std::string, int>& map) {
    std::string result;
    bool f = true;
    for (const auto& [k, v] : map) {
      if (!f) result += ',';
      result += k + '=' + std::to_string(v);
      f = false;
    }
    return Add(key, result);
  }

  JsonBuilder& AddStringMap(const char* key, const std::unordered_map<std::string, std::string>& map) {
    std::string result;
    bool f = true;
    for (const auto& [k, v] : map) {
      if (!f) result += ',';
      result += k + '=' + v;
      f = false;
    }
    return Add(key, result);
  }

  JsonBuilder& AddBatchSizeDurations(const char* prefix, const std::unordered_map<int64_t, long long>& map) {
    for (const auto& [batch_size, duration] : map) {
      std::string key = std::string(prefix) + std::to_string(batch_size);
      Add(key.c_str(), static_cast<int64_t>(duration));
    }
    return *this;
  }

  std::string Build() {
    buf_ += '}';
    return buf_;
  }
};

}  // namespace

// EM_JS bridge: calls into JavaScript with the event name and JSON payload.
// The JS side registers Module.__ortTelemetryCallback during WASM initialization.
EM_JS(void, js_emit_telemetry_event, (const char* event_name, const char* json_data), {
  var name = UTF8ToString(event_name);
  var data = UTF8ToString(json_data);
  if (Module["__ortTelemetryCallback"]) {
    try {
      Module["__ortTelemetryCallback"](name, JSON.parse(data));
    } catch (e) {
      // Telemetry failures must never disrupt the application.
    }
  }
});

void WasmTelemetry::EnableTelemetryEvents() const {
  enabled_.store(true);
}

void WasmTelemetry::DisableTelemetryEvents() const {
  enabled_.store(false);
}

void WasmTelemetry::SetLanguageProjection(uint32_t projection) const {
  projection_.store(projection);
}

bool WasmTelemetry::IsEnabled() const {
  return enabled_.load();
}

unsigned char WasmTelemetry::Level() const {
  return 0;
}

uint64_t WasmTelemetry::Keyword() const {
  return 0;
}

void WasmTelemetry::LogProcessInfo() const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("runtimeVersion", ORT_VERSION)
                  .Add("projection", projection_.load())
                  .Build();
  js_emit_telemetry_event("ProcessInfo", json.c_str());
}

void WasmTelemetry::LogSessionCreationStart(uint32_t session_id) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Build();
  js_emit_telemetry_event("SessionCreationStart", json.c_str());
}

void WasmTelemetry::LogEvaluationStop(uint32_t session_id) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Build();
  js_emit_telemetry_event("EvaluationStop", json.c_str());
}

void WasmTelemetry::LogEvaluationStart(uint32_t session_id) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Build();
  js_emit_telemetry_event("EvaluationStart", json.c_str());
}

void WasmTelemetry::LogSessionCreation(
    uint32_t session_id, int64_t ir_version,
    const std::string& model_producer_name,
    const std::string& model_producer_version,
    const std::string& model_domain,
    const std::unordered_map<std::string, int>& domain_to_version_map,
    const std::string& model_file_name,
    const std::string& model_graph_name,
    const std::string& model_weight_type,
    const std::string& model_graph_hash,
    const std::string& model_weight_hash,
    const std::unordered_map<std::string, std::string>& model_metadata,
    const std::string& loadedFrom,
    const std::vector<std::string>& execution_provider_ids,
    bool use_fp16, bool captureState) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("irVersion", ir_version)
                  .Add("modelProducerName", model_producer_name)
                  .Add("modelProducerVersion", model_producer_version)
                  .Add("modelDomain", model_domain)
                  .AddIntMap("domainToVersionMap", domain_to_version_map)
                  .Add("modelFileName", model_file_name)
                  .Add("modelGraphName", model_graph_name)
                  .Add("modelWeightType", model_weight_type)
                  .Add("modelGraphHash", model_graph_hash)
                  .Add("modelWeightHash", model_weight_hash)
                  .AddStringMap("modelMetadata", model_metadata)
                  .Add("loadedFrom", loadedFrom)
                  .AddStringList("executionProviderIds", execution_provider_ids)
                  .Add("useFp16", use_fp16)
                  .Add("captureState", captureState)
                  .Add("projection", projection_.load())
                  .Build();
  js_emit_telemetry_event("SessionCreation", json.c_str());
}

void WasmTelemetry::LogCompileModelStart(
    uint32_t session_id,
    const std::string& input_source,
    const std::string& output_target,
    uint32_t flags,
    int graph_optimization_level,
    bool embed_ep_context,
    bool has_external_initializers_file,
    const std::vector<std::string>& execution_provider_ids) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("inputSource", input_source)
                  .Add("outputTarget", output_target)
                  .Add("flags", flags)
                  .Add("graphOptimizationLevel", static_cast<int32_t>(graph_optimization_level))
                  .Add("embedEpContext", embed_ep_context)
                  .Add("hasExternalInitializersFile", has_external_initializers_file)
                  .AddStringList("executionProviderIds", execution_provider_ids)
                  .Build();
  js_emit_telemetry_event("CompileModelStart", json.c_str());
}

void WasmTelemetry::LogCompileModelComplete(
    uint32_t session_id,
    bool success,
    uint32_t error_code,
    uint32_t error_category,
    const std::string& error_message) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("success", success)
                  .Add("errorCode", error_code)
                  .Add("errorCategory", error_category)
                  .Add("errorMessage", error_message)
                  .Build();
  js_emit_telemetry_event("CompileModelComplete", json.c_str());
}

void WasmTelemetry::LogRuntimeError(
    uint32_t session_id, const common::Status& status,
    const char* file, const char* function, uint32_t line) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("errorCode", static_cast<int32_t>(status.Code()))
                  .Add("errorCategory", static_cast<int32_t>(status.Category()))
                  .Add("errorMessage", status.ErrorMessage())
                  .Add("file", file)
                  .Add("function", function)
                  .Add("line", line)
                  .Build();
  js_emit_telemetry_event("RuntimeError", json.c_str());
}

void WasmTelemetry::LogRuntimePerf(
    uint32_t session_id, uint32_t total_runs_since_last,
    int64_t total_run_duration_since_last,
    const std::unordered_map<int64_t, long long>& duration_per_batch_size) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("totalRunsSinceLast", total_runs_since_last)
                  .Add("totalRunDurationSinceLast", total_run_duration_since_last)
                  .AddBatchSizeDurations("batchSize_", duration_per_batch_size)
                  .Build();
  js_emit_telemetry_event("RuntimePerf", json.c_str());
}

void WasmTelemetry::LogExecutionProviderEvent(LUID* /*adapterLuid*/) const {
  // LUID is Windows-specific; not applicable for WebAssembly.
}

void WasmTelemetry::LogDriverInfoEvent(
    const std::string_view /*device_class*/,
    const std::wstring_view& /*driver_names*/,
    const std::wstring_view& /*driver_versions*/) const {
  // Driver info is Windows-specific; not applicable for WebAssembly.
}

void WasmTelemetry::LogAutoEpSelection(
    uint32_t session_id, const std::string& selection_policy,
    const std::vector<std::string>& requested_execution_provider_ids,
    const std::vector<std::string>& available_execution_provider_ids) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("selectionPolicy", selection_policy)
                  .AddStringList("requestedExecutionProviderIds", requested_execution_provider_ids)
                  .AddStringList("availableExecutionProviderIds", available_execution_provider_ids)
                  .Build();
  js_emit_telemetry_event("EpAutoSelection", json.c_str());
}

void WasmTelemetry::LogProviderOptions(
    const std::string& provider_id,
    const std::string& provider_options_string,
    bool captureState) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("providerId", provider_id)
                  .Add("providerOptions", provider_options_string)
                  .Add("captureState", captureState)
                  .Build();
  js_emit_telemetry_event("ProviderOptions", json.c_str());
}

void WasmTelemetry::LogModelLoadStart(uint32_t session_id) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Build();
  js_emit_telemetry_event("ModelLoadStart", json.c_str());
}

void WasmTelemetry::LogModelLoadEnd(uint32_t session_id, const common::Status& status) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("success", status.IsOK())
                  .Add("errorCode", static_cast<int32_t>(status.Code()))
                  .Add("errorMessage", status.ErrorMessage())
                  .Build();
  js_emit_telemetry_event("ModelLoadEnd", json.c_str());
}

void WasmTelemetry::LogSessionCreationEnd(uint32_t session_id, const common::Status& status) const {
  if (!enabled_.load()) return;

  auto json = JsonBuilder()
                  .Add("sessionId", session_id)
                  .Add("success", status.IsOK())
                  .Add("errorCode", static_cast<int32_t>(status.Code()))
                  .Add("errorMessage", status.ErrorMessage())
                  .Build();
  js_emit_telemetry_event("SessionCreationEnd", json.c_str());
}

void WasmTelemetry::LogRegisterEpLibraryWithLibPath(
    const std::string& /*registration_name*/,
    const std::string& /*lib_path*/) const {
  // EP library loading via shared libraries is not applicable for WebAssembly.
}

void WasmTelemetry::LogRegisterEpLibraryStart(const std::string& /*registration_name*/) const {
  // EP library loading via shared libraries is not applicable for WebAssembly.
}

void WasmTelemetry::LogRegisterEpLibraryEnd(
    const std::string& /*registration_name*/,
    const common::Status& /*status*/) const {
  // EP library loading via shared libraries is not applicable for WebAssembly.
}

}  // namespace onnxruntime
