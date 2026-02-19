// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _WIN32  // Only for non-Windows platforms

#include "core/platform/posix/telemetry.h"

// 1DS SDK includes
#include <LogManager.hpp>
#include <ILogger.hpp>
#include <ISemanticContext.hpp>

#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/utsname.h>
#include <sys/resource.h>

#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "onnxruntime_config.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

using namespace Microsoft::Applications::Events;

namespace onnxruntime {

// Static member initialization
std::atomic<uint32_t> PosixTelemetry::global_register_count_{0};
std::mutex PosixTelemetry::global_mutex_;

// Tenant token for 1DS telemetry ingestion
constexpr const char* TENANT_TOKEN = "5ad963bd4b3a4118a481401cc0211875-da8e8657-47d4-4ed7-ab39-7886e136f53b-6988";

// Event priority mapping (1DS priorities)
enum class EventPriority {
  NORMAL = EventLatency_Normal,     // Most events
  HIGH = EventLatency_RealTime,     // RuntimeError
  CRITICAL = EventLatency_RealTime  // ProcessInfo, SessionCreation
};

// Transmit profiles
constexpr const char* PROFILE_REAL_TIME = "REAL_TIME";
constexpr const char* PROFILE_NEAR_REAL_TIME = "NEAR_REAL_TIME";
constexpr const char* PROFILE_BEST_EFFORT = "BEST_EFFORT";

// Helper class to build events with common properties
class EventBuilder {
 private:
  EventProperties props_;

 public:
  explicit EventBuilder(const char* event_name, EventPriority priority)
      : props_(event_name) {
    // Set latency/priority
    props_.SetLatency(static_cast<EventLatency>(priority));

    // Set schema version for compatibility with Windows
    props_.SetProperty("schemaVersion", static_cast<int64_t>(0));

    // Privacy flags - no PII collection
    props_.SetPIIKind(PiiKind_None);
  }

  EventBuilder& AddString(const char* key, const std::string& value) {
    if (!value.empty()) {
      props_.SetProperty(key, value);
    }
    return *this;
  }

  EventBuilder& AddInt32(const char* key, int32_t value) {
    props_.SetProperty(key, static_cast<int64_t>(value));
    return *this;
  }

  EventBuilder& AddInt64(const char* key, int64_t value) {
    props_.SetProperty(key, value);
    return *this;
  }

  EventBuilder& AddBool(const char* key, bool value) {
    props_.SetProperty(key, value);
    return *this;
  }

  EventBuilder& AddUInt32(const char* key, uint32_t value) {
    props_.SetProperty(key, static_cast<int64_t>(value));
    return *this;
  }

  EventBuilder& AddDouble(const char* key, double value) {
    props_.SetProperty(key, value);
    return *this;
  }

  // Helper for vector to comma-separated string
  EventBuilder& AddStringList(const char* key, const std::vector<std::string>& vec) {
    if (!vec.empty()) {
      std::string result;
      for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) result += ',';
        result += vec[i];
      }
      props_.SetProperty(key, result);
    }
    return *this;
  }

  // Helper for map to key=value,key=value format
  EventBuilder& AddIntMap(const char* key, const std::unordered_map<std::string, int>& map) {
    if (!map.empty()) {
      std::string result;
      bool first = true;
      for (const auto& [k, v] : map) {
        if (!first) result += ',';
        result += k + '=' + std::to_string(v);
        first = false;
      }
      props_.SetProperty(key, result);
    }
    return *this;
  }

  // Helper for string map
  EventBuilder& AddStringMap(const char* key, const std::unordered_map<std::string, std::string>& map) {
    if (!map.empty()) {
      std::string result;
      bool first = true;
      for (const auto& [k, v] : map) {
        if (!first) result += ',';
        result += k + '=' + v;
        first = false;
      }
      props_.SetProperty(key, result);
    }
    return *this;
  }

  // Helper for batch size duration map
  EventBuilder& AddBatchSizeDurations(const std::unordered_map<int64_t, long long>& durations) {
    for (const auto& [batch_size, duration] : durations) {
      std::string key = "batchSize_" + std::to_string(batch_size);
      props_.SetProperty(key, duration);
    }
    return *this;
  }

  // Add common platform/device context
  EventBuilder& AddCommonContext(const PosixTelemetry* telemetry) {
    props_.SetProperty("platform", telemetry->GetPlatformInfo());
    props_.SetProperty("device", telemetry->GetDeviceInfo());
    props_.SetProperty("projection", static_cast<int64_t>(telemetry->projection_.load()));
    return *this;
  }

  EventProperties Build() { return std::move(props_); }
};

PosixTelemetry::PosixTelemetry() {
  std::lock_guard<std::mutex> lock(global_mutex_);

  if (global_register_count_ == 0) {
    try {
      Initialize();
      global_register_count_++;
    } catch (const std::exception& ex) {
      // Log error but don't fail construction
      // Telemetry failures should not break application functionality
      LOGS_DEFAULT(WARNING) << "Failed to initialize telemetry: " << ex.what();
    }
  }
}

PosixTelemetry::~PosixTelemetry() {
  std::lock_guard<std::mutex> lock(global_mutex_);

  if (global_register_count_ > 0) {
    global_register_count_--;
    if (global_register_count_ == 0) {
      try {
        Shutdown();
      } catch (const std::exception& ex) {
        // Log error but don't throw from destructor
        LOGS_DEFAULT(WARNING) << "Error during telemetry shutdown: " << ex.what();
      }
    }
  }
}

// Safe async event logging with error handling
void PosixTelemetry::LogEventAsync(microsoft::applications::events::EventProperties&& props) const {
  if (!enabled_ || !logger_) {
    return;
  }

  try {
    // Use async LogEvent for non-blocking telemetry
    logger_->LogEvent(std::move(props));
  } catch (const std::exception& ex) {
    // Log telemetry failures to ORT logging system
    LOGS_DEFAULT(WARNING) << "[Telemetry] Failed to log event: " << ex.what();
  }
}

void PosixTelemetry::Initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Configure 1DS SDK for optimal async performance
  LogConfiguration config;
  config[CFG_STR_COLLECTOR_URL] = "https://mobile.events.data.microsoft.com/OneCollector/1.0";
  config[CFG_INT_TRACE_LEVEL_MASK] = 0;                      // Disable SDK internal logging
  config[CFG_INT_SDK_MODE] = SdkModeTypes::SdkModeTypes_CS;  // Common Schema 4.0 mode
  config[CFG_INT_MAX_TEARDOWN_TIME] = 10;                    // 10 seconds max for shutdown

  // Configure cache for offline scenarios
  config[CFG_STR_CACHE_FILE_PATH] = "/tmp/onnxruntime_telemetry_cache";

  // Configure RAM queue for async batching
  config[CFG_INT_RAM_QUEUE_SIZE] = 512 * 1024;  // 512KB RAM queue
  config[CFG_INT_RAM_QUEUE_BUFFERS] = 3;        // Triple buffering for smooth async operation

  // Sampling configuration (percentage: 100 = 100%, 10 = 10%)
  // Sample 100% of critical events, 10% of routine events for performance
  config[CFG_STR_SAMPLING_PERCENTAGE] =
      "ProcessInfo=100,SessionCreation=100,SessionCreationStart=100,"
      "RuntimeError=100,EvaluationStart=10,EvaluationStop=10,"
      "RuntimePerf=10,CompileModelStart=50,CompileModelComplete=50,"
      "EpAutoSelection=50,ProviderOptions=10";

  // Create logger instance
  logger_ = LogManager::Initialize(TENANT_TOKEN, config);

  if (logger_) {
    // Set privacy level - no PII collection
    logger_->SetContext("PrivacyLevel", "o:0");

    // Set platform information as context
    logger_->SetContext("Platform", GetPlatformInfo());
    logger_->SetContext("Device", GetDeviceInfo());

    // Set application information
    logger_->SetContext("AppName", "ONNXRuntime");
    logger_->SetContext("AppVersion", ORT_VERSION);

    enabled_ = true;
  }
}

void PosixTelemetry::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (logger_) {
    // According to cpp_client_telemetry use-after-free docs:
    // 1. Stop using ILogger before calling FlushAndTeardown
    // 2. Reset shared_ptr to release reference before teardown
    // 3. Call FlushAndTeardown only once when count reaches zero

    // Disable logging first to prevent new events
    enabled_ = false;

    // Release our reference to the logger
    logger_.reset();

    // Now safely call FlushAndTeardown
    // This will block until all pending events are sent or timeout
    LogManager::FlushAndTeardown();
  }
}

std::string PosixTelemetry::GetPlatformInfo() const {
  struct utsname system_info;
  if (uname(&system_info) == 0) {
    std::ostringstream oss;
    oss << system_info.sysname << " " << system_info.release;
    return oss.str();
  }
  return "Unknown";
}

std::string PosixTelemetry::GetDeviceInfo() const {
#ifdef __APPLE__
#if TARGET_OS_IOS
  return "iOS";
#elif TARGET_OS_MAC
  return "macOS";
#endif
#elif defined(__ANDROID__)
  return "Android";
#elif defined(__linux__)
  return "Linux";
#else
  return "Unknown";
#endif
}

void PosixTelemetry::EnableTelemetryEvents() const {
  enabled_ = true;
}

void PosixTelemetry::DisableTelemetryEvents() const {
  enabled_ = false;
}

void PosixTelemetry::SetLanguageProjection(uint32_t projection) const {
  projection_ = projection;
}

bool PosixTelemetry::IsEnabled() const {
  return enabled_;
}

unsigned char PosixTelemetry::Level() const {
  return level_;
}

uint64_t PosixTelemetry::Keyword() const {
  return keyword_;
}

void PosixTelemetry::LogProcessInfo() const {
  if (!enabled_ || !logger_) {
    return;
  }

  // Log process info only once
  if (process_info_logged_.exchange(true)) {
    return;
  }

  auto event = EventBuilder("ProcessInfo", EventPriority::CRITICAL)
                   .AddCommonContext(this)
                   .AddString("runtimeVersion", ORT_VERSION)
                   .AddInt32("processId", static_cast<int32_t>(getpid()))
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogSessionCreationStart(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("SessionCreationStart", EventPriority::CRITICAL)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogEvaluationStop(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("EvaluationStop", EventPriority::NORMAL)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogEvaluationStart(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("EvaluationStart", EventPriority::NORMAL)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogSessionCreation(
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
  if (!enabled_ || !logger_) {
    return;
  }

  const char* event_name = captureState ? "SessionCreation_CaptureState" : "SessionCreation";

  auto builder = EventBuilder(event_name, EventPriority::CRITICAL)
                     .AddCommonContext(this)
                     .AddUInt32("sessionId", session_id)
                     .AddInt64("irVersion", ir_version)
                     .AddString("modelProducerName", model_producer_name)
                     .AddString("modelProducerVersion", model_producer_version)
                     .AddString("modelDomain", model_domain)
                     .AddIntMap("domainToVersionMap", domain_to_version_map)
                     .AddString("modelFileName", model_file_name)
                     .AddString("modelGraphName", model_graph_name)
                     .AddString("modelWeightType", model_weight_type)
                     .AddString("modelGraphHash", model_graph_hash)
                     .AddString("modelWeightHash", model_weight_hash)
                     .AddStringMap("modelMetadata", model_metadata)
                     .AddString("loadedFrom", loadedFrom)
                     .AddStringList("executionProviderIds", execution_provider_ids)
                     .AddBool("useFp16", use_fp16);

  LogEventAsync(builder.Build());
}

void PosixTelemetry::LogCompileModelStart(
    uint32_t session_id,
    const std::string& input_source,
    const std::string& output_target,
    uint32_t flags,
    int graph_optimization_level,
    bool embed_ep_context,
    bool has_external_initializers_file,
    const std::vector<std::string>& execution_provider_ids) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("CompileModelStart", EventPriority::NORMAL)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .AddString("inputSource", input_source)
                   .AddString("outputTarget", output_target)
                   .AddUInt32("flags", flags)
                   .AddInt32("graphOptimizationLevel", graph_optimization_level)
                   .AddBool("embedEpContext", embed_ep_context)
                   .AddBool("hasExternalInitializersFile", has_external_initializers_file)
                   .AddStringList("executionProviderIds", execution_provider_ids)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogCompileModelComplete(
    uint32_t session_id,
    bool success,
    uint32_t error_code,
    uint32_t error_category,
    const std::string& error_message) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("CompileModelComplete", EventPriority::NORMAL)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .AddBool("success", success)
                   .AddUInt32("errorCode", error_code)
                   .AddUInt32("errorCategory", error_category)
                   .AddString("errorMessage", error_message)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogRuntimeError(
    uint32_t session_id, const common::Status& status,
    const char* file, const char* function, uint32_t line) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("RuntimeError", EventPriority::HIGH)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .AddInt32("errorCode", static_cast<int32_t>(status.Code()))
                   .AddInt32("errorCategory", static_cast<int32_t>(status.Category()))
                   .AddString("errorMessage", status.ErrorMessage())
                   .AddString("file", file ? file : "")
                   .AddString("function", function ? function : "")
                   .AddUInt32("line", line)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogRuntimePerf(
    uint32_t session_id, uint32_t total_runs_since_last,
    int64_t total_run_duration_since_last,
    std::unordered_map<int64_t, long long> duration_per_batch_size) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("RuntimePerf", EventPriority::NORMAL)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .AddUInt32("totalRunsSinceLast", total_runs_since_last)
                   .AddInt64("totalRunDurationSinceLast", total_run_duration_since_last)
                   .AddBatchSizeDurations(duration_per_batch_size)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogExecutionProviderEvent(LUID* adapterLuid) const {
  // Not applicable for non-Windows platforms (LUID is Windows-specific)
  (void)adapterLuid;
}

void PosixTelemetry::LogDriverInfoEvent(
    const std::string_view device_class,
    const std::wstring_view& driver_names,
    const std::wstring_view& driver_versions) const {
  // Not applicable for non-Windows platforms
  (void)device_class;
  (void)driver_names;
  (void)driver_versions;
}

void PosixTelemetry::LogAutoEpSelection(
    uint32_t session_id, const std::string& selection_policy,
    const std::vector<std::string>& requested_execution_provider_ids,
    const std::vector<std::string>& available_execution_provider_ids) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("EpAutoSelection", EventPriority::NORMAL)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .AddString("selectionPolicy", selection_policy)
                   .AddStringList("requestedExecutionProviderIds", requested_execution_provider_ids)
                   .AddStringList("availableExecutionProviderIds", available_execution_provider_ids)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogProviderOptions(
    const std::string& provider_id,
    const std::string& provider_options_string,
    bool captureState) const {
  if (!enabled_ || !logger_) {
    return;
  }

  const char* event_name = captureState ? "ProviderOptions_CaptureState" : "ProviderOptions";

  auto event = EventBuilder(event_name, EventPriority::NORMAL)
                   .AddCommonContext(this)
                   .AddString("providerId", provider_id)
                   .AddString("providerOptions", provider_options_string)
                   .Build();

  LogEventAsync(std::move(event));
}

// Posix-specific: Log system resource metrics
void PosixTelemetry::LogPosixSystemMetrics(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    // Note: ru_maxrss is in KB on Linux, bytes on macOS
#ifdef __APPLE__
    int64_t max_rss_kb = usage.ru_maxrss / 1024;
#else
    int64_t max_rss_kb = usage.ru_maxrss;
#endif

    auto event = EventBuilder("PosixSystemMetrics", EventPriority::NORMAL)
                     .AddCommonContext(this)
                     .AddUInt32("sessionId", session_id)
                     .AddInt64("maxRssKb", max_rss_kb)
                     .AddInt64("userCpuTimeSec", usage.ru_utime.tv_sec)
                     .AddInt64("userCpuTimeUsec", usage.ru_utime.tv_usec)
                     .AddInt64("systemCpuTimeSec", usage.ru_stime.tv_sec)
                     .AddInt64("systemCpuTimeUsec", usage.ru_stime.tv_usec)
                     .AddInt64("minorPageFaults", usage.ru_minflt)
                     .AddInt64("majorPageFaults", usage.ru_majflt)
                     .AddInt64("voluntaryContextSwitches", usage.ru_nvcsw)
                     .AddInt64("involuntaryContextSwitches", usage.ru_nivcsw)
                     .Build();

    LogEventAsync(std::move(event));
  }
}

}  // namespace onnxruntime

#endif  // !_WIN32
