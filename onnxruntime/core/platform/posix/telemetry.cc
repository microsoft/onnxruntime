// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/telemetry.h"
#include "core/platform/posix/device_id.h"

// 1DS SDK
#include <LogManagerProvider.hpp>
#include <ILogConfiguration.hpp>
#include <api/ContextFieldsProvider.hpp>

#include <unistd.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <TargetConditionals.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <fstream>
#endif

#include <thread>
#include <sstream>
#include <iomanip>

#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "onnxruntime_config.h"

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

// Helper class to build events with common properties
class EventBuilder {
 private:
  EventProperties props_;

 public:
  explicit EventBuilder(std::string event_name, EventPriority priority,
                        uint64_t privacy_tags = PDT_ProductAndServicePerformance)
      : props_(std::move(event_name)) {
    // Set latency/priority
    props_.SetLatency(static_cast<EventLatency>(priority));

    // Set schema version for compatibility with Windows
    props_.SetProperty("schemaVersion", static_cast<int64_t>(0));

    // All ORT telemetry is required system metadata (no PII)
    props_.SetLevel(DIAG_LEVEL_REQUIRED);

    // Privacy data tags for GDPR compliance classification
    props_.SetProperty(COMMONFIELDS_EVENT_PRIVTAGS, static_cast<int64_t>(privacy_tags));
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
      props_.SetProperty(key, static_cast<int64_t>(duration));
    }
    return *this;
  }

  // Add common platform/device context
  EventBuilder& AddCommonContext(const PosixTelemetry* telemetry) {
    props_.SetProperty("projection", static_cast<int64_t>(telemetry->projection_.load()));
    return *this;
  }

  EventProperties Build() { return std::move(props_); }
};

// Hash a device ID string using std::hash and format as fixed-width hex.
// Ensures raw device identifiers are never sent over the wire.
static std::string HashDeviceId(const std::string& id) {
  size_t hash = std::hash<std::string>{}(id);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(sizeof(size_t) * 2) << hash;
  return oss.str();
}

PosixTelemetry::PosixTelemetry() {
  std::lock_guard<std::mutex> lock(global_mutex_);

  // Always increment so destructor pairing is symmetric
  global_register_count_++;

  if (global_register_count_ == 1) {
    try {
      Initialize();
    } catch (const std::exception& ex) {
      // Log error but don't fail construction
      // Telemetry failures should not break application functionality
      LOGS_DEFAULT(WARNING) << "Failed to initialize telemetry: " << ex.what();
    }
  }
}

PosixTelemetry::~PosixTelemetry() {
  std::lock_guard<std::mutex> lock(global_mutex_);

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

void PosixTelemetry::LogEventAsync(Microsoft::Applications::Events::EventProperties&& props) const {
  try {
    logger_->LogEvent(std::move(props));
  } catch (const std::exception& ex) {
    LOGS_DEFAULT(WARNING) << "[Telemetry] Failed to log event: " << ex.what();
  }
}

void PosixTelemetry::Initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  // NOTE: On Android, the Java layer must be initialized before calling this:
  //   System.loadLibrary("maesdk");
  //   new HttpClient(getApplicationContext());
  //   OfflineRoom.connectContext(getApplicationContext());  // if using Room DB
  // See cpp_client_telemetry/docs/cpp-start-android.md for details.

  // Create SDK configuration — stored as member because LogManagerImpl holds a reference
  // and the configuration must remain valid for the lifetime of the log manager.
  config_ = std::make_unique<ILogConfiguration>();
  auto& config = *config_;

  config[CFG_STR_COLLECTOR_URL] = "https://mobile.events.data.microsoft.com/OneCollector/1.0";
  config[CFG_INT_TRACE_LEVEL_MASK] = 0;                      // Disable SDK internal logging
  config[CFG_INT_SDK_MODE] = SdkModeTypes::SdkModeTypes_CS;  // Common Schema 4.0 mode
  config[CFG_INT_MAX_TEARDOWN_TIME] = 10;                    // 10 seconds max for shutdown

  // Configure cache for offline scenarios — use same directory as device ID storage
  {
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IOS)
    constexpr bool is_mobile = true;
#else
    constexpr bool is_mobile = false;
#endif
    std::string cache_dir = DeviceId::GetStorageDirectory(is_mobile);
    if (!cache_dir.empty()) {
      std::string cache_path = cache_dir + "/telemetry_cache.db";
      config[CFG_STR_CACHE_FILE_PATH] = cache_path;
    }
  }

  // Configure RAM queue for async batching
  config[CFG_INT_RAM_QUEUE_SIZE] = 512 * 1024;  // 512KB RAM queue

  // Create log manager via LogManagerProvider (recommended for production use,
  // per LogManager_Creation_and_Lifecycle_Management.md).
  status_t status;
  log_manager_ = LogManagerProvider::CreateLogManager(*config_, status);
  if (status != STATUS_SUCCESS || !log_manager_) {
    LOGS_DEFAULT(WARNING) << "Failed to create telemetry LogManager, status: " << status;
    config_.reset();
    return;
  }

  // Get logger for our tenant
  logger_ = log_manager_->GetLogger(TENANT_TOKEN);
  if (!logger_) {
    LOGS_DEFAULT(WARNING) << "Failed to get telemetry logger";
    LogManagerProvider::Release(*config_);
    log_manager_ = nullptr;
    config_.reset();
    return;
  }

  // Use BEST_EFFORT transmit profile to minimize battery and network impact.
  // Events are batched and uploaded at a lower cadence.
  log_manager_->SetTransmitProfile(TransmitProfile_BestEffort);

  // Override device ID with hashed version for privacy.
  // The "c:" prefix tells the backend it's a caller-supplied identifier.
  auto& ctx = log_manager_->GetSemanticContext();
  std::string raw_device_id;
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IOS)
  // Mobile: read SDK's auto-generated platform device ID (e.g., identifierForVendor
  // on iOS, ANDROID_ID on Android) and hash it before sending.
  auto* provider = static_cast<ContextFieldsProvider*>(&ctx);
  auto& fields = provider->GetCommonFields();
  auto it = fields.find(COMMONFIELDS_DEVICE_ID);
  if (it != fields.end()) {
    raw_device_id = it->second.to_string();
  }
#else
  // Desktop: use our custom persistent UUID.
  raw_device_id = DeviceId::Instance().GetValue();
#endif
  if (!raw_device_id.empty()) {
    ctx.SetDeviceId("c:" + HashDeviceId(raw_device_id));
  }

  // Set application information as logger context (attached to all events)
  logger_->SetContext("AppName", "ONNXRuntime");
  logger_->SetContext("AppVersion", ORT_VERSION);
  logger_->SetContext("Platform", GetPlatformInfo());

  enabled_ = true;
}

void PosixTelemetry::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Disable logging first to prevent new events during shutdown
  enabled_ = false;
  logger_ = nullptr;  // Owned by log_manager_, will be destroyed with it

  if (log_manager_ && config_) {
    // Per SDK use-after-free docs (use-after-free.md):
    // Flush() must be called before FlushAndTeardown() to ensure all pending
    // events are persisted to offline storage. FlushAndTeardown() internally
    // calls PauseActivity() + WaitPause() to quiesce the SDK.
    log_manager_->Flush();
    log_manager_->FlushAndTeardown();

    // Release the log manager instance via LogManagerProvider
    LogManagerProvider::Release(*config_);
    log_manager_ = nullptr;
    config_.reset();
  }
}

std::string PosixTelemetry::GetPlatformInfo() const {
#if defined(__APPLE__)
#if TARGET_OS_IOS
  return "iOS";
#elif TARGET_OS_MAC
  return "macOS";
#else
  return "Apple";
#endif
#elif defined(__ANDROID__)
  return "Android";
#elif defined(__linux__)
  return "Linux";
#else
  return "Unknown";
#endif
}

// ---------------------------------------------------------------------------
// Process / system info helpers for LogProcessInfo
// ---------------------------------------------------------------------------

// Get detailed OS version string (e.g., "macOS 15.2", "Ubuntu 22.04 LTS")
std::string PosixTelemetry::GetOsDescription() const {
#if defined(__APPLE__)
  char version[64] = {};
  size_t len = sizeof(version);
  if (sysctlbyname("kern.osproductversion", version, &len, nullptr, 0) == 0) {
#if TARGET_OS_IOS
    return std::string("iOS ") + version;
#else
    return std::string("macOS ") + version;
#endif
  }
  return GetPlatformInfo();

#elif defined(__ANDROID__)
  // Read Android system properties via /system/build.prop
  std::string release, sdk;
  std::ifstream prop("/system/build.prop");
  if (prop.is_open()) {
    std::string line;
    while (std::getline(prop, line)) {
      if (line.rfind("ro.build.version.release=", 0) == 0)
        release = line.substr(25);
      else if (line.rfind("ro.build.version.sdk=", 0) == 0)
        sdk = line.substr(21);
    }
  }
  if (!release.empty()) {
    std::string result = "Android " + release;
    if (!sdk.empty()) result += " (API " + sdk + ")";
    return result;
  }
  return "Android";

#elif defined(__linux__)
  // Parse /etc/os-release for PRETTY_NAME (e.g., "Ubuntu 22.04.3 LTS")
  std::ifstream os_release("/etc/os-release");
  if (os_release.is_open()) {
    std::string line;
    while (std::getline(os_release, line)) {
      if (line.rfind("PRETTY_NAME=", 0) == 0) {
        std::string value = line.substr(12);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.size() - 2);
        }
        return value;
      }
    }
  }
  return "Linux";

#else
  return "Unknown";
#endif
}

// Get the name of the current process
std::string PosixTelemetry::GetProcessName() const {
#if defined(__APPLE__) || defined(__FreeBSD__)
  const char* name = getprogname();
  return name ? name : "";

#elif defined(__linux__) || defined(__ANDROID__)
  // /proc/self/comm contains the process name (up to 15 chars)
  std::ifstream comm("/proc/self/comm");
  if (comm.is_open()) {
    std::string name;
    std::getline(comm, name);
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r'))
      name.pop_back();
    return name;
  }
  return "";

#else
  return "";
#endif
}

// Get the CPU architecture the binary was compiled for
std::string PosixTelemetry::GetArchitecture() {
#if defined(__x86_64__)
  return "x86_64";
#elif defined(__i386__)
  return "x86";
#elif defined(__aarch64__)
  return "arm64";
#elif defined(__arm__)
  return "arm";
#elif defined(__riscv)
  return "riscv";
#elif defined(__wasm__)
  return "wasm";
#else
  return "unknown";
#endif
}

// Get total physical memory in MB
int64_t PosixTelemetry::GetTotalMemoryMB() {
#if defined(__APPLE__)
  int64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    return mem / (1024 * 1024);
  }
  return -1;

#elif defined(__linux__) || defined(__ANDROID__)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<int64_t>(pages) * page_size / (1024 * 1024);
  }
  return -1;

#else
  return -1;
#endif
}

// Get system locale (e.g., "en-US", "ja-JP")
std::string PosixTelemetry::GetLocale() {
  const char* lang = std::getenv("LANG");
  if (lang && lang[0]) {
    std::string loc(lang);
    // Strip encoding suffix (e.g., "en_US.UTF-8" → "en_US")
    auto dot = loc.find('.');
    if (dot != std::string::npos) loc = loc.substr(0, dot);
    // Normalize separator: "en_US" → "en-US"
    for (auto& c : loc) {
      if (c == '_') c = '-';
    }
    return loc;
  }
  return "";
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
  // LogProcessInfo only collects system metadata and always fires if we have a valid logger.
  if (!logger_) {
    return;
  }

  // Log process info only once
  if (process_info_logged_.exchange(true)) {
    return;
  }

  auto builder = EventBuilder("ProcessInfo", EventPriority::CRITICAL,
                              PDT_DeviceConnectivityAndConfiguration | PDT_SoftwareSetupAndInventory)
                     .AddCommonContext(this)
                     .AddString("runtimeVersion", ORT_VERSION)
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IOS)
                     .AddString("DeviceInfo.Status", "Mobile")
#else
                     .AddString("DeviceInfo.Status", DeviceId::Instance().GetStatusString())
#endif
                     .AddString("osDescription", GetOsDescription())
                     .AddString("processName", GetProcessName())
                     .AddString("architecture", GetArchitecture())
                     .AddInt32("cpuCount", static_cast<int32_t>(std::thread::hardware_concurrency()))
                     .AddInt64("totalMemoryMB", GetTotalMemoryMB())
                     .AddString("locale", GetLocale());

  LogEventAsync(builder.Build());
}

void PosixTelemetry::LogSessionCreationStart(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("SessionCreationStart", EventPriority::CRITICAL,
                            PDT_SoftwareSetupAndInventory | PDT_ProductAndServicePerformance)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogEvaluationStop(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("EvaluationStop", EventPriority::NORMAL,
                            PDT_ProductAndServicePerformance)
                   .AddCommonContext(this)
                   .AddUInt32("sessionId", session_id)
                   .Build();

  LogEventAsync(std::move(event));

  // Capture system metrics after each inference run to observe impact
  LogSystemMetrics(session_id);
}

void PosixTelemetry::LogEvaluationStart(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  auto event = EventBuilder("EvaluationStart", EventPriority::NORMAL,
                            PDT_ProductAndServicePerformance)
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

  // captureState is currently only triggered on Windows via ETW's EVENT_CONTROL_CODE_CAPTURE_STATE callback
  // (LogAllSessions). Kept here for future compatibility if a similar mechanism is added for POSIX.
  std::string event_name = captureState ? "SessionCreation_CaptureState" : "SessionCreation";

  auto builder = EventBuilder(std::move(event_name), EventPriority::CRITICAL,
                              PDT_SoftwareSetupAndInventory | PDT_ProductAndServicePerformance)
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

  auto event = EventBuilder("CompileModelStart", EventPriority::NORMAL,
                            PDT_SoftwareSetupAndInventory | PDT_ProductAndServicePerformance)
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

  auto event = EventBuilder("CompileModelComplete", EventPriority::NORMAL,
                            PDT_SoftwareSetupAndInventory | PDT_ProductAndServicePerformance)
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

  auto event = EventBuilder("RuntimeError", EventPriority::HIGH,
                            PDT_ProductAndServicePerformance)
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

  auto event = EventBuilder("RuntimePerf", EventPriority::NORMAL,
                            PDT_ProductAndServicePerformance)
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

  auto event = EventBuilder("EpAutoSelection", EventPriority::NORMAL,
                            PDT_SoftwareSetupAndInventory)
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

  std::string event_name = captureState ? "ProviderOptions_CaptureState" : "ProviderOptions";

  auto event = EventBuilder(std::move(event_name), EventPriority::NORMAL,
                            PDT_SoftwareSetupAndInventory)
                   .AddCommonContext(this)
                   .AddString("providerId", provider_id)
                   .AddString("providerOptions", provider_options_string)
                   .Build();

  LogEventAsync(std::move(event));
}

void PosixTelemetry::LogSystemMetrics(uint32_t session_id) const {
  if (!enabled_ || !logger_) {
    return;
  }

  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    // ru_maxrss is in KB on Linux, bytes on macOS
#ifdef __APPLE__
    int64_t max_rss_kb = usage.ru_maxrss / 1024;
#else
    int64_t max_rss_kb = usage.ru_maxrss;
#endif

    auto event = EventBuilder("SystemMetrics", EventPriority::NORMAL,
                              PDT_ProductAndServicePerformance | PDT_DeviceConnectivityAndConfiguration)
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
