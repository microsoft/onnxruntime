// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_profile_serializer.h"
#include "core/providers/qnn/qnn_telemetry.h"

namespace onnxruntime {
namespace qnn {
namespace profile {

const std::unordered_map<QnnProfile_EventUnit_t, std::string>& GetUnitStringMap() {
  static const std::unordered_map<QnnProfile_EventUnit_t, std::string> unitStringMap = {
      {QNN_PROFILE_EVENTUNIT_MICROSEC, "US"},
      {QNN_PROFILE_EVENTUNIT_BYTES, "BYTES"},
      {QNN_PROFILE_EVENTUNIT_CYCLES, "CYCLES"},
      {QNN_PROFILE_EVENTUNIT_COUNT, "COUNT"},
      {QNN_PROFILE_EVENTUNIT_OBJECT, "OBJECT"},
      {QNN_PROFILE_EVENTUNIT_BACKEND, "BACKEND"}};
  return unitStringMap;
}

const std::string& GetUnitString(QnnProfile_EventUnit_t unitType) {
  const auto& unitStringMap = GetUnitStringMap();
  auto it = unitStringMap.find(unitType);
  if (it != unitStringMap.end()) {
    return it->second;
  }
  static const std::string unknown = "UNKNOWN";
  return unknown;
}

std::string GetEventTypeString(QnnProfile_EventType_t event_type) {
  // Interpret the event type
  switch (event_type) {
    case QNN_PROFILE_EVENTTYPE_INIT:
      return "INIT";
    case QNN_PROFILE_EVENTTYPE_FINALIZE:
      return "FINALIZE";
    case QNN_PROFILE_EVENTTYPE_EXECUTE:
      return "EXECUTE";
    case QNN_PROFILE_EVENTTYPE_NODE:
      return "NODE";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_QUEUE_WAIT:
      return "EXECUTE QUEUE WAIT";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_PREPROCESS:
      return "EXECUTE PREPROCESS";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE:
      return "EXECUTE DEVICE";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_POSTPROCESS:
      return "EXECUTE POSTPROCESS";
    case QNN_PROFILE_EVENTTYPE_DEINIT:
      return "DE-INIT";
    case QNN_PROFILE_EVENTTYPE_BACKEND:
      return "BACKEND";
    default:
      if (event_type > QNN_PROFILE_EVENTTYPE_BACKEND) {
        return "BACKEND";
      }
      return "UNKNOWN";
  }
}

std::string ExtractQnnScalarValue(const Qnn_Scalar_t& scalar) {
  switch (scalar.dataType) {
    case QNN_DATATYPE_INT_8:
      return std::to_string(static_cast<int>(scalar.int8Value));
    case QNN_DATATYPE_INT_16:
      return std::to_string(scalar.int16Value);
    case QNN_DATATYPE_INT_32:
      return std::to_string(scalar.int32Value);
    case QNN_DATATYPE_INT_64:
      return std::to_string(scalar.int64Value);
    case QNN_DATATYPE_UINT_8:
      return std::to_string(static_cast<unsigned int>(scalar.uint8Value));
    case QNN_DATATYPE_UINT_16:
      return std::to_string(scalar.uint16Value);
    case QNN_DATATYPE_UINT_32:
      return std::to_string(scalar.uint32Value);
    case QNN_DATATYPE_UINT_64:
      return std::to_string(scalar.uint64Value);
    case QNN_DATATYPE_FLOAT_16:
      return std::to_string(scalar.floatValue);
    case QNN_DATATYPE_FLOAT_32:
      return std::to_string(scalar.floatValue);
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_32:
      return std::to_string(scalar.int32Value);  // Assume using int types for signed fixed points.
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return std::to_string(scalar.uint32Value);  // Assume using unsigned int types for unsigned fixed points.
    case QNN_DATATYPE_BOOL_8:
      return scalar.bool8Value ? "true" : "false";
    case QNN_DATATYPE_STRING:
      return scalar.stringValue ? scalar.stringValue : "NULL";
    default:
      return "UNKNOWN";
  }
}

#ifdef _WIN32
void Serializer::LogQnnProfileEventAsTraceLogging(
    uint64_t timestamp,
    const std::string& message,
    const std::string& qnnScalarValue,
    const std::string& unit,
    const std::string& timingSource,
    const std::string& event_level,
    const char* eventIdentifier) {
  QnnTelemetry& qnn_telemetry = QnnTelemetry::Instance();
  qnn_telemetry.LogQnnProfileEvent(timestamp, message, qnnScalarValue, unit, timingSource, event_level, eventIdentifier);
}
#endif

Status Serializer::ProcessEvent(const QnnProfile_EventId_t event_id, const std::string& event_level,
                                const QnnProfile_EventData_t& event_data) {
  const std::string& message = GetEventTypeString(event_data.type);
  const std::string& unit = GetUnitString(event_data.unit);

  ORT_UNUSED_PARAMETER(event_id);

  if (outfile_) {
    outfile_ << "UNKNOWN"
             << ","
             << message << ","
             << event_data.value << ","
             << unit << ","
             << "BACKEND"
             << ","
             << event_level << ","
             << (event_data.identifier ? event_data.identifier : "NULL") << "\n";
  }
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  QnnSystemProfile_ProfileEventV1_t* created_event = nullptr;
  if (event_level == "SUB-EVENT") {
    auto parent_system_event = GetParentSystemEvent(event_id);
    ORT_RETURN_IF(parent_system_event == nullptr, "Serialization of subevent failed: parent event pointer is null");
    created_event = AddSubEvent(event_id, event_data, parent_system_event);
  } else {
    created_event = AddEvent(event_id, event_data);
  }

  ORT_RETURN_IF(created_event == nullptr, "Serialization of event failed: Unable to create system profile event");
#endif

  if (tracelogging_provider_ep_enabled_) {
#ifdef _WIN32
    LogQnnProfileEventAsTraceLogging(
        (uint64_t)0,
        message,
        std::to_string(event_data.value),
        unit,
        "BACKEND",
        event_level,
        (event_data.identifier ? event_data.identifier : "NULL"));
#endif
  }

  return Status::OK();
}

Status Serializer::ProcessExtendedEvent(const QnnProfile_EventId_t event_id, const std::string& event_level,
                                        const QnnProfile_ExtendedEventData_t& event_data) {
  // need to check the version first
  const std::string& message = GetEventTypeString(event_data.v1.type);
  const std::string& unit = GetUnitString(event_data.v1.unit);

  ORT_UNUSED_PARAMETER(event_id);

  if (outfile_) {
    if (event_data.version == QNN_PROFILE_DATA_VERSION_1) {
      outfile_ << event_data.v1.timestamp << ","
               << message << ","
               << ExtractQnnScalarValue(event_data.v1.value) << ","
               << unit << ","
               << "BACKEND"
               << ","
               << event_level << ","
               << (event_data.v1.identifier ? event_data.v1.identifier : "NULL")
               << "\n";
    }
  }
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  QnnSystemProfile_ProfileEventV1_t* created_event = nullptr;
  if (event_level == "SUB-EVENT") {
    auto parent_system_event = GetParentSystemEvent(event_id);
    ORT_RETURN_IF(parent_system_event == nullptr, "Serialization of subevent failed: parent event pointer is null");
    created_event = AddExtendedSubEvent(event_id, event_data, parent_system_event);
  } else {
    created_event = AddExtendedEvent(event_id, event_data);
  }

  ORT_RETURN_IF(created_event == nullptr, "Serialization of event failed: Unable to create system profile event");
#endif

  if (tracelogging_provider_ep_enabled_) {
#ifdef _WIN32
    LogQnnProfileEventAsTraceLogging(
        event_data.v1.timestamp,
        message,
        ExtractQnnScalarValue(event_data.v1.value),
        unit,
        "BACKEND",
        event_level,
        (event_data.v1.identifier ? event_data.v1.identifier : "NULL"));
#endif
  }

  return Status::OK();
}

Status Serializer::InitCsvFile() {
  auto output_filepath = profiling_info_.csv_output_filepath;
  // Write to CSV in append mode
  std::ifstream infile(output_filepath.c_str());
  bool exists = infile.good();
  if (infile.is_open()) {
    infile.close();
  }

  outfile_.open(output_filepath.c_str(), std::ios_base::app);
  ORT_RETURN_IF(!outfile_.is_open(), "Failed to open profiling file: ", output_filepath);
  // If file didn't exist before, write the header
  if (!exists) {
    outfile_ << "Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier\n";
  }

  return Status::OK();
}

Serializer::Serializer(const ProfilingInfo& profiling_info,
                       QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface,
                       bool tracelogging_provider_ep_enabled)
    : profiling_info_(profiling_info),
      qnn_system_interface_(qnn_system_interface),
      tracelogging_provider_ep_enabled_(tracelogging_provider_ep_enabled) {
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  std::filesystem::path output_fs_filepath(profiling_info.csv_output_filepath);
  qnn_log_filename_ = output_fs_filepath.filename().string();
  // Remove extension (assumed to be ".csv") then add "_qnn.log"
  size_t extension_start_idx = qnn_log_filename_.rfind(".");
  qnn_log_filename_ = qnn_log_filename_.substr(0, extension_start_idx);
  qnn_log_filename_.append("_qnn.log");

  std::filesystem::path abs_output_path;
  if (output_fs_filepath.has_root_path()) {
    abs_output_path = output_fs_filepath.parent_path();
  } else {
    abs_output_path = std::filesystem::current_path() / output_fs_filepath.parent_path();
  }
  output_directory_ = abs_output_path.string();

  event_data_list_.reserve(profiling_info.num_events);
#endif
}

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
QnnSystemProfile_MethodType_t ParseMethodType(ProfilingMethodType method_type) {
  switch (method_type) {
    case ProfilingMethodType::EXECUTE:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE;
    case ProfilingMethodType::FINALIZE:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE;
    case ProfilingMethodType::EXECUTE_ASYNC:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE_ASYNC;
    case ProfilingMethodType::CREATE_FROM_BINARY:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_CREATE_FROM_BINARY;
    case ProfilingMethodType::DEINIT:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_DEINIT;
    case ProfilingMethodType::CONTEXT_CREATE:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_CONTEXT_CREATE;
    case ProfilingMethodType::COMPOSE_GRAPHS:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_COMPOSE_GRAPHS;
    case ProfilingMethodType::EXECUTE_IPS:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_EXECUTE_IPS;
    case ProfilingMethodType::GRAPH_COMPONENT:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_GRAPH_COMPONENT;
    case ProfilingMethodType::LIB_LOAD:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_BACKEND_LIB_LOAD;
    case ProfilingMethodType::APPLY_BINARY_SECTION:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_APPLY_BINARY_SECTION;
    case ProfilingMethodType::CONTEXT_FINALIZE:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_FINALIZE;
    default:
      return QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE;
  }
}

std::string GetSystemProfileErrorString(Qnn_ErrorHandle_t error) {
  switch (error) {
    case QNN_SYSTEM_PROFILE_ERROR_UNSUPPORTED_FEATURE:
      return "Unsupported Feature";
    case QNN_SYSTEM_PROFILE_ERROR_INVALID_HANDLE:
      return "Invalid Handle";
    case QNN_SYSTEM_PROFILE_ERROR_INVALID_ARGUMENT:
      return "Invalid Argument";
    case QNN_SYSTEM_PROFILE_ERROR_MEM_ALLOC:
      return "Memory Allocation Error";
    default:
      return "Unknown";
  }
}

QnnSystemProfile_ProfileEventV1_t* Serializer::AddEvent(const QnnProfile_EventId_t event_id,
                                                        const QnnProfile_EventData_t event) {
  return CreateSystemEvent(event_data_list_, event_id, event);
}
QnnSystemProfile_ProfileEventV1_t* Serializer::AddExtendedEvent(const QnnProfile_EventId_t event_id,
                                                                const QnnProfile_ExtendedEventData_t event) {
  return CreateSystemExtendedEvent(event_data_list_, event_id, event);
}

QnnSystemProfile_ProfileEventV1_t* Serializer::AddSubEvent(const QnnProfile_EventId_t event_id,
                                                           const QnnProfile_EventData_t& sub_event,
                                                           QnnSystemProfile_ProfileEventV1_t* const parent_system_event) {
  if (sub_event_data_map_.find(parent_system_event) == sub_event_data_map_.end()) {
    return nullptr;
  }

  auto& sub_event_list = sub_event_data_map_.at(parent_system_event);
  return CreateSystemEvent(sub_event_list, event_id, sub_event);
}
QnnSystemProfile_ProfileEventV1_t* Serializer::AddExtendedSubEvent(const QnnProfile_EventId_t event_id,
                                                                   const QnnProfile_ExtendedEventData_t& sub_event,
                                                                   QnnSystemProfile_ProfileEventV1_t* const parent_system_event) {
  if (sub_event_data_map_.find(parent_system_event) == sub_event_data_map_.end()) {
    return nullptr;
  }

  auto& sub_event_list = sub_event_data_map_.at(parent_system_event);
  return CreateSystemExtendedEvent(sub_event_list, event_id, sub_event);
}

Status Serializer::SerializeEventsToQnnLog() {
  bool result = nullptr == qnn_system_interface_.systemProfileCreateSerializationTarget ||
                nullptr == qnn_system_interface_.systemProfileSerializeEventData ||
                nullptr == qnn_system_interface_.systemProfileFreeSerializationTarget;
  ORT_RETURN_IF(result, "Failed to get system profile API pointers.");

  auto method_type = profiling_info_.method_type;
  ORT_RETURN_IF(method_type == ProfilingMethodType::UNKNOWN, "Invalid serialization method type");

  QnnSystemProfile_SerializationTargetConfig_t config;
  config.type = QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER;

  std::string backend_version(std::to_string(QNN_API_VERSION_MAJOR) + "." + std::to_string(QNN_API_VERSION_MINOR) + "." + std::to_string(QNN_API_VERSION_PATCH));

  std::string app_version(std::to_string(ORT_API_VERSION));
  config.serializationHeader.appName = "OnnxRuntime";
  config.serializationHeader.appVersion = app_version.c_str();
  config.serializationHeader.backendVersion = backend_version.c_str();

  QnnSystemProfile_SerializationTargetFile_t serialization_file{qnn_log_filename_.c_str(), output_directory_.c_str()};
  QnnSystemProfile_SerializationTarget_t serialization_target = {
      QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE,
      {serialization_file}};

  QnnSystemProfile_SerializationTargetHandle_t serialization_target_handle;

  auto status = qnn_system_interface_.systemProfileCreateSerializationTarget(serialization_target, &config, 1,
                                                                             &serialization_target_handle);
  ORT_RETURN_IF(QNN_SYSTEM_PROFILE_NO_ERROR != status, "Failed to create serialization target handle: ",
                GetSystemProfileErrorString(status));

  ManagedSerializationTargetHandle managed_target_handle(serialization_target_handle, qnn_system_interface_);

  // Set subevent data pointers for all event data
  // Must be done here as underlying array ptrs can change as vectors are resized
  for (auto it = sub_event_data_map_.begin(); it != sub_event_data_map_.end(); it++) {
    it->first->profileSubEventData = it->second.data();
    it->first->numSubEvents = static_cast<uint32_t>(it->second.size());
  }

  // Create QnnSystemProfile_ProfileData_t obj here
  QnnSystemProfile_ProfileData_t system_profile_data = QNN_SYSTEM_PROFILE_DATA_INIT;
  system_profile_data.version = QNN_SYSTEM_PROFILE_DATA_VERSION_1;
  system_profile_data.v1.header.startTime = profiling_info_.start_time;
  system_profile_data.v1.header.stopTime = profiling_info_.stop_time;
  system_profile_data.v1.header.graphName = profiling_info_.graph_name.c_str();
  system_profile_data.v1.header.methodType = ParseMethodType(method_type);
  system_profile_data.v1.profilingEvents = event_data_list_.data();
  system_profile_data.v1.numProfilingEvents = static_cast<uint32_t>(event_data_list_.size());

  std::vector<const QnnSystemProfile_ProfileData_t*> system_profile_data_list = {&system_profile_data};
  status = qnn_system_interface_.systemProfileSerializeEventData(serialization_target_handle,
                                                                 system_profile_data_list.data(),
                                                                 1);

  ORT_RETURN_IF(QNN_SYSTEM_PROFILE_NO_ERROR != status, "Failed to serialize QNN profiling data: ",
                GetSystemProfileErrorString(status));

  status = managed_target_handle.FreeHandle();
  ORT_RETURN_IF(QNN_SYSTEM_PROFILE_NO_ERROR != status, "Failed to free serialization target: ",
                GetSystemProfileErrorString(status));

  return Status::OK();
}

void Serializer::AddSubEventList(const uint32_t num_sub_events, QnnSystemProfile_ProfileEventV1_t* event_ptr) {
  if (num_sub_events > 0U) {
    auto it = sub_event_data_map_.emplace(event_ptr, std::vector<QnnSystemProfile_ProfileEventV1_t>()).first;
    it->second.reserve(num_sub_events);
  }
}

QnnSystemProfile_ProfileEventV1_t* Serializer::GetSystemEventPointer(QnnProfile_EventId_t event_id) {
  auto it = event_profile_id_lookup_map_.find(event_id);
  if (it == event_profile_id_lookup_map_.end()) {
    return nullptr;
  }

  return it->second;
}

Status Serializer::SetParentSystemEvent(
    const QnnProfile_EventId_t event_id,
    QnnSystemProfile_ProfileEventV1_t* const system_parent_event) {
  ORT_RETURN_IF(!(system_parent_event_lookup_map_.emplace(event_id, system_parent_event).second),
                "Failed to add subevent-parent event mapping");
  return Status::OK();
}
QnnSystemProfile_ProfileEventV1_t* Serializer::GetParentSystemEvent(const QnnProfile_EventId_t event_id) {
  if (system_parent_event_lookup_map_.find(event_id) == system_parent_event_lookup_map_.end()) {
    return nullptr;
  }

  return system_parent_event_lookup_map_.at(event_id);
}

QnnSystemProfile_ProfileEventV1_t* Serializer::CreateSystemEvent(
    std::vector<QnnSystemProfile_ProfileEventV1_t>& event_list,
    QnnProfile_EventId_t event_id,
    QnnProfile_EventData_t event_data) {
  auto system_event = &(event_list.emplace_back());

  system_event->type = QNN_SYSTEM_PROFILE_EVENT_DATA;
  system_event->eventData = event_data;
  system_event->profileSubEventData = NULL;
  system_event->numSubEvents = 0;

  event_profile_id_lookup_map_.emplace(event_id, system_event);

  return system_event;
}

QnnSystemProfile_ProfileEventV1_t* Serializer::CreateSystemExtendedEvent(std::vector<QnnSystemProfile_ProfileEventV1_t>& event_list,
                                                                         QnnProfile_EventId_t event_id,
                                                                         QnnProfile_ExtendedEventData_t event_data) {
  auto system_event = &(event_list.emplace_back());

  system_event->type = QNN_SYSTEM_PROFILE_EXTENDED_EVENT_DATA;
  system_event->extendedEventData = event_data;
  system_event->profileSubEventData = NULL;
  system_event->numSubEvents = 0;

  event_profile_id_lookup_map_.emplace(event_id, system_event);

  return system_event;
}
#endif  // QNN_SYSTEM_PROFILE_API_ENABLED

}  // namespace profile
}  // namespace qnn
}  // namespace onnxruntime