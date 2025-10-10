// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <System/QnnSystemInterface.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED

#include <vector>

#include <QnnInterface.h>
#include <QnnProfile.h>
#include <System/QnnSystemProfile.h>

#endif

namespace onnxruntime {

namespace qnn {
namespace profile {

struct ProfilingInfo {
  std::string graph_name = "";
  std::string output_filepath = "";

  bool tracelogging_provider_ep_enabled = false;

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  uint64_t start_time = 0;
  uint64_t stop_time = 0;
  uint32_t num_events = 0;

  ProfilingMethodType method_type = ProfilingMethodType::UNKNOWN;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
#endif
};

class Serializer {
 public:
  Serializer(const ProfilingInfo& profiling_info);

  Status ProcessEvent(const QnnProfile_EventId_t event_Id, const std::string& event_level,
                      const QnnProfile_EventData_t& event_data);

  Status ProcessExtendedEvent(const QnnProfile_EventId_t event_id, const std::string& event_level,
                              const QnnProfile_ExtendedEventData_t& event_data);

 private:
#ifdef _WIN32
  void LogQnnProfileEventAsTraceLogging(
      uint64_t timestamp,
      const std::string& message,
      const std::string& qnnScalarValue,
      const std::string& unit,
      const std::string& timingSource,
      const std::string& eventLevel,
      const char* eventIdentifier);
#endif

  // If QNN API is too old, turn Serializer into an ofstream wrapper class
  // Keeps code clean, any performance impacts can be ignored when profiling is enabled
 public:
  ~Serializer() {
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    event_data_list_.clear();
    system_parent_event_lookup_map_.clear();
    event_profile_id_lookup_map_.clear();
    sub_event_data_map_.clear();
#endif
  }

  Status InitFile();

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
 public:
  Status SerializeEvents();

  QnnSystemProfile_ProfileEventV1_t* GetParentSystemEvent(const QnnProfile_EventId_t event_id);

  QnnSystemProfile_ProfileEventV1_t* GetSystemEventPointer(const QnnProfile_EventId_t event_id);

  void AddSubEventList(const uint32_t num_sub_events, QnnSystemProfile_ProfileEventV1_t* event_ptr);

  Status SetParentSystemEvent(const QnnProfile_EventId_t event_id,
                              QnnSystemProfile_ProfileEventV1_t* const system_parent_event);

 private:
  class ManagedSerializationTargetHandle {
   public:
    ManagedSerializationTargetHandle(const QnnSystemProfile_SerializationTargetHandle_t& raw_handle,
                                     QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface) : qnn_system_interface_(qnn_system_interface),
                                                                                           handle_(raw_handle) {}

    ~ManagedSerializationTargetHandle() {
      auto status = FreeHandle();
      ORT_UNUSED_PARAMETER(status);
    }

    Qnn_ErrorHandle_t FreeHandle() {
      return qnn_system_interface_.systemProfileFreeSerializationTarget(handle_);
    }

   private:
    QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;
    QnnSystemProfile_SerializationTargetHandle_t handle_;
  };  // ManagedSerializationTargetHandle

  QnnSystemProfile_ProfileEventV1_t* AddEvent(const QnnProfile_EventId_t event_Id,
                                              const QnnProfile_EventData_t event);

  QnnSystemProfile_ProfileEventV1_t* AddExtendedEvent(const QnnProfile_EventId_t event_id,
                                                      const QnnProfile_ExtendedEventData_t event);

  QnnSystemProfile_ProfileEventV1_t* AddSubEvent(const QnnProfile_EventId_t event_id,
                                                 const QnnProfile_EventData_t& sub_event,
                                                 QnnSystemProfile_ProfileEventV1_t* const system_parent_event);

  QnnSystemProfile_ProfileEventV1_t* AddExtendedSubEvent(const QnnProfile_EventId_t event_id,
                                                         const QnnProfile_ExtendedEventData_t& sub_event,
                                                         QnnSystemProfile_ProfileEventV1_t* const system_parent_event);

  QnnSystemProfile_ProfileEventV1_t* CreateSystemEvent(std::vector<QnnSystemProfile_ProfileEventV1_t>& event_list,
                                                       const QnnProfile_EventId_t event_id,
                                                       QnnProfile_EventData_t event_data);

  QnnSystemProfile_ProfileEventV1_t* CreateSystemExtendedEvent(std::vector<QnnSystemProfile_ProfileEventV1_t>& event_list,
                                                               const QnnProfile_EventId_t event_id,
                                                               QnnProfile_ExtendedEventData_t event_data);

  std::string qnn_log_filename_;
  std::string output_directory_;
  std::vector<QnnSystemProfile_ProfileEventV1_t> event_data_list_;
  std::unordered_map<QnnProfile_EventId_t, QnnSystemProfile_ProfileEventV1_t*> system_parent_event_lookup_map_;
  std::unordered_map<QnnProfile_EventId_t, QnnSystemProfile_ProfileEventV1_t*> event_profile_id_lookup_map_;
  std::unordered_map<QnnSystemProfile_ProfileEventV1_t*, std::vector<QnnSystemProfile_ProfileEventV1_t> > sub_event_data_map_;

  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;
#endif  // QNN_SYSTEM_PROFILE_API_ENABLED
  const ProfilingInfo profiling_info_;
  bool tracelogging_provider_ep_enabled_ = false;
  std::ofstream outfile_;
};

}  // namespace profile
}  // namespace qnn
}  // namespace onnxruntime