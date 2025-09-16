// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iostream>

#include <System/QnnSystemInterface.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

#ifndef QNN_SYSTEM_PROFILE_API_ENABLED

#include <fstream>

#else

#include <vector>

#include <QnnInterface.h>
#include <QnnProfile.h>
#include <System/QnnSystemProfile.h>

#endif

namespace onnxruntime {

using namespace common;

namespace qnn {
namespace profile {

struct ProfilingInfo {
  uint64_t start_time;
  uint64_t stop_time;

  std::string graph_name;
  uint32_t num_events;
  std::string output_filepath;
  ProfilingMethodType method_type;

  bool tracelogging_provider_ep_enabled = false;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface;
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
#ifndef QNN_SYSTEM_PROFILE_API_ENABLED
 public:
  ~Serializer() {
    cleanup();
  }

  Status Init();

 private:
  void cleanup() {
    if (outfile_.is_open()) {
      outfile_.close();
    }
  }
  std::ofstream outfile_;
#else
 public:
  ~Serializer() {
    event_data_list_.clear();
    system_parent_event_lookup_map_.clear();
    event_profile_id_lookup_map_.clear();
    sub_event_data_map_.clear();
  };

  Status SerializeEvents();

  QnnSystemProfile_ProfileEventV1_t* GetParentSystemEvent(const QnnProfile_EventId_t event_id);

  QnnSystemProfile_ProfileEventV1_t* GetSystemEventPointer(const QnnProfile_EventId_t event_id);

  void AddSubEventList(const uint32_t num_sub_events, QnnSystemProfile_ProfileEventV1_t* event_ptr);

  Status SetParentSystemEvent(const QnnProfile_EventId_t event_id,
                              QnnSystemProfile_ProfileEventV1_t* const system_parent_event);

 private:
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

  std::string output_filename_;
  std::string output_directory_;
  std::vector<QnnSystemProfile_ProfileEventV1_t> event_data_list_;
  std::unordered_map<QnnProfile_EventId_t, QnnSystemProfile_ProfileEventV1_t*> system_parent_event_lookup_map_;
  std::unordered_map<QnnProfile_EventId_t, QnnSystemProfile_ProfileEventV1_t*> event_profile_id_lookup_map_;
  std::unordered_map<QnnSystemProfile_ProfileEventV1_t*, std::vector<QnnSystemProfile_ProfileEventV1_t> > sub_event_data_map_;

  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;
#endif
  const ProfilingInfo profiling_info_;
  bool tracelogging_provider_ep_enabled_ = false;
};

}  // namespace profile
}  // namespace qnn
}  // namespace onnxruntime