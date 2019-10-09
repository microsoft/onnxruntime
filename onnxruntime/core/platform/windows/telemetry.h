// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/telemetry.h"

// Note: this needs to get moved to a release pipeline still (paulm)
// ***
#define TraceLoggingOptionMicrosoftTelemetry() \
  TraceLoggingOptionGroup(0x4f50731a, 0x89cf, 0x4782, 0xb3, 0xe0, 0xdc, 0xe8, 0xc9, 0x4, 0x76, 0xba)
#define MICROSOFT_KEYWORD_MEASURES 0x0000400000000000  // Bit 46
#define TelemetryPrivacyDataTag(tag) TraceLoggingUInt64((tag), "PartA_PrivTags")
#define PDT_ProductAndServiceUsage 0x0000000002000000u
// ***

namespace onnxruntime {

/**
  * derives and implments a Telemetry provider on Windows
  */
class WindowsTelemetry : public Telemetry {

 public:

  // these are allowed to be created, WindowsEnv will create one
  WindowsTelemetry();
  ~WindowsTelemetry();

  void LogProcessInfo(const std::string& runtimeVersion, bool isRedist);

  void LogSessionCreation(uint32_t sessionId, int64_t irVersion, const std::string& modelProducerName,
                          const std::string& modelProducerVersion, const std::string& modelDomain,
                          const std::vector<std::string>& modelOpsetImports, uint32_t modelPrecision,
                          const std::string& modelGraphName, const std::string& modelGraphVersion,
                          const std::unordered_map<std::string, std::string>& modelMetaData,
                          bool modelFromStream, const std::string& executionProviders);
    
  void LogRuntimeError(uint32_t sessionId, const common::Status& status, const char* file,
                       const char* function, uint32_t line);

  void LogRuntimePerf(uint32_t sessionId, uint32_t runTotalTimeMs);

 private:
  bool register_succeeded_ = false;
};

}  // namespace onnxruntime
