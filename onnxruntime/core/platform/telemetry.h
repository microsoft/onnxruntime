// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "core/common/status.h"
#include "core/common/common.h"

namespace onnxruntime {

/**
  * Configuration information for a session.
  * An interface used by the onnxruntime implementation to
  * access operating system functionality for telemetry
  * 
  * look at env.h and the Env objection which is the activation factory
  * for telemetry instances
  * 
  * All Telemetry implementations are safe for concurrent access from
  * multiple threads without any external synchronization.
  */
class Telemetry {
 public:
  virtual ~Telemetry() = default;

  virtual void LogProcessInfo(const std::string& runtimeVersion, bool isRedist) = 0;

  virtual void LogSessionCreation(uint32_t sessionId, int64_t irVersion, const std::string& modelProducerName,
                                  const std::string& modelProducerVersion,const std::string& modelDomain,
                                  const std::vector<std::string>& modelOpsetImports, uint32_t modelPrecision,
                                  const std::string& modelGraphName, const std::string& modelGraphVersion,
                                  const std::unordered_map<std::string, std::string>& modelMetaData,
                                  bool modelFromStream, const std::string& executionProviders) = 0;

  virtual void LogRuntimeError(uint32_t sessionId, const common::Status& status, const char* file,
                               const char* function, uint32_t line) = 0;

  virtual void LogRuntimePerf(uint32_t sessionId, uint32_t runTotalTimeMs) = 0;

 protected:
  // don't create these, use Env::GetTelemetryProvider()
  Telemetry();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Telemetry);
};

}  // namespace onnxruntime
