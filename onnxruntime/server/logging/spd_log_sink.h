// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/formatter.h>
#include <spdlog/details/fmt_helper.h>
#include <spdlog/sinks/sink.h>
#include "core/common/logging/logging.h"
#include "core/common/logging/isink.h"

namespace onnxruntime {
namespace server {


class ORTFormatter : public spdlog::formatter{
  
 private:
    const char severityPrefix_;
    const std::string logger_id_;
    const logging::Timestamp timestamp_;

  public:
  ORTFormatter(const logging::Timestamp& timestamp, const char severityPrefix, const std::string& logger_id);
  ~ORTFormatter() = default;
  void format(const spdlog::details::log_msg &msg, fmt::memory_buffer &dest) override;

  std::unique_ptr<spdlog::formatter> clone() const override{
      return std::make_unique<ORTFormatter>(timestamp_, severityPrefix_, logger_id_);
  };

 
};

class LogSink : public onnxruntime::logging::ISink {

  spdlog::sink_ptr sink_;

protected: 
  LogSink(spdlog::sink_ptr sink) : sink_(sink){
    
  }

public:
   void SendImpl(const logging::Timestamp& timestamp, const std::string& logger_id, const logging::Capture& message);
  };

}  // namespace server
}  // namespace onnxruntime
