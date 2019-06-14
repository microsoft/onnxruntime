// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "spd_log_sink.h"
#include <spdlog/details/null_mutex.h>
#include <spdlog/sinks/syslog_sink.h>
namespace onnxruntime{
    namespace server{
 
 class SysLogSink : public LogSink{
     public:
    SysLogSink() : LogSink(std::make_shared<spdlog::sinks::syslog_sink_mt>()){

    }
};
    }
}

