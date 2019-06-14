#include "spd_log_sink.h"

namespace onnxruntime{
    namespace server{
        ORTFormatter::ORTFormatter(const logging::Timestamp& timestamp, const char severityPrefix, const std::string& logger_id): 
  timestamp_(timestamp), severityPrefix_(severityPrefix), logger_id_(logger_id){
  };

  void ORTFormatter::format(const spdlog::details::log_msg &msg, fmt::memory_buffer &dest) {
      //HH:MM:SS
      auto time = std::chrono::system_clock::to_time_t(timestamp_);
      auto tm = std::gmtime(&time);

      spdlog::details::fmt_helper::append_int(tm->tm_hour, dest);
      dest.push_back(':');
      spdlog::details::fmt_helper::append_int(tm->tm_min, dest);
      dest.push_back(':');
      spdlog::details::fmt_helper::append_int(tm->tm_sec, dest);
      dest.push_back(' ');
      dest.push_back('[');
      dest.push_back(severityPrefix_);
      dest.push_back(':');
      spdlog::details::fmt_helper::append_string_view(msg.logger_name, dest);
      dest.push_back(':');
      spdlog::details::fmt_helper::append_string_view(logger_id_.c_str(), dest);
      dest.push_back(',');
      dest.push_back(' ');
      spdlog::details::fmt_helper::append_string_view(msg.source.filename, dest);
      dest.push_back(':');
      spdlog::details::fmt_helper::append_int(msg.source.line, dest);
      dest.push_back(' ');
      spdlog::details::fmt_helper::append_string_view(msg.source.funcname, dest);
      dest.push_back(']');
      dest.push_back(' ');
      spdlog::details::fmt_helper::append_string_view(msg.payload, dest);
  }


void LogSink::SendImpl(const logging::Timestamp& timestamp, const std::string& logger_id, const logging::Capture& message){
    if(sink_->should_log((spdlog::level::level_enum)message.Severity())){
       auto loc = message.Location();
    sink_->set_formatter(std::make_unique<spdlog::formatter>(timestamp, message.SeverityPrefix(), logger_id));
    sink_->log(spdlog::details::log_msg{spdlog::source_loc{loc.file_and_path.c_str(), loc.line_num, loc.function.c_str()}, message.Category(), (spdlog::level::level_enum)message.Severity(), message.Message()});
    }
}
    }
}