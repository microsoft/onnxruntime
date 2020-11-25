// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include "core/platform/path_lib.h"

namespace tensorboard {
  class Event;
  class HistogramProto;
  class Summary;
}

namespace onnxruntime {
namespace training {
namespace tensorboard {

class EventWriter {
 public:
  EventWriter(const std::basic_string<PATH_CHAR_TYPE>& log_dir);
  EventWriter(std::ofstream&& stream);
  ~EventWriter();

  void AddEvent(const ::tensorboard::Event& event);
  void AddHistogram(const std::string& tag, const ::tensorboard::HistogramProto& histogram, int64_t step = 0);
  void AddScalar(const std::string& tag, float value, int64_t step = 0);
  void AddSummary(const ::tensorboard::Summary& summary, int64_t step = 0, const std::string& tag_prefix = "");
  void AddSummary(const std::string& summary, int64_t step = 0, const std::string& tag_prefix = "");

 private:
  void WriteRecord(const std::string& data);

  std::ofstream stream_;
};

}  // namespace tensorboard
}  // namespace training
}  // namespace onnxruntime
