// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/tensorboard/event_writer.h"

#include "onnxruntime_config.h"
#include "orttraining/core/framework/tensorboard/crc32c.h"
#include "core/platform/env.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if defined(HAS_SHORTEN_64_TO_32)
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif
#endif
#include "tensorboard/compat/proto/event.pb.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace training {
namespace tensorboard {

static std::basic_string<PATH_CHAR_TYPE> GenerateFilePath(const std::basic_string<PATH_CHAR_TYPE>& log_dir) {
  std::basic_ostringstream<PATH_CHAR_TYPE> filename;
  if (!log_dir.empty()) {
    ORT_ENFORCE(Env::Default().CreateFolder(log_dir).IsOK(), "Failed to create log directory");
    filename << log_dir << GetPathSep<PATH_CHAR_TYPE>();
  }

  // TODO: get hostname for inclusion in Tensorboard events filename
  filename << "events.out.tfevents." << std::setfill(static_cast<PATH_CHAR_TYPE>('0')) << std::setw(10) << std::time(0) << ".localhost";
  return filename.str();
}

// Return the masked crc32c of data[0,size-1] used by Tensorboard
static inline uint32_t MaskedCrc32c(const char* data, size_t size) {
  uint32_t value = Crc32c(data, size);
  return ((value >> 15) | (value << 17)) + 0xa282ead8ul;
}

template <typename T>
static void Encode(char* buffer, T value) {
  memcpy(buffer, &value, sizeof(value));
}

EventWriter::EventWriter(const std::basic_string<PATH_CHAR_TYPE>& log_dir) : stream_(GenerateFilePath(log_dir), std::ios::binary) {
}

EventWriter::EventWriter(std::ofstream&& stream) : stream_(std::move(stream)) {
}

EventWriter::~EventWriter() {
}

void EventWriter::WriteRecord(const std::string& data) {
  char header[sizeof(uint64_t) + sizeof(uint32_t)];
  Encode(header, static_cast<uint64_t>(data.size()));
  Encode(header + sizeof(uint64_t), MaskedCrc32c(header, sizeof(uint64_t)));

  char footer[sizeof(uint32_t)];
  Encode(footer, MaskedCrc32c(data.data(), data.size()));

  stream_.write(header, sizeof(header));
  stream_.write(data.data(), data.size());
  stream_.write(footer, sizeof(footer));
  stream_.flush();
}

void EventWriter::AddEvent(const ::tensorboard::Event& event) {
  WriteRecord(event.SerializeAsString());
}

void EventWriter::AddHistogram(const std::string& tag, const ::tensorboard::HistogramProto& histogram, int64_t step) {
  ::tensorboard::Summary summary;
  ::tensorboard::Summary::Value* summary_value = summary.add_value();
  summary_value->set_tag(tag.c_str());
  summary_value->mutable_histo()->CopyFrom(histogram);

  AddSummary(summary, step);
}

void EventWriter::AddScalar(const std::string& tag, float value, int64_t step) {
  ::tensorboard::Summary summary;
  ::tensorboard::Summary::Value* summary_value = summary.add_value();
  summary_value->set_tag(tag.c_str());
  summary_value->set_simple_value(value);

  AddSummary(summary, step);
}

void EventWriter::AddSummary(const ::tensorboard::Summary& summary, int64_t step, const std::string& tag_prefix) {
  ::tensorboard::Event event;
  event.set_step(step);
  event.set_wall_time(static_cast<double>(std::time(0)));

  ::tensorboard::Summary* event_summary = event.mutable_summary();
  event_summary->CopyFrom(summary);

  if (!tag_prefix.empty()) {
    for (int i = 0; i < event_summary->value_size(); ++i) {
      ::tensorboard::Summary::Value* summary_value = event_summary->mutable_value(i);
      summary_value->set_tag(tag_prefix + "/" + summary_value->tag());
    }
  }

  AddEvent(event);
}

void EventWriter::AddSummary(const std::string& summary, int64_t step, const std::string& tag_prefix) {
  ::tensorboard::Event event;
  event.set_step(step);
  event.set_wall_time(static_cast<double>(std::time(0)));

  ::tensorboard::Summary* event_summary = event.mutable_summary();
  event_summary->ParseFromString(summary);

  if (!tag_prefix.empty()) {
    for (int i = 0; i < event_summary->value_size(); ++i) {
      ::tensorboard::Summary::Value* summary_value = event_summary->mutable_value(i);
      summary_value->set_tag(tag_prefix + "/" + summary_value->tag());
    }
  }

  AddEvent(event);
}

}  // namespace tensorboard
}  // namespace training
}  // namespace onnxruntime
