// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/tensorboard/event_writer.h"
#include "core/training/tensorboard/crc32c.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "tensorboard/compat/proto/event.pb.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace training {
namespace tensorboard {

static std::string GetHostname() {
  // TODO: get hostname for inclusion in Tensorboard events filename
  return "localhost";
}

static std::string GenerateFilePath(const std::string& log_dir) {
  std::ostringstream filename;
  if (!log_dir.empty()) {
    filename << log_dir << "/";
  }

  filename << "events.out.tfevents." << std::setfill('0') << std::setw(10) << std::time(0) << "." << GetHostname();
  return filename.str();
}

// Return the masked crc32c of data[0,size-1] used by Tensorboard
static inline uint32_t MaskedCrc32c(const char* data, size_t size) {
  uint32_t value = Crc32c(data, size);
  return ((value >> 15) | (value << 17)) + 0xa282ead8ul;
}

template<typename T>
static void Encode(char* buffer, T value) {
  memcpy(buffer, &value, sizeof(value));
}

EventWriter::EventWriter(const std::string& log_dir) : stream_(GenerateFilePath(log_dir), std::ios::binary) {
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

void EventWriter::AddSummary(const ::tensorboard::Summary& summary, int64_t step) {
  ::tensorboard::Event event;
  event.set_step(step);
  event.set_wall_time(static_cast<double>(std::time(0)));

  event.mutable_summary()->CopyFrom(summary);
  AddEvent(event);
}

}  // namespace tensorboard
}  // namespace training
}  // namespace onnxruntime
