// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora_adapters.h"
#include "lora_format_utils.h"

#include <fstream>
#include <stdexcept>

namespace onnxruntime {
namespace lora {
namespace details {

LoraParam::LoraParam(std::string name, OrtValue ort_value)
    : name_(std::move(name)), ort_value_(std::move(ort_value)) {}

BinaryFormatHolder::~BinaryFormatHolder() = default;

void BinaryFormatHolder::Load(const std::filesystem::path& file_path) {
  auto buffer = utils::LoadLoraAdapterBytes(file_path);
  adapter_ = utils::ValidateAndGetAdapterFromBytes(buffer);
  buffer_.emplace<BufferHolder>(std::move(buffer));
}

void BinaryFormatHolder::MemoryMap(const std::filesystem::path& file_path) {
  auto [mapped_memory, file_size] = utils::MemoryMapAdapterFile(file_path);
  auto u8_span = ReinterpretAsSpan<const uint8_t>(gsl::make_span(mapped_memory.get(), file_size));
  adapter_ = utils::ValidateAndGetAdapterFromBytes(u8_span);
  buffer_.emplace<MemMapHolder>(std::move(mapped_memory), file_size);
}

size_t BinaryFormatHolder::GetSize() const {
  if (std::holds_alternative<MemMapHolder>(buffer_)) {
    return std::get<0>(buffer_).file_size_;
  } else if (std::holds_alternative<BufferHolder>(buffer_)) {
    return std::get<1>(buffer_).buffer_.size();
  }
  ORT_THROW("Non-exhaustive visitor for BinaryFormatHolder::GetSize()");
}

}  // namespace details

}  // namespace lora
}  // namespace onnxruntime