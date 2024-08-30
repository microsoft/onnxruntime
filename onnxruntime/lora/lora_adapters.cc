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

void BinaryFormatHolder::Load(const std::filesystem::path& file_path) {
  auto buffer = utils::LoadLoraAdapterBytes(file_path);
  adapter_ = utils::ValidateAndGetAdapterFromBytes(buffer);
  buffer_.emplace<BufferHolder>(std::move(buffer));
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