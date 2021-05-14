/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <core/common/common.h>
#include <core/common/safeint.h>

#include "NeuralNetworksWrapper.h"

namespace android {
namespace nn {
namespace wrapper {

OperandType::OperandType(Type type, const std::vector<uint32_t>& d, float scale, int32_t zeroPoint)
    : type(type), dimensions(d) {
  operandType = {
      .type = static_cast<int32_t>(type),
      .dimensionCount = static_cast<uint32_t>(dimensions.size()),
      .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
      .scale = scale,
      .zeroPoint = zeroPoint,
  };
}

OperandType::OperandType(Type type, const std::vector<uint32_t>& d, SymmPerChannelQuantParams&& channelQuant)
    : type(type), dimensions(d), channelQuant(std::move(channelQuant)) {
  operandType = {
      .type = static_cast<int32_t>(type),
      .dimensionCount = static_cast<uint32_t>(dimensions.size()),
      .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
      .scale = 0.0f,
      .zeroPoint = 0,
  };
}

OperandType::OperandType(const OperandType& other)
    : operandType(other.operandType),
      type(other.type),
      dimensions(other.dimensions),
      channelQuant(other.channelQuant) {
  operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
}

OperandType& OperandType::operator=(const OperandType& other) {
  if (this != &other) {
    type = other.type;
    dimensions = other.dimensions;
    operandType = other.operandType;
    channelQuant = other.channelQuant;
    operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
  }

  return *this;
}

size_t OperandType::GetElementByteSize() const {
  size_t element_size;
  switch (type) {
    case Type::TENSOR_BOOL8:
      element_size = 1;
      break;
    case Type::TENSOR_FLOAT16:
      element_size = 2;
      break;
    case Type::TENSOR_FLOAT32:
    case Type::FLOAT32:
      element_size = 4;
      break;
    case Type::TENSOR_INT32:
      element_size = 4;
      break;
    case Type::TENSOR_QUANT8_SYMM_PER_CHANNEL:
      element_size = 1;
      break;
    case Type::TENSOR_QUANT8_ASYMM:
      element_size = 1;
      break;
    case Type::TENSOR_QUANT16_SYMM:
      element_size = 2;
      break;
    case Type::TENSOR_QUANT16_ASYMM:
      element_size = 2;
      break;
    default:
      ORT_THROW("Wrong type: " + TypeToStr(type));
  }

  return element_size;
}

size_t OperandType::GetOperandBlobByteSize() const {
  // use uin64_t even dimension is uint32_t to prevent overflow
  uint64_t num_elements = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<uint64_t>());
  return SafeInt<size_t>(num_elements) * GetElementByteSize();
}

void OperandType::SetDimensions(const std::vector<uint32_t>& d) {
  dimensions = d;
  operandType.dimensionCount = dimensions.size();
  operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
}

}  // namespace wrapper
}  // namespace nn
}  // namespace android