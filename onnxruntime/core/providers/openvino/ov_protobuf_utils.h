// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace openvino_ep {

using TensorRankMap = std::unordered_map<std::string, int64_t>;

float get_float_initializer_data(const void* initializer);
void set_float_initializer_data(const void* initializer, float data);
void normalize_negative_resize_axes(void* model_proto, const TensorRankMap& tensor_ranks);

}  // namespace openvino_ep
}  // namespace onnxruntime
