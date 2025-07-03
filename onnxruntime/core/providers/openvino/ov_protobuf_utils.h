// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once
namespace onnxruntime {
namespace openvino_ep {
float get_float_initializer_data(const void* initializer);
void set_float_initializer_data(const void* initializer, float data);
}
}  // namespace onnxruntime
