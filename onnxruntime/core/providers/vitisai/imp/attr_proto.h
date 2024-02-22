// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <functional>
#include "vaip/my_ort.h"
#include "core/common/gsl.h"

namespace vaip {

ONNX_NAMESPACE::AttributeProto* attr_proto_new_int(const std::string& name, int64_t value);
ONNX_NAMESPACE::AttributeProto* attr_proto_new_float(const std::string& name, float value);
ONNX_NAMESPACE::AttributeProto* attr_proto_new_string(const std::string& name, const std::string& value);
ONNX_NAMESPACE::AttributeProto* attr_proto_new_tensor(const std::string& name, const ONNX_NAMESPACE::TensorProto& value);
ONNX_NAMESPACE::AttributeProto* attr_proto_new_ints(const std::string& name, const std::vector<int64_t>& value);
ONNX_NAMESPACE::AttributeProto* attr_proto_new_floats(const std::string& name, const std::vector<float>& value);
ONNX_NAMESPACE::AttributeProto* attr_proto_new_strings(const std::string& name, const std::vector<std::string>& value);

/// attr_proto getters
int64_t attr_proto_get_int(const ONNX_NAMESPACE::AttributeProto& attr);
float attr_proto_get_float(const ONNX_NAMESPACE::AttributeProto& attr);
const std::string& attr_proto_get_string(const ONNX_NAMESPACE::AttributeProto& attr);
const ONNX_NAMESPACE::TensorProto& attr_proto_get_tensor(const ONNX_NAMESPACE::AttributeProto& attr);
gsl::span<const int64_t> attr_proto_get_ints(const ONNX_NAMESPACE::AttributeProto& attr);
gsl::span<const float> attr_proto_get_floats(const ONNX_NAMESPACE::AttributeProto& attr);
std::vector<std::string> attr_proto_get_strings(const ONNX_NAMESPACE::AttributeProto& attr);

}  // namespace vaip
