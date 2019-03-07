// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace training {

using namespace ONNX_NAMESPACE;

AttributeProto MakeAttribute(const std::string& attr_name, const float& value);
AttributeProto MakeAttribute(const std::string& attr_name, const int64_t& value);
AttributeProto MakeAttribute(const std::string& attr_name, const std::string& value);
AttributeProto MakeAttribute(const std::string& attr_name, const TensorProto& value);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<float>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<int64_t>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<std::string>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<TensorProto>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<GraphProto>& values);

}  // namespace training
}  // namespace onnxruntime
