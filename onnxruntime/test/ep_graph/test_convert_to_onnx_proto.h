// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include "core/session/onnxruntime_cxx_api.h"
#include "onnx/onnx_pb.h"

namespace test {

// Signature of user-provided function to write initializer data.
// If function returns false, initializer data will be stored inline within the TensorProto as raw_data.
// If function returns true, the implementation may store the initializer data in a file and should set the file
// location and offset via the output parameters.
using WriteInitializerDataFunc = std::function<bool(const char* name, const void* data, size_t size,
                                                    std::string& location, int64_t& offset)>;

Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::GraphProto& graph_proto,
                            WriteInitializerDataFunc write_initializer_data_func = nullptr);

}  // namespace test
