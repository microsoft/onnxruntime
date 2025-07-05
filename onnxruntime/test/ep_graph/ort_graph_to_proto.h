// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include "core/session/onnxruntime_cxx_api.h"
#include "onnx/onnx_pb.h"

namespace ort_ep_utils {

// Signature of user-provided function to write initializer data. Called by OrtGraphToProto().
//
// If the function returns false, OrtGraphToProto() stores initializer data inline within the TensorProto as raw_data.
//
// Otherwise, if the function returns true, OrtGraphToProto() assumes that this function stores the initializer data
// in a file. In this case, OrtGraphToProto() configures the corresponding TensorProto to point the location and
// offset returned via the `location` and `offset` output parameters.
//
// It is recommended to keep small initializers with byte size <= 127 stored inline the TensorProto to ensure
// ONNX shape inference works correctly with the serialized ONNX model.
using WriteInitializerDataFunc = std::function<bool(const char* name, const void* data, size_t size,
                                                    std::string& location, int64_t& offset)>;

/// <summary>
/// Serializes the provided OrtGraph to a onnx::GraphProto.
/// Allows the caller to provide a function that specifies whether an initializer should be stored
/// within a TensorProto, written to a file, or remain as an in-memory external initializer (not valid ONNX).
/// </summary>
/// <param name="ort_graph"></param>
/// <param name="graph_proto"></param>
/// <param name="write_initializer_data_func"></param>
/// <returns></returns>
Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::GraphProto& graph_proto,
                            WriteInitializerDataFunc write_initializer_data_func = nullptr);

}  // namespace ort_ep_utils
