// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include "core/session/onnxruntime_cxx_api.h"
#include "onnx/onnx_pb.h"

namespace ort_ep_utils {

/// <summary>
/// Signature of user-provided function to handle initializer data. Called by OrtGraphToProto() for every initializer.
///
/// If the function sets the `is_external` output parameter to false, OrtGraphToProto() stores initializer data
/// within the TensorProto as raw_data.
///
/// Otherwise, if the function sets `is_external` to true, OrtGraphToProto() assumes that this function stores the
/// initializer data in a file. In this case, OrtGraphToProto() configures the corresponding TensorProto to point the
/// location and offset returned via the `location` and `offset` output parameters.
///
/// It is recommended to keep small initializers with byte size <= 127 stored inline the TensorProto to ensure
/// ONNX shape inference works correctly with the serialized ONNX model.
/// </summary>
/// <param name="value_info">OrtValueInfo for the initializer. Can be used to query name, type, shape,
///                           and consumer nodes.</param>
/// <param name="data">Opaque pointer to the initializer data.</param>
/// <param name="size">Size in bytes of the initializer data.</param>
/// <param name="is_external">Output parameter set to true if the initializer data is stored externally. The
///                           implementer is responsible for writing the initializer data to file. If set to false,
///                           the initializer will be stored within the TensorProto.</param>
/// <param name="location">Output parameter set to the location (e.g., file) into which the initializer is stored
///                        by the implementer of this function. Ignored if `is_external` is set to false.</param>
/// <param name="offset">Output parameter set to the offset (e.g., file offset) into which the initializer is stored
///                      by the implementer of this function. Ignored if `is_external` is set to false.</param>
/// <returns>An Ort::Status indicating success or an error. Serialization exits if this returns an error.</returns>
using HandleInitializerDataFunc = std::function<Ort::Status(const OrtValueInfo* value_info,
                                                            const void* data, size_t size,
                                                            /*out*/ bool& is_external, /*out*/ std::string& location,
                                                            /*out*/ int64_t& offset)>;

/// <summary>
/// Serializes the provided OrtGraph to a onnx::GraphProto.
/// Allows the caller to provide a function that specifies whether an initializer should be stored
/// within a TensorProto, written to a file, or remain as an in-memory external initializer (not valid ONNX).
/// </summary>
/// <param name="ort_graph">OrtGraph instance to serialize.</param>
/// <param name="graph_proto">Destination GraphProto into which to serialize the input OrtGraph.</param>
/// <param name="handle_initializer_data_func">Optional function called to allow the user to determine
///                                            where the initializer data is stored.</param>
/// <returns>An Ort::Status indicating success or an error.</returns>
Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::GraphProto& graph_proto,
                            HandleInitializerDataFunc handle_initializer_data_func = nullptr);

}  // namespace ort_ep_utils
