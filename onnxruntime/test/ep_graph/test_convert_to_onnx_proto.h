// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_cxx_api.h"
#include "onnx/onnx_pb.h"

namespace test {

Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::GraphProto& graph_proto,
                            bool save_as_external_data = false,
                            const ORTCHAR_T* external_data_location = nullptr,
                            size_t external_data_size_threshold = 1024);

}  // namespace test
