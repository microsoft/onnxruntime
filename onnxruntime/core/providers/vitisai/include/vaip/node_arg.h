// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "vaip/dll_safe.h"
#include "vaip/my_ort.h"
namespace vaip {
using namespace onnxruntime;

bool node_arg_is_exists(const NodeArg& node_arg);
bool node_arg_is_constant(const Graph& graph, const NodeArg& node_arg);
NodeArg& node_arg_clone(Graph& graph, const NodeArg& node_arg,
                        const std::string& name);
NodeArg& node_arg_new(Graph& graph,
                      const std::string& name, const std::vector<int64_t>* shape, int element_type);

int node_arg_get_element_type(const NodeArg& node_arg);
vaip_core::DllSafe<std::vector<int64_t>> node_arg_get_shape_i64(const NodeArg& node_arg);
vaip_core::DllSafe<std::vector<std::string>> node_arg_get_denotation(const NodeArg& node_arg);
/// here, it is cheating so that even `node_arg` is a const reference,
/// it still change the shape internally. But we cannot change the
/// rank, i.e. the size of shape must be same.
void node_arg_set_shape_i64(const NodeArg& node_arg,
                            const std::vector<int64_t>& shape);
void node_arg_set_denotation(const NodeArg& node_arg,
                             const std::vector<std::string>& denotation);
void node_arg_set_element_type(NodeArg& node_arg,
                               int data_type);
const ONNX_NAMESPACE::TensorProto& node_arg_get_const_data_as_tensor(const Graph& graph,
                                                                     const NodeArg& node_arg);

}  // namespace vaip
