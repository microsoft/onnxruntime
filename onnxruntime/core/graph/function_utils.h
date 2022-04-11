#pragma once
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "onnx/onnx_pb.h"
#include "core/graph/graph.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/function.h"
#include "core/graph/schema_registry.h"

namespace onnxruntime {
namespace function_utils {

std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const Graph& graph,
                                                       const IndexedSubGraph& nodes_to_fuse);

std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const std::string& function_domain,
                                                       const std::string& function_name,
                                                       const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
                                                       const std::unordered_map<std::string, int>& domain_version_map,
                                                       const SchemaRegistryManager& schema_registry,
                                                       const logging::Logger& logger,
                                                       bool allow_released_opsets_only);

/** Get the unique id for function. This is used as a key to find the 
* relevant model local function from it's container.
* @param function_domain Domain for the function.
* @param function_name Name of the function. Name should match the OpType of the node which references the function.
*/
inline std::string GetFunctionIdentifier(const std::string& function_domain, const std::string& function_name) {
  return function_domain + ":" + function_name;
}

std::unique_ptr<Function> Instantiate(const onnxruntime::Graph& graph,
                                      const onnxruntime::NodeIndex& node_index,
                                      const ONNX_NAMESPACE::FunctionProto& onnx_func_proto,
                                      const logging::Logger& logger);

}

}  // namespace onnxruntime
