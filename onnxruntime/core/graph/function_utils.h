#pragma once
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/common/common.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/function.h"
#include "core/graph/schema_registry.h"

namespace onnxruntime {
namespace function_utils {

/** Create a OpSchema given a subgraph EP want to fuse.
 * This is used when EP return fusion in GetCapability implementation.
 * @param graph The graph which host the subgraph.
 * @param nodes_to_fuse The metadata for the subgraph that EP want to fuse.
 * @param allow_aggregated_tensor_type if true, it will use a type constraint called
 * TAggregatedTypes for all inputs and outputs,
 * and that this will match all tensor types in the all_tensor_types_ir4 list.
 */
std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const Graph& graph,
                                                       const IndexedSubGraph& nodes_to_fuse,
                                                       bool allow_aggregated_tensor_type = false);

/** Create a OpSchema given from a local function in onnx model.
 * @param function_domain The domain of the function.
 * @param function_name The name of the function.
 * @param model_local_functions The map of local functions in the same onnx model.
 *                              This will be used as context for the function's type/shape inference.
 *                              This argument is captured by shape inferencing lambda by reference and must
 *                              be alive at the time of the shape inferencing.
 * @param domain_version_map Domain to version map used in current onnx model.
 * @param schema_registry The schema registry current model is using.
 * @param logger The logger current model is using.
 * @param allow_released_opsets_only The flag whether we only enable released opset.
 */
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
inline std::string GetFunctionIdentifier(std::string_view function_domain, std::string_view function_name) {
  return function_domain.data() + std::string(":") + function_name.data();
}

void Specialize(ONNX_NAMESPACE::FunctionProto& called_function, const ONNX_NAMESPACE::NodeProto& calling_node,
                const onnxruntime::NodeAttributes& attr_map, const std::string& unique_prefix);

void Specialize(ONNX_NAMESPACE::FunctionProto& called_function, const Node& calling_node, const std::string& unique_prefix);

}  // namespace function_utils

}  // namespace onnxruntime
