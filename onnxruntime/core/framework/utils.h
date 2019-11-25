// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/framework_common.h"
#include "core/framework/iexecutor.h"
#include "core/framework/session_state.h"
#include "core/framework/session_options.h"

namespace ONNX_NAMESPACE {
class TensorShapeProto;
class TensorProto;
std::ostream& operator<<(std::ostream& out, const TensorShapeProto& shape_proto);
std::ostream& operator<<(std::ostream& out, const TensorProto& tensor_proto);
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
class ExecutionProviders;
struct FeedsFetchesInfo;
class FeedsFetchesManager;
struct MLValueCopyInfo;
class Graph;
class KernelDef;
class KernelRegistryManager;
class IExecutionProvider;
class Node;
class Tensor;

namespace logging {
class Logger;
}

namespace utils {
void* DefaultAlloc(size_t size);
void DefaultFree(void* p);

AllocatorPtr GetAllocator(const SessionState& session_state, const OrtMemoryInfo& memory_info);

common::Status AllocateHelper(const IExecutionProvider& execution_provider, int device_id, const Tensor& fetched_tensor,
                              OrtValue& output_mlvalue);

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info);

common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue);

// Searches the allocation plan from the session_state to find the OrtMemoryInfo for the value 'name'.
const OrtMemoryInfo& FindMemoryInfoForValue(const SessionState& session_state,
                                            const std::string& name);

// Initialize the feed and fetch copy info using session_state.
// Determines the device that each graph input that will be fed will be consumed on,
// and the device that each graph output that will be fetched will be created on.
common::Status InitializeFeedFetchCopyInfo(const SessionState& session_state,
                                           FeedsFetchesManager& feeds_fetches_manager);

// Finalize the feed and fetch copy info using session_state and the device and location information from the feeds
// and fetches that will be used in graph execution.
void FinalizeFeedFetchCopyInfo(const SessionState& session_state,
                               FeedsFetchesManager& feeds_fetches_manager,
                               const std::vector<OrtDevice>& feed_locations,
                               const std::vector<const OrtMemoryInfo*>& fetch_alloc_info);

// Execute the main graph. The feed_fetches_manager will be finalized based on the provided feeds and fetches.
common::Status ExecuteGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                            ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger);

// Execute a subgraph. The feeds_fetches_manager should have been finalized prior to calling this function.
// See IControlFlowNode::SetupSubgraphExecutionInfo usage in the control flow kernels.
common::Status ExecuteSubgraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                               const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               ExecutionMode execution_mode, const bool& terminate_flag, const logging::Logger& logger);

#if defined(DEBUG_NODE_INPUTS_OUTPUTS)
// to create a build with these enabled run the build script with 1 to dump just shapes, or 2 to dump shapes and data
// e.g.
//   --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
// To unset you'll need to either delete CMakeCache.txt or run with
//   --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=0
void DumpNodeInputs(const OpKernelContext& context, const Node& node);
void DumpNodeOutputs(OpKernelContext& context, const Node& node, const SessionState& session_state);
#endif

#define DispatchOnTensorType(tensor_type, function, ...)        \
  if (tensor_type == DataTypeImpl::GetType<float>())            \
    function<float>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<double>())      \
    function<double>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<int8_t>())      \
    function<int8_t>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<int16_t>())     \
    function<int16_t>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<int32_t>())     \
    function<int32_t>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<int64_t>())     \
    function<int64_t>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<uint8_t>())     \
    function<uint8_t>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<uint16_t>())    \
    function<uint16_t>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<uint32_t>())    \
    function<uint32_t>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<uint64_t>())    \
    function<uint64_t>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<bool>())        \
    function<bool>(__VA_ARGS__);                                \
  else if (tensor_type == DataTypeImpl::GetType<MLFloat16>())   \
    function<MLFloat16>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<BFloat16>())    \
    function<BFloat16>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<std::string>()) \
    function<std::string>(__VA_ARGS__);                         \
  else                                                          \
    ORT_ENFORCE(false, "Unknown tensor type of ", tensor_type)

#define DispatchOnTensorTypeWithReturn(tensor_type, retval, function, ...) \
  if (tensor_type == DataTypeImpl::GetType<float>())                       \
    retval = function<float>(__VA_ARGS__);                                 \
  else if (tensor_type == DataTypeImpl::GetType<double>())                 \
    retval = function<double>(__VA_ARGS__);                                \
  else if (tensor_type == DataTypeImpl::GetType<int8_t>())                 \
    retval = function<int8_t>(__VA_ARGS__);                                \
  else if (tensor_type == DataTypeImpl::GetType<int16_t>())                \
    retval = function<int16_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<int32_t>())                \
    retval = function<int32_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<int64_t>())                \
    retval = function<int64_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<uint8_t>())                \
    retval = function<uint8_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<uint16_t>())               \
    retval = function<uint16_t>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<uint32_t>())               \
    retval = function<uint32_t>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<uint64_t>())               \
    retval = function<uint64_t>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<bool>())                   \
    retval = function<bool>(__VA_ARGS__);                                  \
  else if (tensor_type == DataTypeImpl::GetType<MLFloat16>())              \
    retval = function<MLFloat16>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<BFloat16>())               \
    retval = function<BFloat16>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<std::string>())            \
    retval = function<std::string>(__VA_ARGS__);                           \
  else                                                                     \
    ORT_ENFORCE(false, "Unknown tensor type of ", tensor_type)

}  // namespace utils
}  // namespace onnxruntime
