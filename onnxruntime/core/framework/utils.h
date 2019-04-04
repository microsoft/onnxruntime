// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/framework_common.h"
#include "core/framework/iexecutor.h"
#include "core/framework/session_state.h"

namespace onnxruntime {
class ExecutionProviders;
class FeedsFetchesManager;
class Graph;
class KernelDef;
class KernelRegistryManager;
class IExecutionProvider;
class MLValue;
class Node;
class Tensor;

namespace logging {
class Logger;
}

namespace utils {

AllocatorPtr GetAllocator(const ExecutionProviders& exec_providers, const OrtAllocatorInfo& allocator_info);

AllocatorPtr GetAllocator(const SessionState& session_state, const OrtAllocatorInfo& allocator_info);

common::Status AllocateHelper(const IExecutionProvider& execution_provider,
                              int device_id,
                              const Tensor& fetched_tensor,
                              MLValue& output_mlvalue);

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info);

common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue);

// ExecuteGraph, writing cache info to FeedsFetchesManager to optimize feed and fetch usage across invocations when the
// order and location of the feeds and fetches is unchanged.
common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<MLValue>& feeds,
                            std::vector<MLValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger,
                            bool cache_copy_info = true);

// ExecuteGraph used the cached information in feeds_fetches_manager.
common::Status ExecuteGraphWithCachedInfo(const SessionState& session_state,
                                          const FeedsFetchesManager& feeds_fetches_manager,
                                          const std::vector<MLValue>& feeds,
                                          std::vector<MLValue>& fetches,
                                          const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                          bool sequential_execution,
                                          const bool& terminate_flag,
                                          const logging::Logger& logger);

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
