// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
class Node;
class Graph;
}  // namespace onnxruntime

namespace onnxruntime {
class ExecutionProviders;
class KernelDef;
class KernelRegistryManager;
class SessionState;

namespace logging {
class Logger;
}

namespace utils {
const KernelDef* GetKernelDef(const KernelRegistryManager& kernel_registry,
                              const onnxruntime::Node& node);

const KernelDef* GetKernelDef(const onnxruntime::Graph& graph,
                              const KernelRegistryManager& kernel_registry,
                              const onnxruntime::NodeIndex node_id);

AllocatorPtr GetAllocator(const ExecutionProviders& exec_providers, const OrtAllocatorInfo& allocator_info);

AllocatorPtr GetAllocator(const SessionState& session_state,
                          const OrtAllocatorInfo& allocator_info);

#define DispatchOnTensorType(tensor_type, function, ...)      \
  if (tensor_type == DataTypeImpl::GetType<float>())          \
    function<float>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<double>())    \
    function<double>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<int8_t>())    \
    function<int8_t>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<int16_t>())   \
    function<int16_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<int32_t>())   \
    function<int32_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<int64_t>())   \
    function<int64_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<uint8_t>())   \
    function<uint8_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<uint16_t>())  \
    function<uint16_t>(__VA_ARGS__);                          \
  else if (tensor_type == DataTypeImpl::GetType<uint32_t>())  \
    function<uint32_t>(__VA_ARGS__);                          \
  else if (tensor_type == DataTypeImpl::GetType<uint64_t>())  \
    function<uint64_t>(__VA_ARGS__);                          \
  else if (tensor_type == DataTypeImpl::GetType<bool>())      \
    function<bool>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<MLFloat16>()) \
    function<MLFloat16>(__VA_ARGS__);                         \
  else if (tensor_type == DataTypeImpl::GetType<BFloat16>())  \
  function<BFloat16>(__VA_ARGS__)

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
  retval = function<BFloat16>(__VA_ARGS__)

}  // namespace utils
}  // namespace onnxruntime
