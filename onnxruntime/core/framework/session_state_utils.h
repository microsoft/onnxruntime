// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/common/const_pointer_container.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_allocator.h"
#include "core/framework/session_options.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/platform/path_lib.h"

namespace onnxruntime {
class Env;
class KernelRegistryManager;
class Node;
class SessionState;
class GraphViewer;
class OrtValueNameIdxMap;
class DataTransferManager;
class ExternalDataLoaderManager;
class NodeArg;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
class MemoryInfo;
#endif

namespace logging {
class Logger;
}

namespace session_state_utils {
using SaveTensorFunction = std::function<Status(const std::string& name, int idx, const OrtValue& value,
                                                const OrtCallback& d, bool constant, bool sparse)>;
using MemoryProfileFunction = std::function<void(ITensorAllocator& planner)>;

common::Status SaveInitializedTensors(
    const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
    const GraphViewer& graph, const AllocatorPtr& default_cpu_memory_info,
    const OrtValueNameIdxMap& ort_value_name_idx_map, const std::vector<OrtValueIndex>& initializer_allocation_order,
    ITensorAllocator& planner,
    const SaveTensorFunction& save_tensor_func,
    const logging::Logger& logger,
    const DataTransferManager& data_transfer_mgr,
    const ExternalDataLoaderManager& external_data_loader_mgr,
    const ExecutionPlanBase& exec_plan,
    const SessionOptions& session_options,
    const MemoryProfileFunction& memory_profile_func,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& buffered_tensors);

common::Status AllocateTensor(
    const onnxruntime::MemBuffer* m,
    std::unique_ptr<onnxruntime::Tensor>& p_tensor,
    const onnxruntime::DataTypeImpl* const& type,
    onnxruntime::TensorShape& tensor_shape,
    bool use_device_allocator_for_initializers,
    const onnxruntime::AllocatorPtr& alloc);

common::Status AllocateTensorOnDeviceOrMemory(
    bool use_device_allocator_for_initializers,
    onnxruntime::TensorShape& tensor_shape,
    const onnxruntime::DataTypeImpl* const& type,
    const onnxruntime::AllocatorPtr& alloc,
    std::unique_ptr<onnxruntime::Tensor>& p_tensor);

common::Status CopyTensorFromCPUToDevice(
    const onnxruntime::DataTransferManager& data_transfer_mgr,
    std::unique_ptr<onnxruntime::Tensor>& p_deserialize_tensor,
    std::unique_ptr<onnxruntime::Tensor>& p_tensor,
    OrtValue& ort_value);

common::Status SaveInputOutputNamesToNodeMapping(const GraphViewer& graph,
                                                 SessionState& session_state,
                                                 gsl::span<const NodeArg* const> implicit_inputs);
}  // namespace session_state_utils
}  // namespace onnxruntime
