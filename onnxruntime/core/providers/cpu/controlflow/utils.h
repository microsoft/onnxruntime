// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "core/framework/feeds_fetches_manager.h"
#ifdef SHARED_PROVIDER
#include "core/framework/ort_value.h"
#else
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

// Creates a scalar MLValue based on given value and allocator.
template <typename T>
OrtValue MakeScalarMLValue(const AllocatorPtr& allocator, T value, bool is_1d) {
  std::vector<int64_t> dims;
  if (is_1d) {
    dims.push_back(1);
  }
  TensorShape shape(std::move(dims));
  auto* data_type = DataTypeImpl::GetType<T>();
  OrtValue ort_value;
  Tensor::InitOrtValue(data_type, shape, allocator, ort_value);
  *ort_value.GetMutable<Tensor>()->MutableData<T>() = value;
  return ort_value;
}

namespace controlflow {

/** Interface for control flow kernels    */
class IControlFlowKernel : public OpKernel {
 public:
  explicit IControlFlowKernel(const OpKernelInfo& info) : OpKernel(info) {}
  /** Setup information that is re-used each time to execute the subgraph.
  @param session_state SessionState for graph containing the control flow node
  @param attribute_name Control flow node's attribute name that contained the subgraph 
  @param subgraph_session_state SessionState for the subgraph
  */
  virtual common::Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                                    const std::string& attribute_name,
                                                    const SessionState& subgraph_session_state) = 0;
};

namespace detail {

// Searches the allocation plan from the session_state to find the OrtDevice each value in 'names' is located on,
// and updates the entry in devices with the same index.
// Resizes 'devices' if needed, defaulting to CPU as the OrtDevice.
// Use 'start_at' to skip entries in 'names' and 'devices'.
common::Status FindDevicesForValues(const SessionState& session_state,
                                    const std::vector<std::string>& names,
                                    std::vector<OrtDevice>& devices,
                                    size_t start_at = 0);

}  // namespace detail
}  // namespace controlflow
}  // namespace onnxruntime
