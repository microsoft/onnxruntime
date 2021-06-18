// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

#include "core/framework/feeds_fetches_manager.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/framework/ort_value_tensor_slicer.h"

namespace onnxruntime {
namespace scan {
namespace detail {
/**
Helper struct for keeping static information about the Scan node and its subgraph.
Used to create the FeedsFetchesManager needed for efficient subgraph execution.
*/
struct Info {
  Info(const Node& node, const GraphViewer& subgraph_in, int num_scan_inputs_in, bool is_v8);

  const GraphViewer& subgraph;

  int num_inputs;
  int num_variadic_inputs;
  int num_outputs;
  int num_loop_state_variables;
  int num_scan_inputs;
  int num_scan_outputs;

  int num_implicit_inputs;

  std::vector<std::string> subgraph_input_names;
  std::vector<std::string> subgraph_output_names;
};

// helpers for handling data on a non-CPU device.
// Provide as needed when Scan is being run by a non-CPU based ExecutionProvider
struct DeviceHelpers {
  using ZeroData = std::function<common::Status(void* data, size_t size_in_bytes)>;
  using Transpose = std::function<common::Status(const std::vector<size_t>& permutations,
                                                 const Tensor& input, Tensor& output)>;
  using CreateConstSlicer = std::function<OrtValueTensorSlicer<const OrtValue>(const OrtValue& ort_value,
                                                                               int64_t slice_dimension /*=0*/,
                                                                               int64_t dim0_offset /*=0*/)>;
  using CreateMutableSlicer = std::function<OrtValueTensorSlicer<OrtValue>(OrtValue& ort_value,
                                                                           int64_t slice_dimension /*=0*/,
                                                                           int64_t dim0_offset /*=0*/)>;
  // Scan 8 may need to zero out unused output data for short sequences
  ZeroData set_data_to_zero_func;

  // Scan 9 may need to transpose input or output data on a non-CPU device
  Transpose transpose_func;

  // Custom logic may be required to slice a tensor on a non-CPU device if the data pointer in an OrtValue
  // can not be validly incremented using pointer arithmetic
  // (e.g. it is a pointer to a handle rather than the actual data)
  CreateConstSlicer create_const_slicer_func = OrtValueTensorSlicer<const OrtValue>::Create;
  CreateMutableSlicer create_mutable_slicer_func = OrtValueTensorSlicer<OrtValue>::Create;
};
}  // namespace detail
}  // namespace scan

template <int OpSet>
class Scan : public controlflow::IControlFlowKernel {
 public:
  Scan(const OpKernelInfo& info) : IControlFlowKernel(info) { Init(info); }
  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

  struct Info : scan::detail::Info {
    Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in, int num_scan_inputs_in)
        : scan::detail::Info(node, subgraph_in, num_scan_inputs_in, /* is_v8 */ OpSet == 8) {}
  };

  void SetDeviceHelpers(const scan::detail::DeviceHelpers& device_helpers) {
    device_helpers_ = device_helpers;  // copy
  }

 private:
  int64_t num_scan_inputs_;
  std::vector<int64_t> input_directions_;
  std::vector<int64_t> output_directions_;
  std::vector<int64_t> input_axes_;
  std::vector<int64_t> output_axes_;

  // Info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<Info> info_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;

  scan::detail::DeviceHelpers device_helpers_;
};
}  // namespace onnxruntime
