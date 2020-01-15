// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/runtime/control_flow/loop_exec_ctx.h"

namespace onnxruntime {
namespace nuphar {

// Note ScanExecInfo have all ort related meta data
struct ScanExecInfo : ControlFlowInfo {
  std::vector<bool> scan_input_forwards;
  std::vector<bool> scan_output_forwards;
  std::vector<int64_t> scan_input_axes;
  std::vector<int64_t> scan_output_axes;

  int64_t num_state_variables;
  int64_t num_scan_inputs;
  int64_t num_scan_outputs;
  int64_t num_scan_implicit_inputs;

  std::vector<int> state_to_output_indices;

  ScanExecInfo() : ControlFlowInfo(ControlFlowInfoType::Scan) {}

  DYN_PROMOTE_DERIVED(ControlFlowInfo, ControlFlowInfoType, Scan)
};

class ScanExecCtx final : public LoopExecCtx {
 public:
  ScanExecCtx() : seq_length_(0) {
  }

  void InitContext(KernelComputeCtx* compute_ctx,
                   const NupharFuncInfo* func_info) override;
  void UpdateContext(KernelComputeCtx* compute_ctx,
                     const NupharFuncInfo* func_info) override;
  void InitIteration(KernelComputeCtx* compute_ctx,
                     const NupharFuncInfo* func_info) override;

  void LoopFinalizer() override;
  void Advance(const ControlFlowInfo* cf_info) override;

 private:
  // Current input/output holds the current ptr of Scan, not PackedFunc
  std::vector<void*> current_input_ptrs_;
  std::vector<void*> current_output_ptrs_;
  std::vector<void*> current_ort_state_input_ptrs_;
  std::vector<void*> current_ort_state_output_ptrs_;

  std::vector<int64_t> input_strides_;
  std::vector<int64_t> output_strides_;

  // shapes for scan inputs inside subgraph, as ORT shape would have scan_axis
  // it also serves as ptr in DLTensor's shape for input args
  std::vector<std::vector<int64_t>> scan_input_in_subgraph_shapes_;

  // shapes for scan outputs outside subgraph
  // it is used by ort to create output tensor
  std::vector<std::vector<int64_t>> scan_output_shapes_;

  // the ptr holds func's input and output
  // func outputs are more irregular due to alias.
  // Therefore, we use double ptr to hold the ptr
  // to make code more readable
  std::vector<void**> current_func_output_ptrs_;

  // allocated state buffers
  // This is unqiue_ptr from Ort, and will be freed after this class is free
  std::vector<IAllocatorUniquePtr<void>> ort_state_buffer_unique_ptrs_;

  // state buffers (raw ptrs)
  // The raw pointer of the above.
  // These two the one we common use for address calculation
  std::vector<void*> ort_state_input_buffers_;
  std::vector<void*> ort_state_output_buffers_;

  std::vector<std::size_t> state_bytes_size_;

  int64_t seq_length_;

  ;
};

}  // namespace nuphar
}  // namespace onnxruntime
