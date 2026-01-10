// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

class LoopHelper : public OrtLoopKernelHelper {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel) noexcept;
  LoopHelper(Ort::ConstKernelInfo info, void* state);

  // Static functions assigned to the OrtLoopKernelHelper fields:
  static void ORT_API_CALL ReleaseImpl(_In_ OrtLoopKernelHelper* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL ConcatOutputImpl(
      _In_ OrtLoopKernelHelper* this_ptr,
      _In_opt_ void* stream_handle,
      _In_reads_(num_per_iteration_outputs) const OrtValue* const* per_iteration_outputs,
      _In_ size_t num_per_iteration_outputs,
      _Out_writes_bytes_all_(output_size_in_bytes) void* output,
      _In_ size_t output_size_in_bytes) noexcept;

 private:
  Ort::ConstKernelInfo info_;
  OrtDataTransferImpl* data_transfer_impl_;  // Custom state passed from OrtEp
};
