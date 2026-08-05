// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

class ScanHelper : public OrtScanKernelHelper {
 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel) noexcept;
  ScanHelper(Ort::ConstKernelInfo info, void* state);

  // Static functions assigned to the OrtScanKernelHelper fields:
  static void ORT_API_CALL ReleaseImpl(_In_ OrtScanKernelHelper* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL TransposeImpl(_In_ OrtScanKernelHelper* this_ptr,
                                               _In_reads_(num_permutation_elems) const size_t* permutation,
                                               _In_ size_t num_permutation_elems,
                                               _In_ const OrtValue* input, _In_opt_ OrtSyncStream* stream,
                                               _Inout_ OrtValue* output) noexcept;

 private:
  Ort::ConstKernelInfo info_;
  OrtDataTransferImpl* data_transfer_impl_;  // Custom state passed from OrtEp
};
