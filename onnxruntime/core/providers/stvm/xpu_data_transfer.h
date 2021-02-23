// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef XPU_DATA_TRANSFER
#define XPU_DATA_TRANSFER

#include "core/framework/data_transfer.h"
#include "stvm_common.h"

namespace onnxruntime {

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer();
  ~GPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
  DLDevice get_context(const OrtDevice& device) const;
};

}  // namespace onnxruntime
#endif // XPU_DATA_TRANSFER
