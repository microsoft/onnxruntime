// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef XPU_DATA_TRANSFER
#define XPU_DATA_TRANSFER

#include "core/framework/data_transfer.h"
#include "tvm_common.h"


namespace onnxruntime {
namespace tvm {

class XPUDataTransfer : public IDataTransfer {
public:
  XPUDataTransfer();
  ~XPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
  DLDevice get_context(const OrtDevice& device) const;
};

class TvmCPUDataTransfer : public IDataTransfer {
public:
  TvmCPUDataTransfer() = default;
  // Dampen MSVC warning about not fully overriding CopyTensor
  using IDataTransfer::CopyTensor;
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
};

}   // namespace tvm
}   // namespace onnxruntime

#endif // XPU_DATA_TRANSFER
