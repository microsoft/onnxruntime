#pragma once

#include "core/framework/data_transfer.h"

namespace onnxruntime {

class CustomDataTransfer : public IDataTransfer {
 public:
  CustomDataTransfer() {}
  ~CustomDataTransfer() {}

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
};

}  // namespace onnxruntime
