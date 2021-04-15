
#pragma once

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

class MyEPDataTransfer : public IDataTransfer {
 public:
  MyEPDataTransfer() {}
  ~MyEPDataTransfer() {}

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

};

}  // namespace onnxruntime