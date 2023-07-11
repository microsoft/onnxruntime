#include "custom_data_transfer.h"

namespace onnxruntime {
bool CustomDataTransfer::CanCopy(const OrtDevice& /*src_device*/, const OrtDevice& /*dst_device*/) const {
  return false;
}

common::Status CustomDataTransfer::CopyTensor(const Tensor& /*src*/, Tensor& /*dst*/) const {
  return Status::OK();
}

}  // namespace onnxruntime
