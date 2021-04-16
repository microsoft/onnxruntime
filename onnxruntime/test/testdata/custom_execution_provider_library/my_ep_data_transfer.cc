
#include "my_ep_data_transfer.h"

namespace onnxruntime {
bool MyEPDataTransfer::CanCopy(const OrtDevice& /*src_device*/, const OrtDevice& /*dst_device*/) const {
  return false;
}

common::Status MyEPDataTransfer::CopyTensor(const Tensor& /*src*/, Tensor& /*dst*/, int /*exec_queue_id*/) const {
  return Status::OK();
}

}  // namespace onnxruntime