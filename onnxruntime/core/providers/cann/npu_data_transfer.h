// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cann_inc.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

enum CANNStreamType : int {
  kCannStreamDefault = 0,
  kCannStreamCopyIn,
  kCannStreamCopyOut,
  kTotalCannStreams,
};

class NPUDataTransfer : public IDataTransfer {
 public:
  explicit NPUDataTransfer(aclrtStream stream, bool do_copy_in_default_stream = true);
  ~NPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

  aclrtStream GetStream(int queue_id) const {
    ORT_ENFORCE(queue_id >= 0 && queue_id < kTotalCannStreams);
    return streams_[queue_id];
  }

 private:
  bool do_copy_in_default_stream_;
  aclrtStream streams_[kTotalCannStreams];
};

}  // namespace onnxruntime
