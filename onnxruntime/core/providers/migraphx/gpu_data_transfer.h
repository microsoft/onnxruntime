// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "migraphx_inc.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

enum HIPStreamType : int {
  kHipStreamDefault = 0,
  kHipStreamCopyIn,
  kHipStreamCopyOut,
  kTotalHipStreams,
};

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer();
  ~GPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

  hipStream_t GetStream(int queue_id) const {
    ORT_ENFORCE(queue_id >= 0 && queue_id < kTotalHipStreams);
    return streams_[queue_id];
  }

 private:
  hipStream_t streams_[kTotalHipStreams];
};

}  // namespace onnxruntime
