// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

// Data transfer interface.
class IDataTransfer {
 public:
  virtual ~IDataTransfer() = default;

  virtual bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const = 0;

  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const = 0;

  struct SrcDstPair {
    std::reference_wrapper<const Tensor> src;
    std::reference_wrapper<Tensor> dst;
    int exec_queue_id;
  };

  // batched copy. default implementation copies each entry sequentially, and returns on first failure.
  virtual common::Status CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const;

  // If this is really a Provider_IDataTransfer, this returns true. Used to convert back & forth with providers efficiently
  virtual bool IsProviderInterface() const { return false; }
};

class CPUDataTransfer : public IDataTransfer {
 public:
  CPUDataTransfer() = default;
  // Dampen MSVC warning about not fully overriding CopyTensor
  using IDataTransfer::CopyTensor;
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
};
}  // namespace onnxruntime
