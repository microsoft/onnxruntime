// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <vector>
#include "core/common/common.h"

struct OrtDevice;

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
#if !defined(DISABLE_SPARSE_TENSORS)
class SparseTensor;
#endif
#endif
class Stream;

namespace common {
class Status;
}

// Data transfer interface.
class IDataTransfer {
 public:
  virtual ~IDataTransfer() = default;

  virtual bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const = 0;

  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst) const;

  virtual common::Status CopyTensorAsync(const Tensor& /*src*/, Tensor& /*dst*/, Stream& /*stream*/) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  struct SrcDstPair {
    std::reference_wrapper<const Tensor> src;
    std::reference_wrapper<Tensor> dst;
    Stream* src_stream;  // producer stream of src
  };

  // batched copy. default implementation copies each entry sequentially, and returns on first failure.
  virtual common::Status CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const;

#if !defined(DISABLE_SPARSE_TENSORS)
  struct SparseSrcDstPair {
    std::reference_wrapper<const SparseTensor> src;
    std::reference_wrapper<SparseTensor> dst;
    int exec_queue_id;
  };

  virtual common::Status CopySparseTensors(const std::vector<SparseSrcDstPair>& src_dst_pairs) const;
#endif
};

class CPUDataTransfer : public IDataTransfer {
 public:
  CPUDataTransfer() = default;
  // Dampen MSVC warning about not fully overriding CopyTensor
  using IDataTransfer::CopyTensor;
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
  // This is needed for calling IDataTransfer::CopyTensors
  // with CPU and GPU tensors. The reason is that IDataTransfer::CopyTensors
  // always calls CopyTensorAsync when a stream is provided.
  using IDataTransfer::CopyTensorAsync;
  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& /*stream*/) const override;
};
}  // namespace onnxruntime
