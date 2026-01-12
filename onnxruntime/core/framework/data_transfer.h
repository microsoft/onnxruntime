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

  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst, size_t src_offset, size_t dst_offset, size_t size) const;

  virtual common::Status CopyTensorAsync(const Tensor& /*src*/, Tensor& /*dst*/, Stream& /*stream*/) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual common::Status CopyTensorAsync(const Tensor& /*src*/, Tensor& /*dst*/, size_t /*src_offset*/, size_t /*dst_offset*/, size_t /*size*/, Stream& /*stream*/) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  struct SrcDstPair {
    std::reference_wrapper<const Tensor> src;
    std::reference_wrapper<Tensor> dst;
    Stream* src_stream;             // producer stream of src
    size_t source_offset = 0;       // offset in source tensor (in bytes)
    size_t destination_offset = 0;  // offset in destination tensor (in bytes)
    size_t size = 0;                // number of bytes to copy (0 means copy entire tensor)
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
  common::Status CopyTensor(const Tensor& src, Tensor& dst, size_t src_offset, size_t dst_offset, size_t size) const override;
};
}  // namespace onnxruntime
