// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"
#include "core/common/optional.h"

namespace onnxruntime {
/// <summary>
/// This is a representation of Coo format that is generic.
/// However, it is possible to create a representation of the same format that is
/// specific to a library being used such as cuSparse.
/// </summary>
class SparseCooFomatRep : public SparseRep {
 public:
  SparseCooFomatRep() {
  }

  ~SparseCooFomatRep() = default;

  const Tensor& Indicies() const {
    return indices_;
  }

  Tensor& MutableIndicies() {
    indices_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, AllocatorPtr allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

 private:
  Tensor indices_;  // may be 1-D or 2-D.
};

template <>
const SparseCooFomatRep* SparseTensor::GetRep<SparseCooFomatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCoo), "Expecting COO format");
  return static_cast<const SparseCooFomatRep*>(rep_.get());
}

template <>
SparseCooFomatRep* SparseTensor::MutableRep<SparseCooFomatRep>() {
  // There might different representations for the same format such as cuSparse
  if (IsSet(format_flags_, SparseFormatFlags::kCoo) && IsSet(format_flags_, SparseFormatFlags::kCooRawBuffer)) {
    return static_cast<SparseCooFomatRep*>(rep_.get());
  } else if (format_flags_ == SparseFormatFlags::kUndefined) {
    rep_.reset(new SparseCooFomatRep());
    Set(format_flags_, SparseFormatFlags::kCoo | SparseFormatFlags::kCooRawBuffer);
  } else {
    ORT_ENFORCE(false, "Wrong format requested for modification");
  }
  return nullptr;
}
}  // namespace onnxruntime
