// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"

namespace onnxruntime {
/// <summary>
/// This is a representation of Coo format that is generic.
/// However, it is possible to create a representation of the same format that is
/// specific to a library being used such as cuSparse.
/// </summary>
class SparseCooFomatRep : public SparseRep {
 public:
  SparseCooFomatRep(const TensorShape& ind_shape, const AllocatorPtr& allocator) : 
    indices_(DataTypeImpl::GetType<int64_t>(),
    ind_shape,
    allocator) {
  }

  ~SparseCooFomatRep() override;

  const Tensor& Indices() const {
    return indices_;
  }

  Tensor& MutableIndices() {
    return indices_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  Status Copy(const AllocatorPtr& allocator, std::unique_ptr<SparseRep>& dst_rep) const override;

 private:
  Tensor indices_;  // may be 1-D or 2-D.
};

template <> inline
const SparseCooFomatRep* SparseTensor::GetRep<SparseCooFomatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCoo), "Expecting COO format");
  return static_cast<const SparseCooFomatRep*>(rep_.get());
}

}  // namespace onnxruntime
