// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sparse_cooformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
SparseCooFomatRep ::~SparseCooFomatRep() = default;

Status SparseCooFomatRep::Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
                               int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = make_unique<SparseCooFomatRep>(indices_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(indices_, rep_copy->indices_, exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCooFomatRep::Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator, int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = make_unique<SparseCooFomatRep>(indices_.Shape(), allocator);
  data_transfer.CopyTensor(rep_copy->Indices(), rep_copy->MutableIndices(), exec_q_id);
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

template <>
SparseCooFomatRep* SparseTensor::MutableRep<SparseCooFomatRep>() {
  if (IsSet(format_flags_, SparseFormatFlags::kCoo)) {
    return static_cast<SparseCooFomatRep*>(rep_.get());
  }

  SparseCooFomatRep* result = nullptr;
  if (format_flags_ == SparseFormatFlags::kUndefined) {
    result = new SparseCooFomatRep({values_.Shape().Size(), static_cast<int64_t>(values_.Shape().NumDimensions())}, allocator_);
    rep_.reset(result);
    format_flags_ = Set(format_flags_, SparseFormatFlags::kCoo);
  } else {
    ORT_ENFORCE(false, "Wrong format requested for modification");
  }
  return result;
}

}  // namespace onnxruntime