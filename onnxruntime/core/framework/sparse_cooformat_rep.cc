// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "sparse_cooformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
Status SparseCooFomatRep::Copy(const DataTransferManager& data_transfer_manager, AllocatorPtr allocator,
                               int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  Tensor indices(DataTypeImpl::GetType<int64_t>(), indices_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(indices_, indices, exec_q_id));
  std::unique_ptr<SparseCooFomatRep> rep_copy = make_unique<SparseCooFomatRep>();
  rep_copy->indices_ = std::move(indices);
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

}  // namespace onnxruntime