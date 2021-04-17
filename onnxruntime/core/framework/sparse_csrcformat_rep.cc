// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_csrcformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {

SparseCsrcFormatRep::~SparseCsrcFormatRep() = default;

void SparseCsrcFormatRep::InitIndices(const TensorShape& inner_shape, void* inner_data,
                 const TensorShape& outer_shape, void* outer_data) {
  auto ml_type = DataTypeImpl::GetType<int64_t>();
  inner_indices_ = Tensor(ml_type, inner_shape, inner_data, Location());
  outer_indices_ = Tensor(ml_type, outer_shape, outer_data, Location());
}

void SparseCsrcFormatRep::InitIndices(const TensorShape& inner_shape, const TensorShape& outer_shape) {
  auto ml_ind_type = DataTypeImpl::GetType<int64_t>();
  void* inner_data = IndicesStart(ml_ind_type->Size());
  const auto inner_indices_size = inner_shape.Size() * ml_ind_type->Size();
  void* outer_data = reinterpret_cast<uint8_t*>(inner_data) + inner_indices_size;
  InitIndices(inner_shape, inner_data, outer_shape, outer_data);
}

Status SparseCsrcFormatRep::Copy(const DataTransferManager& data_transfer_manager,
                                 const AllocatorPtr& allocator,
                                 int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  const int64_t required_size = RequiredAllocationSize();
  auto rep_copy = std::make_unique<SparseCsrcFormatRep>(Values().DataType(),
                                                        Values().Shape(),
                                                        required_size,
                                                        allocator, Major());
  rep_copy->InitIndices(Inner().Shape(), Outer().Shape());
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(Inner(), rep_copy->MutableInner(), exec_q_id));
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(Outer(), rep_copy->MutableOuter(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCsrcFormatRep::Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
                                 int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  const int64_t required_size = RequiredAllocationSize();

  auto rep_copy = std::make_unique<SparseCsrcFormatRep>(Values().DataType(),
                                                        Values().Shape(),
                                                        required_size,
                                                        allocator, Major());
  rep_copy->InitIndices(Inner().Shape(), Outer().Shape());
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(Inner(), rep_copy->MutableInner(), exec_q_id));
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(Outer(), rep_copy->MutableOuter(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

int64_t SparseCsrcFormatRep::RequiredAllocationSize() const noexcept {
  return CalculateRequiredBufferSize(Values().SizeInBytes(),
                              Inner().SizeInBytes() + Outer().SizeInBytes(),
                              sizeof(int64_t));
}

Status SparseCsrcBuilder::Create(SparseCsrcFormatRep::Order major,
                                 size_t nnz, size_t inner_size, size_t outer_size,
                                 SparseCsrcFormatRep*& result) {

  ORT_RETURN_IF_NOT(*rep_ == nullptr, "The instance is not empty");
  ORT_RETURN_IF_NOT(allocator_ != nullptr, "Must have an allocator set with SparseTensor instance");
  auto ml_type = sp_->DataType();
  const int64_t values_size = static_cast<int64_t>(ml_type->Size() * nnz);
  const int64_t total_indices_size = static_cast<int64_t>((inner_size + outer_size) * sizeof(int64_t));
  const auto required_allocation_size = SparseRep::CalculateRequiredBufferSize(values_size, total_indices_size, sizeof(int64_t));

  TensorShape values_shape{static_cast<int64_t>(nnz)};
  TensorShape inner_shape{static_cast<int64_t>(inner_size)};
  TensorShape outer_shape{static_cast<int64_t>(outer_size)};

  auto csr_rep = std::make_unique<SparseCsrcFormatRep>(ml_type, values_shape, required_allocation_size, allocator_, major);
  csr_rep->InitIndices(inner_shape, outer_shape);
  result = csr_rep.get();
  *rep_ = std::move(csr_rep);
  return Status::OK();
}

Status SparseCsrcBuilder::Create(SparseCsrcFormatRep::Order major, 
                                 size_t nnz, size_t inner_size, size_t outer_size,
                                 void* values_data, int64_t* inner_data, int64_t* outer_data) {
  ORT_RETURN_IF_NOT(*rep_ == nullptr, "The instance is not empty");
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Must have NOT an allocator set with Sparse Tensor instance");

  auto ml_type = sp_->DataType();
  TensorShape values_shape{static_cast<int64_t>(nnz)};
  TensorShape inner_shape{static_cast<int64_t>(inner_size)};
  TensorShape outer_shape{static_cast<int64_t>(outer_size)};

  auto csr_rep = std::make_unique<SparseCsrcFormatRep>(ml_type, values_shape, values_data, sp_->Location(), major);
  csr_rep->InitIndices(inner_shape, inner_data, outer_shape, outer_data);
  *rep_ = std::move(csr_rep);
  return Status::OK();
}

}  // namespace onnxruntime