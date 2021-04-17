// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_cooformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
SparseCooFormatRep::~SparseCooFormatRep() = default;

void SparseCooFormatRep::InitIndices(const TensorShape& shape) {
  auto ml_ind_type = DataTypeImpl::GetType<int64_t>();
  void* data = IndicesStart(ml_ind_type->Size());
  indices_ = Tensor(ml_ind_type, shape, data, Location());
}

Status SparseCooFormatRep::Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
                                int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {

  // We may or may not have our own allocation, so recalculate
  const int64_t required_size = RequiredAllocationSize();
  auto rep_copy = std::make_unique<SparseCooFormatRep>(Values().DataType(),
                                                       Values().Shape(),
                                                       required_size,
                                                       allocator);
  rep_copy->InitIndices(Indices().Shape());
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(Indices(), rep_copy->MutableIndices(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCooFormatRep::Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
                                int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  const int64_t required_size = RequiredAllocationSize();
  auto rep_copy = std::make_unique<SparseCooFormatRep>(Values().DataType(), Values().Shape(), required_size, allocator);
  rep_copy->InitIndices(Indices().Shape());
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(Indices(), rep_copy->MutableIndices(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

int64_t SparseCooFormatRep::RequiredAllocationSize() const noexcept {
  return CalculateRequiredBufferSize(Values().SizeInBytes(), Indices().SizeInBytes(), sizeof(int64_t));
}

Status SparseCooBuilder::Create(bool linearized, size_t non_zero, SparseCooFormatRep*& result) {
  ORT_RETURN_IF_NOT(*rep_ == nullptr, "The instance is expected to be empty");
  ORT_RETURN_IF_NOT(allocator_ != nullptr, "Must have an allocator set with Sparse Tensor instance");

  const auto nnz = static_cast<int64_t>(non_zero);
  auto ml_data_type = sp_->DataType();
  const auto values_size = nnz * ml_data_type->Size();
  const auto indices_size = (linearized) ? nnz * sizeof(int64_t) : 2 * nnz * sizeof(int64_t);
  const int64_t total_buffer_size = SparseRep::CalculateRequiredBufferSize(values_size, indices_size, sizeof(int64_t));

  TensorShape values_shape({nnz});
  auto coo_ptr = std::make_unique<SparseCooFormatRep>(ml_data_type, values_shape,
                                                      total_buffer_size, allocator_);
  if (linearized) {
    coo_ptr->InitIndices(values_shape); // Same shape as values
  } else {
    TensorShape ind_shape{nnz, 2};
    coo_ptr->InitIndices(ind_shape);
  }
  result = coo_ptr.get();
  *rep_ = std::move(coo_ptr);
  return Status::OK();
}

Status SparseCooBuilder::Create(size_t nnzero, void* values_data, 
                                const TensorShape& indices_shape,
                                void* indices_data) {
  ORT_RETURN_IF_NOT(*rep_ == nullptr, "The instance is not empty");
  ORT_RETURN_IF_NOT(allocator_ == nullptr, "Should not have an allocator set");

  const auto nnz = static_cast<int64_t>(nnzero);
  const auto num_dim = indices_shape.NumDimensions();
  ORT_RETURN_IF_NOT(num_dim == 1 || num_dim == 2, "Require indices shape to be 1-D or 2-D");
  if (num_dim == 1) {
    ORT_RETURN_IF_NOT(indices_shape.Size() == nnz, "Sparse COO 1-D indices must have the size of NNZ");
  } else {
    ORT_RETURN_IF_NOT(indices_shape.Size() == nnz * 2, "Sparse COO 2-D indices must have the size of 2 * NNZ");
  }

  auto coo_ptr = std::make_unique<SparseCooFormatRep>(sp_->DataType(), TensorShape({nnz}),
                                                      values_data, sp_->Location());
  coo_ptr->InitIndices(indices_shape, indices_data);
  *rep_ = std::move(coo_ptr);
  return Status::OK();
}

}  // namespace onnxruntime