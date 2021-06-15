// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"

namespace onnxruntime {
/// <summary>
/// This is a representation of Coo format.
/// </summary>
class SparseCooFormatRep : public SparseRep {
 public:
  SparseCooFormatRep(MLDataType data_type, const TensorShape& values_shape,
                     int64_t total_buffer_size, AllocatorPtr allocator)
      : SparseRep(data_type, values_shape, total_buffer_size, std::move(allocator)),
        indices_() {
  }

  SparseCooFormatRep(MLDataType data_type, const TensorShape& values_shape, void* values_data,
                     const OrtMemoryInfo& location)
      : SparseRep(data_type, values_shape, values_data, location),
        indices_() {
  }

  ~SparseCooFormatRep() override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SparseCooFormatRep);

  /// <summary>
  /// Initialize indices to an offset in the pre-allocated buffer
  /// </summary>
  /// <param name="shape"></param>
  void InitIndices(const TensorShape& shape);

  /// <summary>
  /// Initializer indices with user data ptr
  /// </summary>
  /// <param name="ind_shape">shape of the indices</param>
  /// <param name="data">may point to either a user provided
  /// or a pre-allocated buffer</param>
  void InitIndices(const TensorShape& shape, void* data) {
    indices_ = Tensor(DataTypeImpl::GetType<int64_t>(), shape, data, Location());
  }

  const Tensor& Indices() const noexcept {
    return indices_;
  }

  Tensor& MutableIndices() noexcept {
    return indices_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  int64_t RequiredAllocationSize() const noexcept override;

 private:
  Tensor indices_;  // may be 1-D or 2-D.
};

/// <summary>
/// Class returns rep specific interface to properly construct a given
/// sparse format
/// </summary>
class SparseCooBuilder {
  AllocatorPtr allocator_;
  SparseTensor* sp_;
  std::unique_ptr<SparseRep>* rep_;

 public:
  SparseCooBuilder(AllocatorPtr allocator, SparseTensor& sp, std::unique_ptr<SparseRep>& rep) noexcept
      : allocator_(std::move(allocator)),
        sp_(&sp),
        rep_(&rep) {}

  /// <summary>
  /// Creates a COO format representation that owns the a buffer
  /// and dense shape dimensions.
  /// </summary>
  /// <param name="linearized">true if indices have 1-D linearized format and have a size of nnz() or
  /// 2-D which is a coordinate format. The latter would have a length of 2 * nnz</param>
  /// <param name="nnz">number of non-zero values</param>
  /// <returns>Created representation</returns>
  Status Create(bool linearized, size_t nnz, SparseCooFormatRep*&);

  /// <summary>
  /// Create a COO representation that does not own the data. Use for inputs/outputs
  /// The builder is going to use the same OrtMemoryInfo as for values
  /// </summary>
  /// <param name="nnz">number of non-zero elements</param>
  /// <param name="values_data">pointer to indices data</param>
  /// <param name="indices_shape">1-D or 2-D indices shape</param>
  /// <param name="indices_data">pointer to indices data</param>
  /// <returns>Status</returns>
  Status Create(size_t nnz, void* values_data, const TensorShape& indices_shape, void* indices_data);
};

template <>
inline const SparseCooFormatRep* SparseTensor::GetRep<SparseCooFormatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCoo), "Expecting COO format");
  return static_cast<const SparseCooFormatRep*>(rep_.get());
}

template <>
inline SparseCooBuilder SparseTensor::RepBuilder<SparseCooBuilder>() {
  if (!rep_) {
    format_flags_ = Set(format_flags_, SparseFormatFlags::kCoo);
  } else {
    ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCoo), "Expecting COO format set");
  }
  return SparseCooBuilder(allocator_, *this, rep_);
}

}  // namespace onnxruntime
