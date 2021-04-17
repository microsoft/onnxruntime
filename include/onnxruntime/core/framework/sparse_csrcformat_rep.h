// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"

namespace onnxruntime {
/// <summary>
/// This class represents Compressed Storage Row(Column) sparse format
/// CSS for Row Major Matrices
/// CSC for Column Major Matrices
/// </summary>
class SparseCsrcFormatRep : public SparseRep {
 public:
  enum Order {
    kRowMajor = 1,
    kColMajor = 2
  };

  /// <summary>
  /// Constructor that allocates memory for values and indices
  /// The indices must be inited separately
  /// </summary>
  /// <param name="data_type"></param>
  /// <param name="values_shape"></param>
  /// <param name="total_allocation_size"></param>
  /// <param name="allocator"></param>
  /// <param name="order"></param>
  SparseCsrcFormatRep(const MLDataType data_type, const TensorShape& values_shape, 
                      int64_t total_allocation_size, const AllocatorPtr& allocator, Order order)
      : SparseRep(data_type, values_shape, total_allocation_size, allocator),
        major_(order),
        inner_indices_(),
        outer_indices_() {}
        //inner_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(inner_size)}, allocator),
        //outer_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(outer_size)}, allocator) {}

  /// <summary>
  /// Constructor that allocates no memory but makes data point to the user provided buffer
  /// </summary>
  /// <param name="data_type"></param>
  /// <param name="values_shape"></param>
  /// <param name="values_data"></param>
  /// <param name="location"></param>
  /// <param name="order"></param>
  SparseCsrcFormatRep(MLDataType data_type, const TensorShape& values_shape, void* values_data,
                      const OrtMemoryInfo& location, Order order)
      : SparseRep(data_type, values_shape, values_data, location),
        major_(order),
        inner_indices_(),
        outer_indices_() {}

        //inner_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(inner_size)}, inner_data, info, 0),
        //outer_indecies_(DataTypeImpl::GetType<int64_t>(), {static_cast<int64_t>(outer_size)}, outer_data, info, 0) {}

  ~SparseCsrcFormatRep() override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SparseCsrcFormatRep);

  void InitIndices(const TensorShape& inner_shape, void* inner_data,
                   const TensorShape& outer_shape, void* outer_data);

  void InitIndices(const TensorShape& inner_shape, const TensorShape& outer_shape);

  /// <summary>
  /// Returns the matrix order currently represented
  /// </summary>
  /// <returns></returns>
  Order Major() const noexcept {
    return major_;
  }

  const Tensor& Inner() const noexcept {
    return inner_indices_;
  }

  const Tensor& Outer() const noexcept {
    return outer_indices_;
  }

  Tensor& MutableInner() noexcept {
    return inner_indices_;
  }

  Tensor& MutableOuter() noexcept {
    return outer_indices_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  int64_t RequiredAllocationSize() const noexcept override;

 private:
  Order major_;
  // This represents indices of nnz within a row/column
  Tensor inner_indices_;
  // This represents indices into values and inner_indecies where each row/column data starts
  Tensor outer_indices_;
};

class SparseCsrcBuilder {
 public:
  SparseCsrcBuilder(AllocatorPtr allocator, SparseTensor& sp, std::unique_ptr<SparseRep>& rep) noexcept
      : allocator_(std::move(allocator)),
        sp_(&sp),
        rep_(&rep) {}

  ~SparseCsrcBuilder() = default;

  Status Create(SparseCsrcFormatRep::Order major, size_t nnz,
                     size_t inner_size, size_t outer_size,
                     SparseCsrcFormatRep*& result);

  Status Create(SparseCsrcFormatRep::Order major,
                     size_t nnz, size_t inner_size, size_t outer_size,
                     void* values_data, int64_t* inner_data, int64_t* outer_data);

 private:
  AllocatorPtr allocator_;
  SparseTensor* sp_;
  std::unique_ptr<SparseRep>* rep_;
};

template <>
inline const SparseCsrcFormatRep* SparseTensor::GetRep<SparseCsrcFormatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCsrc), "Expecting CSR(C) format");
  return static_cast<const SparseCsrcFormatRep*>(rep_.get());
}

template <>
inline SparseCsrcBuilder SparseTensor::RepBuilder<SparseCsrcBuilder>() {
  if (!rep_) {
    format_flags_ = Set(format_flags_, SparseFormatFlags::kCsrc);
  } else {
    ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCsrc), "Expecting CSR(C) format set");
  }
  return SparseCsrcBuilder(allocator_, *this, rep_);
}

}  // namespace onnxruntime
