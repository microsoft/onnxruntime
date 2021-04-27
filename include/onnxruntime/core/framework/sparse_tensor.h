// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_types.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class IDataTransfer;
class DataTransferManager;

/**
 * @brief This is a Sparse Format enumeration representing bitflags
 * 
 * 
 */
enum class SparseFormatFlags : uint32_t {
  kUndefined = 0x0U,  // For completeness
  kCoo = 0x1U,        // 1-D or 2-D indices
  // kCsr = 0x1U << 1, // This may be represented in a variety of ways
  // kBlockedEll = 0x1U << 2, // NVIDIA Blocked Ell
  kBlockSparse = 0x1U << 3  // as in GPT-3
};

inline SparseFormatFlags operator|(SparseFormatFlags flags, SparseFormatFlags flag) {
  return static_cast<SparseFormatFlags>(static_cast<uint32_t>(flags) | static_cast<uint32_t>(flag));
}

inline SparseFormatFlags operator&(SparseFormatFlags flags, SparseFormatFlags flag) {
  return static_cast<SparseFormatFlags>(static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag));
}

inline SparseFormatFlags TurnOff(SparseFormatFlags flags, SparseFormatFlags flag) {
  return static_cast<SparseFormatFlags>(static_cast<uint32_t>(flags) & ~static_cast<uint32_t>(flag));
}

inline bool IsSet(SparseFormatFlags flags, SparseFormatFlags flag) {
  return SparseFormatFlags::kUndefined != (flags & flag);
}

inline SparseFormatFlags Set(SparseFormatFlags flags, SparseFormatFlags flag) {
  return flags | flag;
}

/**
 * @brief This is a base class for virtual sparse format representation
 * 
 */
class SparseRep {
 protected:
  SparseRep() = default;

 public:
  virtual ~SparseRep();
  /// <summary>
  ///  Copy the same format to a different destination.
  /// </summary>
  /// <param name="data_transfer_manager">manager</param>
  /// <param name="allocator">allocator for the destination</param>
  /// <param name="exec_q_id"></param>
  /// <param name="dst_rep">[out] destination representation</param>
  /// <returns></returns>
  virtual Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
                      int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const = 0;

  virtual Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator, int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const = 0;

 private:
};

/**
 * @brief This class implements SparseTensor. 
 * We represent a SparseTensor COO as a triple <values, indices, shape>. "values" and "indices" themselves
 * are implemented as Tensors. 
 * We follow the Tensor design for memory ownership/management: a sparse-tensor does not own the "value"
 * or "indices" tensors.
 */

class SparseTensor final {
 public:
  /// <summary>
  /// Non-owing constructor. Takes values but indicies are format dependent.
  /// Get Mutable representation and populate as see fit.
  /// </summary>
  /// <param name="elt_type"></param>
  /// <param name="values_shape">shape of value data, no verification is made</param>
  /// <param name="nnz"></param>
  /// <param name="values_data"></param>
  /// <param name="dense_shape">shape of the would be dense tensor</param>
  /// <param name="memory_info"></param>
  SparseTensor(MLDataType elt_type,
               const TensorShape& dense_shape,
               size_t nnz,
               void* values_data,
               const OrtMemoryInfo& memory_info);

  /// <summary>
  /// Constructor that would allocate memory using the supplied allocator
  /// </summary>
  /// <param name="elt_type"></param>
  /// <param name="shape"></param>
  /// <param name="nnz"></param>
  /// <param name="allocator"></param>
  SparseTensor(MLDataType elt_type,
               const TensorShape& dense_shape,
               size_t nnz,
               std::shared_ptr<IAllocator> allocator);

  ~SparseTensor();

  // For now, disallow all copy, assignment, and move.
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SparseTensor);

  // Returns the number of entries in the values tensor (aka "NNZ" or "number of nonzero values")
  // For block sparse formats this may include some zeros in the blocks are considered non-zero.
  size_t NumValues() const { return static_cast<size_t>(values_.Shape().Size()); }

  const Tensor& Values() const {
    return values_;
  }

  Tensor& MutableValues() {
    return values_;
  }

  SparseTensor(SparseTensor&& o) noexcept {
    *this = std::move(o);
  }

  SparseTensor& operator=(SparseTensor&& o) noexcept {
    format_flags_ = o.format_flags_;
    values_ = std::move(o.values_);
    dense_shape_ = std::move(o.dense_shape_);
    rep_ = std::move(o.rep_);
    return *this;
  }

  /// <summary>
  /// Returns SparseFormat that the instance currently holds
  /// if the value returned in kUndefined, the instance is not populated
  /// and any action except MutableRep<>() will fail
  /// </summary>
  /// <returns></returns>
  SparseFormatFlags FormatFlags() const noexcept {
    return format_flags_;
  }
  /// <summary>
  /// Returns a would be dense_shape
  /// </summary>
  /// <returns></returns>
  const TensorShape& Shape() const noexcept {
    return dense_shape_;
  }

  /// <summary>
  /// Returns Tensor element type enum.
  /// Useful for type dispatching
  /// </summary>
  /// <returns></returns>
  int32_t GetElementType() const {
    return values_.GetElementType();
  }

  /// <summary>
  /// Return Element MLDataType
  /// </summary>
  /// <returns></returns>
  MLDataType DataType() const {
    return values_.DataType();
  }

  bool IsDataTypeString() const {
    return values_.IsDataTypeString();
  }

  // Checks if the Tensor contains data type T
  template <class T>
  bool IsDataType() const {
    return values_.IsDataType<T>();
  }

  /// <summary>
  /// GetRep with format checked. Specialization comes with the actual
  /// Rep implementation. Any existing rep is destroyed and replaced.
  /// Specialize for each for the Reps when implemented.
  /// </summary>
  /// <typeparam name="Rep"></typeparam>
  /// <returns>Typed sparse format representation</returns>
  template <typename Rep>
  const Rep* GetRep() const;

  /// <summary>
  /// Create or replace existing mutable representation and make
  /// available for modification. Specialization comes with the actual
  /// Rep implementation.
  /// </summary>
  /// <param name="dense shape"></param>
  /// <returns></returns>
  template <typename Rep>
  Rep* MutableRep();

  const OrtMemoryInfo& Location() const { return values_.Location(); }

  bool OwnsBuffer() const noexcept {
    return allocator_ != nullptr;
  }

  /// <summary>
  /// X-device copy
  /// </summary>
  /// <param name="data_transfer_manager"></param>
  /// <param name="exec_q_id"></param>
  /// <param name="dst_tensor"></param>
  /// <returns></returns>
  Status Copy(const DataTransferManager& data_transfer_manager, int exec_q_id, SparseTensor& dst_tensor) const;

  /// <summary>
  /// CPU only copy
  /// </summary>
  /// <param name="dst_tensor"></param>
  /// <returns></returns>
  Status Copy(const IDataTransfer& data_transfer, SparseTensor& dst_tensor, int exec_q_id) const;
  

 private:
  // New API members
  SparseFormatFlags format_flags_;
  Tensor values_;
  TensorShape dense_shape_;
  AllocatorPtr allocator_;
  std::unique_ptr<SparseRep> rep_;
};

}  // namespace onnxruntime
