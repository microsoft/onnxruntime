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
  kUndefined = 0x0U,        // For completeness
  kCoo = 0x1U,              // 1-D or 2-D indices
  kCsrc = 0x1U << 1,        // Both CSR(C)
  kBlockSparse = 0x1U << 2  // as in OpenAI
};

std::ostream& operator<<(std::ostream&, SparseFormatFlags);

inline SparseFormatFlags operator|(SparseFormatFlags flags, SparseFormatFlags flag) noexcept {
  return static_cast<SparseFormatFlags>(static_cast<uint32_t>(flags) | static_cast<uint32_t>(flag));
}

inline SparseFormatFlags operator&(SparseFormatFlags flags, SparseFormatFlags flag) noexcept {
  return static_cast<SparseFormatFlags>(static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag));
}

inline SparseFormatFlags TurnOff(SparseFormatFlags flags, SparseFormatFlags flag) noexcept {
  return static_cast<SparseFormatFlags>(static_cast<uint32_t>(flags) & ~static_cast<uint32_t>(flag));
}

inline bool IsSet(SparseFormatFlags flags, SparseFormatFlags flag) noexcept {
  return SparseFormatFlags::kUndefined != (flags & flag);
}

inline SparseFormatFlags Set(SparseFormatFlags flags, SparseFormatFlags flag) noexcept {
  return flags | flag;
}

/**
 * @brief This is a base class for virtual sparse format representation
 * 
 */
class SparseRep {
 protected:
  SparseRep(MLDataType data_type, const TensorShape& values_shape, int64_t total_buffer_size, const AllocatorPtr& allocator);
  SparseRep(MLDataType data_type, const TensorShape& values_shape, void* values, const OrtMemoryInfo& mem_info);

  void* IndicesStart(int64_t align) {
    return reinterpret_cast<uint8_t*>(values_.MutableDataRaw()) + Roundup(values_.SizeInBytes(), align);
  }

 public:
  virtual ~SparseRep();

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SparseRep);

  const Tensor& Values() const noexcept {
    return values_;
  }

  Tensor& MutableValues() noexcept {
    return values_;
  }

  bool OwnBuffer() const noexcept {
    return allocator_ != nullptr;
  }

  const OrtMemoryInfo& Location() const {
    return values_.Location();
  }

  // Round up size to a multiple of align.
  // Example:
  //   Roundup(13, 5)   => 15
  //   Roundup(201, 16) => 208
  static int64_t Roundup(int64_t size, int64_t align) {
    return ((size + align - 1) / align) * align;
  }

  /// <summary>
  /// Calculate required buffer size. We will place indices
  /// after data and make sure indices start at int64_t aligned place
  /// </summary>
  /// <returns></returns>
  static int64_t CalculateRequiredBufferSize(int64_t data_size, int64_t indices_size, int64_t align) {
    return Roundup(data_size, align) + indices_size;
  }

  /// <summary>
  ///  Copy the same format to a different destination.
  /// </summary>
  /// <param name="data_transfer_manager">manager</param>
  /// <param name="allocator">allocator for the destination</param>
  /// <param name="exec_q_id"></param>
  /// <param name="dst_rep">[out] destination representation</param>
  /// <returns>status</returns>
  virtual Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
                      int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const = 0;

  virtual Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator, int exec_q_id,
                      std::unique_ptr<SparseRep>& dst_rep) const = 0;

  /// <summary>
  /// Calculates and returns how much this fully initialized SparseTensor data (would)
  /// occupy in a contiguous allocation block
  /// </summary>
  /// <returns></returns>
  virtual int64_t RequiredAllocationSize() const noexcept = 0;

 private:
  void ReleaseBuffer();

  AllocatorPtr allocator_;  // Used to allocate p_data_ or nullptr
  void* p_data_;
  Tensor values_;
};

/**
 * @brief This class implements SparseTensor. 
 * We represent a SparseTensor COO as a triple <values, indices, shape>. "values" and "indices" themselves
 * are implemented as Tensors. 
 * We follow the Tensor design for memory ownership/management: a sparse-tensor does not own the "value"
 * or "indices" tensors.
 */

class SparseTensor final {
  SparseTensor(MLDataType elt_type,
               const TensorShape& dense_shape,
               const OrtMemoryInfo& location,
               std::shared_ptr<IAllocator>&& allocator);
 public:
  /// <summary>
  /// Non-owing constructor.
  /// </summary>
  /// <param name="elt_type">MLDataType must be one of the primitive data types</param>
  /// <param name="dense_shape">shape of the would be dense tensor</param>
  /// <param name="location"></param>
  SparseTensor(MLDataType elt_type,
               const TensorShape& dense_shape,
               const OrtMemoryInfo& location)
      : SparseTensor(elt_type, dense_shape, location, AllocatorPtr()) {
  }

  /// <summary>
  /// Constructor that would allocate memory using the supplied allocator.
  /// When the tensor owns the memory it strives to allocate a single contiguous buffer
  /// </summary>
  /// <param name="elt_type">MLDataType must be one of the primitive data types</param>
  /// <param name="dense_shape">shape of the would be dense tensor</param>
  /// <param name="allocator">allocator to allocate all of the memory</param>
  SparseTensor(MLDataType elt_type,
               const TensorShape& dense_shape,
               std::shared_ptr<IAllocator> allocator)
      : SparseTensor(elt_type, dense_shape, allocator->Info(), std::move(allocator)) {
  }

  SparseTensor();

  ~SparseTensor();

  // For now, disallow all copy, assignment, and move.
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SparseTensor);

  // Returns the number of entries in the values tensor (aka "NNZ" or "number of nonzero values")
  // For block sparse formats this may include some zeros in the blocks are considered non-zero.
  size_t NumValues() const { return static_cast<size_t>(rep_->Values().Shape().Size()); }

  const Tensor& Values() const {
    return rep_->Values();
  }

  SparseTensor(SparseTensor&& o) noexcept {
    *this = std::move(o);
  }

  SparseTensor& operator=(SparseTensor&& o) noexcept;

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
  /// Tests if a specific flag set in the mask
  /// </summary>
  /// <param name="flag">flag to be tested</param>
  /// <returns></returns>
  bool IsFormatFlagSet(SparseFormatFlags flag) const noexcept {
    return IsSet(format_flags_, flag);
  }

  /// <summary>
  /// Returns a would be dense_shape
  /// This does not describe shapes of values of indices
  /// and has no relation to
  /// </summary>
  /// <returns></returns>
  const TensorShape& Shape() const noexcept {
    return dense_shape_;
  }

  /// <summary>
  /// Calculates and returns how much this fully initialized SparseTensor data (would)
  /// occupy in a contiguous allocation block, or, in fact, occupies if it owns the buffer.
  /// </summary>
  /// <returns></returns>
  int64_t RequiredAllocationSize() const noexcept {
    return rep_->RequiredAllocationSize();
  }

  /// <summary>
  /// Returns Tensor element type enum.
  /// Useful for type dispatching
  /// </summary>
  /// <returns></returns>
  int32_t GetElementType() const {
    return ml_data_type_->GetDataType();
  }

  /// <summary>
  /// Return Element MLDataType
  /// </summary>
  /// <returns></returns>
  MLDataType DataType() const noexcept {
    return ml_data_type_;
  }

  /// <summary>
  /// Test for string type
  /// </summary>
  /// <typeparam name="Rep"></typeparam>
  /// <returns>true if tensor values are strings</returns>
  bool IsDataTypeString() const {
    return utils::IsPrimitiveDataType<std::string>(ml_data_type_);
  }

  /// <summary>
  /// Checks if the Tensor contains data type T
  /// </summary>
  /// <typeparam name="Builder"></typeparam>
  /// <returns>true if tensor contains</returns>
  template <class T>
  bool IsDataType() const {
    return utils::IsPrimitiveDataType<T>(ml_data_type_);
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
  /// Rep implementation. The rep is not created at the time of the call
  /// as some of the necessary data might be missing. Instead, this function
  /// returns a builder object that allows you call Create() with a rep specific
  /// argument set.
  /// </summary>
  /// <param name="dense shape"></param>
  /// <returns></returns>
  template <typename Builder>
  Builder RepBuilder();

  const OrtMemoryInfo& Location() const noexcept { return location_; }

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
  SparseFormatFlags format_flags_;
  TensorShape dense_shape_;
  AllocatorPtr allocator_;
  OrtMemoryInfo location_;
  const PrimitiveDataTypeBase* ml_data_type_;
  std::unique_ptr<SparseRep> rep_;
};

}  // namespace onnxruntime
