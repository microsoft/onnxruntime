// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include "core/common/common.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace arrow {
// This  is to support Apache Arrow conventions
// but written C++ style.

/// \class Buffer
/// \brief Object containing a pointer to a piece of contiguous memory with a
/// particular size.
///
/// Buffers have two related notions of length: size and capacity. Size is
/// the number of bytes that might have valid data. Capacity is the number
/// of bytes that were allocated for the buffer in total.
///
/// The actual allocation is intended to be reused in various
/// Arrays, columns and allocations as is OR a slice of it.
///
/// If the buffer holds primitive types, it is OK to
///  copy and slice memory at will. However, if the buffer contains
///  instances of first class objects, you must manage the buffer via
///  corresponding instance of Array
///
///  The buffer object is not mutable but the owner can write to the buffer
///  upon construction and Seal() it making it immutable and setting its ultimate size
///  in bytes.
///
///  A buffer slice that is created from the existing buffer is not mutable and its
///   size and capacity are equal.
///
class Buffer {
 public:
  // \brief Unallocated buffer, use Allocate() to make
  // usable
  Buffer() = default;

  // \brief __dtor
  ~Buffer();
  /////////////////////////////////////////////
  // Making copies of the buffer increase rep_
  // reference count via a shared_ptr
  Buffer(const Buffer& o) : Buffer() {
    *this = o;
  }
  Buffer& operator=(const Buffer& o) {
    if (&o != this) {
      DecRefAndDeallocate();
      if (o.IsAllocated()) {
        o.AddRef();
        rep_ = o.rep_;
      }
    }
    return *this;
  }
  Buffer(Buffer&& o) : Buffer() {
    *this = std::move(o);
  }
  Buffer& operator=(Buffer&& o) {
    if (&o != this) {
      DecRefAndDeallocate();
      rep_ = o.rep_;
      SetViewData(o);
    }
    return *this;
  }

  /// \brief Allocates a new buffer and copies std::string content into it
  static Buffer FromString(const std::string& s, const AllocatorPtr& allocator);

  /// \brief Creates a view of the buffer.
  /// The new Buffer instance point to the same memory allocation as the
  /// original buffer but has a narrow view of it.
  Status Slice(int64_t offset, int64_t size, Buffer&) const;

  /// !\brief Reserves enough capacity in the buffer
  Status Reserve(const int64_t capacity, const AllocatorPtr& allocator);

  /// !\brief Sets buffer to the new size and reallocates if necessary
  Status Resize(const int64_t new_size, const AllocatorPtr& allocator, bool shrink_to_fit = true);

  /// \brief Pads the buffer with zeros [size, capacity)
  void ZeroPadding();

  /// \brief Seals the buffer (makes it immutable)
  /// and sets its view_start_ and view_size_ in bytes so
  /// it can be used for reading.
  /// This happens after the content is written to it
  void Seal();

  /// \brief Returns true if two Buffer instances
  /// refer to the same underlying memory allocation
  bool IsSameMemory(const Buffer& o) const {
    return rep_ == o.rep_;
  }

  /// \brief Compares the buffer ptr, its size,
  /// and its content for equality. This is a byte-wise
  /// comparison. I.e. if the space is occupied by first class
  /// C++ objects, this comparison may not be sufficient
  /// and must be carried out by the owner of this buffer
  /// typically Array class that is type-aware.
  // bool operator==(const Buffer&);

  /// \brief Return the used buffer's view in bytes
  int64_t size() const { return static_cast<int64_t>(view_size_); }
  /// \brief Return the buffer's capacity (number of allocated bytes)
  size_t capacity() const { return (rep_ != nullptr) ? rep_->capacity_ : 0; }

  bool IsAllocated() const { return rep_ != nullptr; }

  bool IsMutable() const { return (rep_ != nullptr) && (rep_->is_mutable_); }

  /// !\brief Immutable buffer accessor
  template <class T>
  const T* data() const { return reinterpret_cast<const T*>(view_start_); }

  /// !\brief Raw buffer access
  /// XXX: Consider specifying needed size as an argument so we
  /// can bounds check here against the actual allocation size???
  template <class T>
  T* mutable_data() {
    assert(IsAllocated());
    ORT_ENFORCE(IsMutable());
    return reinterpret_cast<T*>(rep_->GetData());
  }

  /// \brief Standard swap
  void swap(Buffer& o) {
    std::swap(rep_, o.rep_);
    std::swap(view_start_, o.view_start_);
    std::swap(view_size_, o.view_size_);
  }

 private:
  size_t buffer_size() const {
    return rep_->size_;
  }

  size_t buffer_capacity() const {
    return rep_->capacity_;
  }

  const uint8_t* GetBytesPtr() const {
    return reinterpret_cast<const uint8_t*>(rep_->GetData());
  }

  uint8_t* GetBytesPtr() {
    return reinterpret_cast<uint8_t*>(rep_->GetData());
  }

  void SetViewData(int64_t requested_offset, int64_t requested_size) {
    view_start_ = (GetBytesPtr() + requested_offset);
    view_size_ = static_cast<size_t>(requested_size);
  }

  void SetViewData(const Buffer& o) {
    view_start_ = o.view_start_;
    view_size_ = o.view_size_;
  }

  void AddRef() const {
    rep_->AddRef();
  }

  void DecRefAndDeallocate() {
    if (IsAllocated()) {
      rep_->MayBeDeallocate();
      rep_ = nullptr;
      view_start_ = nullptr;
      view_size_ = 0;
    }
  }

  // When allocating we need to allocate aligned buffer
  // plus the size of Ref which will be constructed with placement new
  // at the beginning of the buffer
  static const size_t RefAlignment = 64;

  // To be allocated at the start of the 64-bytes aligned buffer
  struct Rep {
    size_t size_{0};
    size_t capacity_{0};
    bool is_mutable_{true};
    std::atomic_uint32_t ref_cnt_{1};
    /**
     if allocator_ is null, it means Buffer does not own the allocation.
     Otherwise, Buffer instance will use deleter to deallocate memory.
    */
    AllocatorPtr allocator_{};
    // Some of all members here
    static const size_t MemberSize = sizeof(ref_cnt_) + sizeof(size_) + sizeof(capacity_) + sizeof(is_mutable_) + sizeof(allocator_);
    // Fill out the rest of the object to align on 64 bytes boundary so the actual buffer can start right after that.
    char padding_[(RefAlignment - MemberSize)];

    /// !\brief Initial allocation __ctor
    Rep(size_t size, size_t capacity, const AllocatorPtr& allocator) : Rep() {
      Init(size, capacity, allocator);
    }

    Rep(const Rep&) = delete;
    Rep& operator=(const Rep&) = delete;

    void* GetData() {
      return this + sizeof(*this);
    }

    const void* GetData() const {
      return this + sizeof(*this);
    }

    void AddRef() {
      if(allocator_ != nullptr) {
        ref_cnt_.fetch_add(1, std::memory_order_relaxed);
      }
    }

    void MayBeDeallocate() {
      if (allocator_ != nullptr && 1 == DecRef()) {
        auto allocator = std::move(allocator_);
        this->~Rep();
        allocator->Free(this);
      }
    }

   private:
    Rep() = default;
    ~Rep() = default;

    uint32_t DecRef() {
      return ref_cnt_.fetch_sub(1, std::memory_order_acq_rel);
    }

    void Init(size_t size, size_t capacity, const AllocatorPtr& allocator) {
      size_ = 0;
      capacity_ = capacity;
      allocator_ = allocator;
    }
  };

  Rep* rep_{nullptr};
  // By default this points the beginning of the buffer
  // the view_size is the same as rep::size_
  // however, if this represents a view then it will point
  // someplace within the buffer with a view_size less than rep::size_
  const void* view_start_{nullptr};
  size_t view_size_{0};

  Status AllocateInternal(size_t size, const AllocatorPtr& allocator, Rep*&);

  /// !\brief Slicing constructor
  /// No ref-couting, this is a ptr move
  Buffer(Rep* rep, int64_t requested_offset, int64_t requested_size)
      : rep_(rep) {
    SetViewData(requested_offset, requested_size);
  }
};

/// !\brief Swap support
inline void swap(Buffer& lhs, Buffer& rhs) {
  lhs.swap(rhs);
}

}  // namespace arrow
}  // namespace onnxruntime
