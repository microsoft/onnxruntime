// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "core/automl/buffer.h"

namespace onnxruntime {
namespace arrow {

// Per requirements https://arrow.apache.org/docs/format/Layout.html
constexpr size_t BufferAllocationAlignment = 64;

namespace bit_utils {
// Round up x to a multiple of y.
// Example:
//   Roundup(13, 5)   => 15
//   Roundup(201, 16) => 208
inline size_t Roundup(size_t x, size_t y) {
  return ((x + y - 1) / y) * y;
}
}

static inline Status CheckAllocationSize(int64_t requested_size, size_t& size) {
  if (requested_size < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Allocation size is not positive: ", requested_size);
  }
  size = static_cast<size_t>(requested_size);
  return Status::OK();
}

Buffer::~Buffer() {
}

Buffer::Rep::~Rep() {
  if (allocator_) {
    allocator_->Free(data_.p);
  }
}

struct AllocatorDeleter {
  AllocatorPtr alloc_;
  void operator()(void* p) const { if(p) alloc_->Free(p); }
};

using UniqueAllocPtr = std::unique_ptr<void, AllocatorDeleter>;

Buffer::Buffer(const void* data, int64_t alloc_size, AllocatorPtr allocator) {
  size_t size = 0;
  CheckAllocationSize(alloc_size, size);
  rep_ = std::make_shared<Rep>(data, size, allocator);
  rep_->size_;
  rep_->is_mutable_ = false;
  view_start_ = data;
  view_size_ = size;
}

Buffer Buffer::FromString(const std::string& s, const AllocatorPtr& allocator) {
  Buffer result;
  if (!result.Resize(static_cast<int64_t>(s.size()), allocator, false).IsOK()) {
    ORT_THROW("Failed to wrap string of size: ", s.size());
  }
  memcpy_s(result.mutable_data<void>(), result.capacity(), s.data(), s.size());
  result.ZeroPadding();
  result.Seal();
  return result;
}

Status Buffer::AllocateInternal(size_t size, const AllocatorPtr& allocator) {
  size_t capacity = bit_utils::Roundup(size, BufferAllocationAlignment);
  void* allocation = allocator->AllocArrayWithAlignment<BufferAllocationAlignment>(1, capacity);
  if (allocation == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Memory allocation request failed for capacity:", capacity);
  }
  UniqueAllocPtr alloc(allocation, {allocator});
  auto new_rep = std::make_shared<Rep>(alloc.get(), capacity, allocator);
  alloc.release();
  rep_.swap(new_rep);
  return Status::OK();
}

Status Buffer::Reserve(const int64_t requested_capacity, const AllocatorPtr& allocator) {
  size_t new_capacity = 0;
  ORT_RETURN_IF_ERROR(CheckAllocationSize(requested_capacity, new_capacity));
  if (!IsMutable() || capacity() < new_capacity) {
    if (IsMutable()) {
      // reallocate and copy any data there
      auto old_rep = rep_;
      ORT_RETURN_IF_ERROR(AllocateInternal(new_capacity, allocator));
      rep_->size_ = old_rep->size_;
      auto copy_size = std::min(new_capacity, old_rep->capacity_);
      memcpy_s(rep_->data_.p, new_capacity, old_rep->data_.cp, copy_size);
    } else {
      // Old data is immutable will stay in the old buffer or get discarded
      ORT_RETURN_IF_ERROR(AllocateInternal(new_capacity, allocator));
    }
  }

  return Status::OK();
}

Status Buffer::Resize(const int64_t size, const AllocatorPtr& allocator, bool shrink_to_fit) {
  size_t new_size = 0;
  ORT_RETURN_IF_ERROR(CheckAllocationSize(size, new_size));
  if (IsMutable() && size < this->size()) {
    size_t new_capacity = bit_utils::Roundup(new_size, BufferAllocationAlignment);
    if (shrink_to_fit && new_capacity != buffer_capacity()) {
      auto old_rep = rep_;
      ORT_RETURN_IF_ERROR(AllocateInternal(new_capacity, allocator));
      auto copy_size = std::min(new_capacity, old_rep->capacity_);
      memcpy_s(rep_->data_.p, new_capacity, old_rep->data_.cp, copy_size);
    }
  } else {
    ORT_RETURN_IF_ERROR(Reserve(size, allocator));
  }
  rep_->size_ = new_size;
  SetViewData(0, rep_->size_);
  return Status::OK();
}

void Buffer::ZeroPadding() {
  if (IsAllocated()) {
    memset(mutable_data<uint8_t*>() + buffer_size(), 0, capacity() - buffer_size());
  }
}

void Buffer::Seal() {
  assert(IsAllocated());
  // We require the buffer to be Sealed before
  // there are multiple references to it.
  ORT_ENFORCE(rep_.use_count() == 1);
  ORT_ENFORCE(IsMutable());
  rep_->is_mutable_ = false;
  view_start_ = rep_->data_.cp;
  view_size_ = rep_->size_;
}

Status Buffer::Slice(int64_t requested_offset, int64_t requested_size, Buffer& out) const {
  assert(IsAllocated());
  ORT_ENFORCE(!IsMutable());
  if (requested_offset < 0 || static_cast<size_t>(requested_offset) >= buffer_size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Requested offset is out of bounds of the original buffer");
  }

  if (requested_size < 1 || static_cast<size_t>(requested_offset + requested_size) >= buffer_size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Requested slice size is not positive or outside the original buffer");
  }

  Buffer result(rep_, requested_offset, requested_size);
  out.swap(result);
  return Status::OK();
}

}  // namespace arrow
}  // namespace onnxruntime
