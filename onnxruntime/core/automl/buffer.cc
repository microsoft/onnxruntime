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
  if (alloc_size < 1) {
    ORT_THROW("Allocation size is not positive: ", alloc_size);
  }
  size_t size = static_cast<size_t>(alloc_size);
  rep_ = std::make_shared<Rep>(data, size, size, allocator);
  rep_->is_mutable_ = false;
  view_start_ = data;
  view_size_ = size;
}

Buffer Buffer::FromString(const std::string& s, const AllocatorPtr& allocator) {
  Buffer result;
  ORT_ENFORCE(result.AllocateInternal(s.size(), allocator).IsOK());
  memcpy_s(result.mutable_data<void>(), result.capacity(), s.data(), s.size());
  result.ZeroPadding();
  result.Seal();
  return result;
}

Status Buffer::Allocate(int64_t requested_size, const AllocatorPtr& allocator) {
  assert(!IsAllocated());
  if(requested_size < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Requested allocation size is not positive");
  }
  size_t size = static_cast<size_t>(requested_size);
  return AllocateInternal(size, allocator);
}

Status Buffer::AllocateInternal(size_t size, const AllocatorPtr& allocator) {
  size_t capacity = bit_utils::Roundup(size, BufferAllocationAlignment);
  void* allocation = allocator->AllocArrayWithAlignment<BufferAllocationAlignment>(1, capacity);
  if (allocation == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Memory allocation request failed for capacity:", capacity);
  }
  UniqueAllocPtr alloc(allocation, {allocator});
  auto new_rep = std::make_shared<Rep>(alloc.get(), size, capacity, allocator);
  alloc.release();
  rep_.swap(new_rep);
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
  if (requested_offset < 0 || static_cast<size_t>(requested_offset) >= buffer_size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Requested offset is outside of the buffer");
  }

  if (requested_size < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Requested slice size is not positive");
  }

  return Status::OK();
}

}  // namespace arrow
}  // namespace onnxruntime
