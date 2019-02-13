#include "core/providers/brainslice/fpga_buffer.h"

namespace onnxruntime {
namespace fpga {
FPGABuffer::FPGABuffer()
    : FPGABuffer(nullptr, 0) {
}

FPGABuffer::FPGABuffer(void* buffer,
                       size_t size,
                       BufferDeleter deallocator)
    : buffer_start_(reinterpret_cast<char*>(buffer)),
      buffer_size_(size),
      deallocator_(deallocator) {
}

FPGABuffer&
FPGABuffer::operator=(FPGABuffer&& other) {
  if (this != &other) {
    ReleaseBuffer();

    buffer_start_ = other.buffer_start_;
    buffer_size_ = other.buffer_size_;
    deallocator_ = other.deallocator_;

    other.buffer_start_ = nullptr;
    other.buffer_size_ = 0;
    other.deallocator_ = nullptr;
  }

  return *this;
}

FPGABuffer::~FPGABuffer() {
  ReleaseBuffer();
}

const BufferDeleter FPGABuffer::GetDeallocator() const {
  return deallocator_;
}

void FPGABuffer::ReleaseBuffer() {
  if (deallocator_ != nullptr) {
    deallocator_(buffer_start_);
  }
  buffer_start_ = nullptr;
  buffer_size_ = 0;
  deallocator_ = nullptr;
}

char* FPGABuffer::GetStart() const {
  return buffer_start_;
}

char* FPGABuffer::GetEnd() const {
  return buffer_start_ + buffer_size_;
}

size_t FPGABuffer::GetBufferSize() const {
  return buffer_size_;
}

void FPGABuffer::ResetBufferSize(size_t bufferSize) {
  if (bufferSize > buffer_size_) {
    ORT_THROW("New buffer size:%u cannot be bigger than current buffer size:%u",
                      bufferSize,
                      buffer_size_);
  }
  buffer_size_ = bufferSize;
}

FPGABufferStream::FPGABufferStream()
    : buffer_start_(nullptr),
      buffer_size_(0),
      offset_(0) {
}

FPGABufferStream::FPGABufferStream(const FPGABuffer& buffer)
    : buffer_start_(buffer.GetStart()),
      buffer_size_(buffer.GetBufferSize()),
      offset_(0) {
}

FPGABufferStream::FPGABufferStream(FPGABufferStream&& other)
    : FPGABufferStream() {
  *this = std::move(other);
}

FPGABufferStream& FPGABufferStream::operator=(FPGABufferStream&& other) {
  if (this != &other) {
    buffer_start_ = other.buffer_start_;
    buffer_size_ = other.buffer_size_;
    offset_ = other.offset_;

    other.buffer_start_ = nullptr;
    other.buffer_size_ = 0;
    other.offset_ = 0;
  }

  return *this;
}

char* FPGABufferStream::GetCurrent() const {
  return buffer_start_ + offset_;
}

size_t FPGABufferStream::GetCurrentOffset() const {
  if (offset_ > buffer_size_) {
    ORT_THROW("Incorrect current offset:%u, buffer size:%u",
                      offset_,
                      buffer_size_);
  }
  return offset_;
}

size_t
FPGABufferStream::GetRemainingBufferSize() const {
  if (offset_ > buffer_size_) {
    ORT_THROW("Incorrect current offset:%u, buffer size:%u",
                      offset_,
                      buffer_size_);
  }
  return buffer_size_ - offset_;
}

void FPGABufferStream::Reset() {
  Reset(0);
}

void FPGABufferStream::Reset(size_t position) {
  if (position > buffer_size_) {
    throw FPGABufferOverflowException(position, buffer_size_, buffer_size_);
  }
  offset_ = position;
}

void* FPGABufferStream::Reserve(size_t size) {
  if (offset_ + size > buffer_size_) {
    throw FPGABufferOverflowException(size, buffer_size_ - offset_, buffer_size_);
  } else {
    char* const location = buffer_start_ + offset_;
    offset_ += size;
    return location;
  }
}

void* FPGABufferStream::Read(size_t size) {
  return Reserve(size);
}

void* FPGABufferStream::ReadFromEnd(size_t size) {
  if (offset_ + size > buffer_size_) {
    throw FPGABufferOverflowException(size, buffer_size_ - offset_, buffer_size_);
  } else {
    buffer_size_ -= size;
    return buffer_start_ + buffer_size_;
  }
}

void* FPGABufferStream::Serialize(const void* value, size_t size) {
  void* const location = Reserve(size);
  std::memcpy(location, value, size);
  return location;
}

void FPGABufferStream::Deserialize(void* value, size_t size) {
  std::memcpy(value, Read(size), size);
}

void FPGABufferStream::DeserializeFromEnd(void* value, size_t size) {
  std::memcpy(value, ReadFromEnd(size), size);
}

void FPGABufferStream::Ignore(size_t size) {
  Read(size);
}

void FPGABufferStream::IgnoreFromEnd(size_t size) {
  ReadFromEnd(size);
}

}  // namespace fpga
}  // namespace onnxruntime
