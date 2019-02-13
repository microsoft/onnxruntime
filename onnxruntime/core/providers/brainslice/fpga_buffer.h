// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
namespace onnxruntime {
namespace fpga {

// Struct represents buffer overflow exception.
struct FPGABufferOverflowException : public std::exception {
 public:
  // Constructor.
  FPGABufferOverflowException(size_t request_size, size_t buffer_size, size_t total_buffer_size)
      : requestSize_(request_size),
        bufferSize_(buffer_size),
        totalBufferSize_(total_buffer_size) {
    std::ostringstream os;
    os << "Fpga buffer: A request of " << requestSize_
       << " bytes exceeds the current buffer size."
       << "Current buffer size:" << bufferSize_
       << ", total buffer size:" << totalBufferSize_;
    errorMessage_ = os.str();
  }

  virtual const char* what() const noexcept override {
    return errorMessage_.c_str();
  }

  // Requested allocation size.
  size_t requestSize_;

  // Remaining buffer size.
  size_t bufferSize_;

  // Total buffer size.
  size_t totalBufferSize_;

  // Error message.
  std::string errorMessage_;
};

using BufferDeleter = std::function<void(void*)>;

// A wrapper class for fpga buffer.
class FPGABuffer {
 public:
  // Empty constructor.
  FPGABuffer();

  // Construct from a buffer and optional deallocator.
  explicit FPGABuffer(void* buffer,
                      size_t size,
                      BufferDeleter deallocator = nullptr);

  // Move constructor.
  FPGABuffer(FPGABuffer&& other) = default;

  // Move assignment.
  FPGABuffer& operator=(FPGABuffer&& other);

  // Get deallocator.
  const BufferDeleter GetDeallocator() const;

  // Deallocate buffer.
  void ReleaseBuffer();

  // virtual destructor.
  virtual ~FPGABuffer();

  // Get the buffer start.
  char* GetStart() const;

  // Get the buffer end.
  char* GetEnd() const;

  // Get buffer size.
  size_t GetBufferSize() const;

  // Reset buffer size.
  void ResetBufferSize(size_t buffer_size);

 protected:
  // Pointer to the start of the buffer.
  char* buffer_start_;

  // Pointer to the current end of the buffer.
  size_t buffer_size_;

  // Deallocator.
  BufferDeleter deallocator_;

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FPGABuffer);
};

// A wrapper class for reading/writing FPGABuffer.
class FPGABufferStream {
 public:
  // Empty constructor.
  FPGABufferStream();

  // Constructor.
  FPGABufferStream(const FPGABuffer& buffer);

  // Move constructor.
  FPGABufferStream(FPGABufferStream&& other);

  // Move assignment.
  FPGABufferStream& operator=(FPGABufferStream&& other);

  // Get the buffer current.
  char* GetCurrent() const;

  // Get current offset from start.
  size_t GetCurrentOffset() const;

  // Get remaining buffer size.
  size_t GetRemainingBufferSize() const;

  // Reset current pos pointer to start pointer.
  void Reset();

  // Reset to specific position;
  void Reset(size_t position);

  // Reserve a number of bytes on the buffer.
  void* Reserve(size_t size);

  // Serialize APIs.
  // This is used to serialize non-fixed size data to the buffer.
  void* Serialize(const void* value, size_t size);

  // This is used to serialize simple data with fixed size to the buffer.
  template <typename T>
  T* Serialize(const T value) {
    void* location = Serialize(&value, sizeof(T));
    return reinterpret_cast<T*>(location);
  }

  // Construct a type in place at the current write position of the buffer.
  template <typename T, typename... Args>
  T& Emplace(Args&&... args) {
    return *new (Reserve(sizeof(T))) T(std::forward<Args>(args)...);
  }

  // Deserialize APIs.
  // Return a pointer to size bytes from the current buffer position and advance the
  // position that amount.
  void* Read(size_t size);

  // Read size bytes from the end of the buffer, returning a pointer to the beginning of
  // the range, and reduce the buffer size by size.
  void* ReadFromEnd(size_t size);

  // This is used to deserialize non-fixed size data from the buffer.
  void Deserialize(void* value, size_t size);

  // Advance the buffer by size bytes.
  void Ignore(size_t size);

  // This is used to deserialize non-fixed size data from the buffer, from the end.
  void DeserializeFromEnd(void* value, size_t size);

  // Ignore bytes from end of the buffer.
  void IgnoreFromEnd(size_t size);

  // Return a reference to a type which is serialized at the current position in the buffer
  // and advance the buffer position by the size of the type.
  template <typename T>
  T& ReadAs() {
    return *static_cast<T*>(Read(sizeof(T)));
  }

  // Return a reference to a type which is serialized at the end of the buffer and reduce
  // the buffer size by the size of the type.
  template <typename T>
  T& ReadFromEndAs() {
    return *static_cast<T*>(ReadFromEnd(sizeof(T)));
  }

  // This is used to deserialize simple data with fixed size from the buffer.
  template <typename T>
  void Deserialize(T* value) {
    Deserialize(value, sizeof(T));
  }

  // This is used to deserialize simple data with fixed size from the buffer, from the end.
  template <typename T>
  void DeserializeFromEnd(T* value) {
    DeserializeFromEnd(value, sizeof(T));
  }

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FPGABufferStream);
  // Pointer to the start of the buffer.
  char* buffer_start_;

  // Buffer size.
  size_t buffer_size_;

  // Offset to the current read location.
  size_t offset_;
};

}  // namespace fpga
}  // namespace onnxruntime
