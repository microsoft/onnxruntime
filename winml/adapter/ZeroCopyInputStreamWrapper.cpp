// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "ZeroCopyInputStreamWrapper.h"

#include "winrt/Windows.Foundation.h"

using namespace Windows::AI::MachineLearning;

// ZeroCopyInputStreamWrapper
ZeroCopyInputStreamWrapper::ZeroCopyInputStreamWrapper(
    ABI::Windows::Storage::Streams::IRandomAccessStreamReference* stream) :
      data_(nullptr) {
    winrt::copy_from_abi(stream_, (void*)stream);
}

// ZeroCopyInputStreamWrapper
ZeroCopyInputStreamWrapper::ZeroCopyInputStreamWrapper(
    void* data,
    int size) : 
      data_(data),
      size_(size) {
}

bool ZeroCopyInputStreamWrapper::Next(
    const void** data,
    int* size) {
  if (finished_reading_) {
    return false;
  }

  if (stream_ != nullptr)
  {
    auto content = stream_.OpenReadAsync().get();

    wss::Buffer buffer(static_cast<uint32_t>(content.Size()));
    auto result = content.ReadAsync(
                            buffer,
                            buffer.Capacity(),
                            wss::InputStreamOptions::None)
                      .get();

    bytes_ = buffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>();
  #ifdef LAYERING_DONE
    WINML_THROW_HR_IF_NULL_MSG(E_UNEXPECTED, bytes_, "Model stream is invalid.");
    WINML_THROW_IF_FAILED_MSG(
        bytes_->Buffer(reinterpret_cast<byte**>(const_cast<void**>(data))),
        "Failed to acquire buffer from model stream.");
  #else
    bytes_->Buffer(reinterpret_cast<byte**>(const_cast<void**>(data)));
  #endif

    *size = static_cast<uint32_t>(content.Size());
  }
  else if (data_ != nullptr)
  {
    *data = data_;
    *size = size_;
  }

  finished_reading_ = true;
  return true;
}

// BackUp is used when parsing encounters an error and needs to move
// back to the beginning of the erroneous chunk. We don't support random access,
// so we don't have a pointer to move back, but this can also happen for
// decrypted strings since they can have extra memory at the end that
// isn't valid. We don't want to parse non-model related data so we
// don't support this. I'd like to thrown an error here, but protobuf would
// eat that error and terminate the app. So instead we do nothing and handle
// this in LoadFromStream when the protobuf parsing returns false.
void ZeroCopyInputStreamWrapper::BackUp(int count) {
  // purposely do nothing.
}

// the following methods are required by the interface,
// but they aren't actually used by ModelProto parse code,
bool ZeroCopyInputStreamWrapper::Skip(
    int count) {
#ifdef LAYERING_DONE
  WINML_THROW_HR(E_NOTIMPL);
#endif
  return false;
}

__int64
ZeroCopyInputStreamWrapper::ByteCount() const {
#ifdef LAYERING_DONE
  WINML_THROW_HR(E_NOTIMPL);
#endif
  return 0;
}
