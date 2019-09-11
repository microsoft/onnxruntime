#pragma once

#include "winrt/Windows.Storage.Streams.h"
#include <robuffer.h>

namespace Windows::AI::MachineLearning {
// _ZeroCopyInputStreamWrapper is a helper class that allows a ZeroCopyInputStream,
// which is a protobuf type, to read from an IRandomAccessStreamReference, which is
// a winrt type.
class ZeroCopyInputStreamWrapper : public google::protobuf::io::ZeroCopyInputStream {
 public:
  ZeroCopyInputStreamWrapper() = delete;

  ZeroCopyInputStreamWrapper(
      wss::IRandomAccessStreamReference stream);

  // ModelProto load only uses "Next" method
  bool
  Next(
      const void** data,
      int* size);

  void
  BackUp(
      int count);

  bool
  Skip(
      int count);

  __int64
  ByteCount() const;

 private:
  wss::IRandomAccessStreamReference stream_;
  bool finished_reading_ = false;
  winrt::com_ptr<::Windows::Storage::Streams::IBufferByteAccess> bytes_;
};

}  // namespace Windows::AI::MachineLearning