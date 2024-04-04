// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"

namespace _winml {

class vector_backed_buffer
  : public winrt::implements<vector_backed_buffer, wss::IBuffer, Windows::Storage::Streams::IBufferByteAccess> {
 public:
  vector_backed_buffer(size_t size);

  uint32_t Capacity() const;
  uint32_t Length() const;
  void Length(uint32_t /*value*/);

  STDMETHOD(Buffer)(uint8_t** value);

 private:
  std::vector<BYTE> buffer_;
};

}  // namespace _winml
