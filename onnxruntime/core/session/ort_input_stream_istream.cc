// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ort_input_stream_istream.h"
#include <vector>

namespace onnxruntime {

class OrtInputStreamIStreamBuf final : public std::streambuf {
 public:
  OrtInputStreamIStreamBuf(OrtInputStream& stream, size_t buffer_size)
      : stream_(stream), buffer_(buffer_size) {
    setg(buffer_.data(), buffer_.data(), buffer_.data());
  }

 protected:
  int_type underflow() override {
    size_t num_read = stream_.Read(buffer_.data(), buffer_.size(), stream_.user_object);

    if (num_read > 0) {
      setg(buffer_.data(), buffer_.data(), buffer_.data() + num_read);
      return traits_type::to_int_type(*gptr());
    } else {
      return traits_type::eof();
    }
  }

 private:
  OrtInputStream& stream_;
  std::vector<char> buffer_;
};

OrtInputStreamIStream::OrtInputStreamIStream(OrtInputStream& stream, size_t buffer_size)
    : std::istream(new OrtInputStreamIStreamBuf(stream, buffer_size)) {
}

OrtInputStreamIStream::~OrtInputStreamIStream() {
  delete rdbuf();
}

}  // namespace onnxruntime
