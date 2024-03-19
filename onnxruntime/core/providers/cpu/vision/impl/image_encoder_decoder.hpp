#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace ort_extensions {
class BaseImageDecoder {
 public:
  virtual ~BaseImageDecoder() {}

  // HWC
  const std::vector<int64_t>& Shape() const { return shape_; }
  int64_t NumDecodedBytes() const { return decoded_bytes_; }

  bool Decode(uint8_t* output, uint64_t out_bytes) {
    // temporary hack to validate size
    assert(shape_.size() == 3 && out_bytes == shape_[0] * shape_[1] * shape_[2]);
    return DecodeImpl(output, out_bytes);
  }

 protected:
  BaseImageDecoder(const uint8_t* bytes, uint64_t num_bytes)
      : bytes_{bytes}, num_bytes_{num_bytes} {
  }

  const uint8_t* Bytes() const { return bytes_; }
  uint64_t NumBytes() const { return num_bytes_; }

  void SetShape(int height, int width, int channels) {
    assert(height > 0 && width > 0 && channels == 3);
    shape_ = {height, width, channels};
    decoded_bytes_ = height * width * channels;
  }

 private:
  virtual bool DecodeImpl(uint8_t* output, uint64_t out_bytes) = 0;

  const uint8_t* bytes_;
  const uint64_t num_bytes_;
  std::vector<int64_t> shape_;
  int64_t decoded_bytes_{0};
};

class BaseImageEncoder {
 public:
  BaseImageEncoder(const uint8_t* bytes, const std::vector<int64_t>& shape)
      : bytes_{bytes}, shape_{shape} {
    num_bytes_ = 1;
    for (auto dim : shape) {
      num_bytes_ *= dim;
    }

    // avoid re-allocs by assuming worst case of no compression when encoding.
    // use resize so we can memcpy to buffer.
    encoded_image_.resize(num_bytes_, 0);
  }

  const std::vector<int64_t>& Shape() const { return shape_; }

  const std::vector<uint8_t>& Encode() {
    bool encoded = EncodeImpl();
    assert(encoded);
    return encoded_image_;
  }

  virtual ~BaseImageEncoder() {}

 protected:
  const uint8_t* Bytes() const { return bytes_; }
  uint64_t NumBytes() const { return num_bytes_; }

  std::vector<uint8_t>& Buffer() { return encoded_image_; }

 private:
  virtual bool EncodeImpl() = 0;

  const uint8_t* bytes_;
  uint64_t num_bytes_;
  const std::vector<int64_t> shape_;

  std::vector<uint8_t> encoded_image_;
};
}  // namespace ort_extensions
