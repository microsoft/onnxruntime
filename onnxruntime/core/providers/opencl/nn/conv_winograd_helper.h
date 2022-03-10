#pragma once

#include <memory>
#include <tuple>
#include "core/framework/tensor.h"

namespace onnxruntime {
typedef std::vector<int64_t> FBshape;
struct DirectBuffer;
using DirectBufferPtr = std::shared_ptr<DirectBuffer>;

struct DirectBuffer {
  FBshape shape;
  int64_t size;
  std::shared_ptr<float[]> buff;
  ~DirectBuffer() {}
  DirectBuffer() : size(0) {}
  DirectBuffer(const DirectBuffer& other) = delete;
  DirectBuffer(DirectBuffer&& other) = delete;

  // pass value here won't impact  performance
  void Create(FBshape shape_) {
    shape = shape_;
    size = 1;
    for (auto n : shape) {
      size *= n;
    }
    buff = std::shared_ptr<float[]>(new float[size], [](float* p) { delete[] p; });
    return;
  }
  void Create(int64_t w, int64_t h) {
    Create({w, h});
  }
  int Fill(std::vector<float> v) {
    if (v.size() != size) {
      return -1;
    }
    std::memcpy(buff.get(), v.data(), size * sizeof(float));
    return 0;
  }
};

class WinogradHelper {
 public:
  WinogradHelper(int computeUnit, int kernelSize);
  ~WinogradHelper() = default;

  DirectBufferPtr TransformWeight(const float* source, int output_channel, int input_channel);

 private:
  DirectBufferPtr AllocWeightTensor(int batch, int channel, int unitCi, int unitCo);

 private:
  DirectBufferPtr G_;
  int wino_size_;
  int unit_;
  int kernel_size_;
};

}  // namespace onnxruntime
