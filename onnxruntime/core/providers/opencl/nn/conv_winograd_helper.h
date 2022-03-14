#pragma once

#include <tuple>
#include "core/framework/tensor.h"

namespace onnxruntime {

class WinogradHelper {
 public:
  WinogradHelper(AllocatorPtr& cpu_alloc, int compute_unit, int kernel_size);
  ~WinogradHelper() = default;

  std::unique_ptr<Tensor> TransformWeight(const Tensor* source, int output_channel, int input_channel);

 private:
  std::unique_ptr<Tensor> AllocWeightTensor(int batch, int channel, int unit_ci, int unit_co);

  AllocatorPtr cpu_alloc_;
  std::unique_ptr<Tensor> G_;
  int wino_size_;
  int unit_;
  int kernel_size_;
};

}  // namespace onnxruntime
