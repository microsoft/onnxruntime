#include "my_kernel.h"

namespace my_kernel_lib {
Status AddKernelTypeShapeInference(const DataType a_type,
                                   const std::vector<int64_t>& a_shape,
                                   const DataType b_type,
                                   const std::vector<int64_t>& b_shape,
                                   DataType* c_type, std::vector<int64_t>* c_shape) {
  if (a_type != b_type) {
    return kError;
  }
  if (a_shape.size() != b_shape.size()) {
    return kError;
  }
  for (size_t i = 0; i < a_shape.size(); i++) {
    if (a_shape[i] != b_shape[i]) {
      return kError;
    }
  }
  *c_type = a_type;
  *c_shape = a_shape;
  return kOK;
}

template <typename T>
Status AddKernel(const T* a, const std::vector<int64_t>& a_shape,
                 const T* b, const std::vector<int64_t>& b_shape,
                 T* c, const std::vector<int64_t>& c_shape) {
  if (a == nullptr || b == nullptr) {
    return Status::kError;
  }

  int64_t numel = 1;
  for (size_t i = 0; i < a_shape.size(); i++) {
    numel *= a_shape[i];
  }

  for (size_t i = 0; i < numel; i++) {
    c[i] = a[i] + b[i];
  }

  return Status::kOK;
}

template Status AddKernel<float>(const float* a, const std::vector<int64_t>& a_shape,
                                 const float* b, const std::vector<int64_t>& b_shape,
                                 float* c, const std::vector<int64_t>& c_shape);
template Status AddKernel<int32_t>(const int32_t* a, const std::vector<int64_t>& a_shape,
                                   const int32_t* b, const std::vector<int64_t>& b_shape,
                                   int32_t* c, const std::vector<int64_t>& c_shape);
template Status AddKernel<int64_t>(const int64_t* a, const std::vector<int64_t>& a_shape,
                                   const int64_t* b, const std::vector<int64_t>& b_shape,
                                   int64_t* c, const std::vector<int64_t>& c_shape);

}  // namespace my_kernel_lib
