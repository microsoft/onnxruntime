#pragma once
#include <vector>
#include <stdint.h>

namespace my_kernel_lib {

enum Status {
  kOK = 0,
  kError = 1,
};

enum DataType {
  kFloat = 0,
  kInt32 = 1,
  kInt64 = 2,
  kString = 3,
  kBool = 4,
  kUnknown = 5,
};

Status AddKernelTypeShapeInference(const DataType a_type,
                                   const std::vector<int64_t>& a_shape,
                                   const DataType b_type,
                                   const std::vector<int64_t>& b_shape,
                                   DataType* c_type, std::vector<int64_t>* c_shape);

template <typename T>
Status AddKernel(const T* a, const std::vector<int64_t>& a_shape,
                 const T* b, const std::vector<int64_t>& b_shape,
                 T* c, const std::vector<int64_t>& c_shape);

}  // namespace my_kernel_lib
