// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include "hip/hip_fp16.h"
#include "fast_gelu_kernel.h"

namespace py = pybind11;

template <typename T, int ThreadsPerBlock, int VecSize>
class FastGelu: public Operator<T> {
 public:
  FastGelu(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length) :
    input_(reinterpret_cast<T*>(input.ptr())),
    bias_(reinterpret_cast<T*>(bias.ptr())),
    output_(reinterpret_cast<T*>(output.ptr())),
    input_length_(input_length),
    bias_length_(bias_length),
    Operator<T>() {}

  void Run() {
    LaunchFastGelu<T, ThreadsPerBlock, VecSize>(input_, bias_, output_, input_length_, bias_length_);
  }

 private:
  T* input_;
  T* bias_;
  T* output_;
  int input_length_;
  int bias_length_;
};

#define REGISTER_OP(name, type, threads_per_block, vec_size)                                              \
  py::class_<name<type, threads_per_block, vec_size>>(m, #name"_"#type"_"#threads_per_block"_"#vec_size)  \
    .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int>())                                  \
    .def("SetRepeats", &name<type, threads_per_block, vec_size>::SetRepeats)                              \
    .def("Profile", &name<type, threads_per_block, vec_size>::Profile)                                    \
    .def("Run", &name<type, threads_per_block, vec_size>::Run);
  
#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, threads_per_block)  \
  REGISTER_OP(name, type, threads_per_block, 1)                      \
  REGISTER_OP(name, type, threads_per_block, 2)                      \
  REGISTER_OP(name, type, threads_per_block, 4)                      \
  REGISTER_OP(name, type, threads_per_block, 8)                      \
  REGISTER_OP(name, type, threads_per_block, 16)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(name, type)  \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 64)             \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 128)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 192)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 256)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 320)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 384)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 448)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 512)         
  
void InitFastGelu(py::module m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(FastGelu, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(FastGelu, float);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(FastGelu, double);
}
