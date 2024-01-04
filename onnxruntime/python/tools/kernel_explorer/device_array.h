// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tunable/util.h"

#define CALL_THROW CUDA_CALL_THROW
#define MALLOC cudaMalloc
#define FREE cudaFree
#define MEMCPY cudaMemcpy
#define MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#elif USE_ROCM
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/util.h"

#define CALL_THROW HIP_CALL_THROW
#define MALLOC hipMalloc
#define FREE hipFree
#define MEMCPY hipMemcpy
#define MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#endif

namespace py = pybind11;

namespace onnxruntime {

class DeviceArray {
 public:
  DeviceArray(size_t ptr, ssize_t size, ssize_t itemsize)
      : host_{reinterpret_cast<void*>(ptr)}, size_{size}, itemsize_{itemsize} {
    void* dev_ptr;
    CALL_THROW(MALLOC(&dev_ptr, size_ * itemsize_));
    device_.reset(dev_ptr, [](void* dev_ptr) { CALL_THROW(FREE(dev_ptr)); });
    CALL_THROW(MEMCPY(device_.get(), host_, size_ * itemsize_, MEMCPY_HOST_TO_DEVICE));
  }
  explicit DeviceArray(py::array x) : DeviceArray(x.request()) {}
  DeviceArray(const DeviceArray&) = default;
  DeviceArray& operator=(const DeviceArray&) = default;

  void UpdateHostNumpyArray() {
    CALL_THROW(MEMCPY(host_, device_.get(), size_ * itemsize_, MEMCPY_DEVICE_TO_HOST));
  }

  void UpdateDeviceArray() {
    CALL_THROW(MEMCPY(device_.get(), host_, size_ * itemsize_, MEMCPY_HOST_TO_DEVICE));
  }

  void* ptr() const {
    return device_.get();
  }

 private:
  explicit DeviceArray(py::buffer_info buf) : DeviceArray(reinterpret_cast<size_t>(buf.ptr), buf.size, buf.itemsize) {}

  std::shared_ptr<void> device_;
  void* host_;
  py::ssize_t size_;
  py::ssize_t itemsize_;
};

}  // namespace onnxruntime

#undef CALL_THROW
#undef MALLOC
#undef FREE
#undef MEMCPY
#undef MEMCPY_HOST_TO_DEVICE
#undef MEMCPY_DEVICE_TO_HOST
