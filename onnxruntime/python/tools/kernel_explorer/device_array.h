// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/bert/util.h"

namespace py = pybind11;

namespace onnxruntime {

class DeviceArray {
 public:
  DeviceArray(py::array x) {
    py::buffer_info buf = x.request();
    size_ = buf.size;
    itemsize_ = buf.itemsize;
    HIP_CALL_THROW(hipMalloc(&x_device_, size_ * itemsize_));
    x_host_ = x.request().ptr;
    HIP_CALL_THROW(hipMemcpy(x_device_, x_host_, size_ * itemsize_, hipMemcpyHostToDevice));
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(DeviceArray&) = delete;

  void UpdateHostNumpyArray() {
    HIP_CALL_THROW(hipMemcpy(x_host_, x_device_, size_ * itemsize_, hipMemcpyDeviceToHost));
  }

  void* ptr() const {
    return x_device_;
  }

  ~DeviceArray() {
    HIP_CALL_THROW(hipFree(x_device_));
  }

 private:
  void* x_device_;
  void* x_host_;
  ssize_t size_;
  ssize_t itemsize_;
};

}  // namespace onnxruntime
