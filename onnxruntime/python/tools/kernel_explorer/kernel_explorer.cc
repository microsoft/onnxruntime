// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

static py::module::module_def _kernel_explorer_module_def;

bool TuningInfo::collect_enabled_{true};
std::vector<TuningResults> TuningInfo::collected_tuning_results_ = {};
std::optional<int> TuningInfo::max_tuning_duration_ms_ = {};

py::module GetKernelExplorerModule() {
  static pybind11::module_ m = []() {
    auto tmp = pybind11::module_::create_extension_module(
        "_kernel_explorer", "", &_kernel_explorer_module_def);
    tmp.dec_ref();
    return tmp;
  }();
  return m;
}

PYBIND11_PLUGIN_IMPL(_kernel_explorer) {
  PYBIND11_CHECK_PYTHON_VERSION;
  PYBIND11_ENSURE_INTERNALS_READY;
  return GetKernelExplorerModule().ptr();
}

KE_REGISTER(m) {
  py::class_<DeviceArray>(m, "DeviceArray")
      .def(py::init<py::array>())
      .def(py::init<size_t, ssize_t, ssize_t>())
      .def("UpdateHostNumpyArray", &DeviceArray::UpdateHostNumpyArray)
      .def("UpdateDeviceArray", &DeviceArray::UpdateDeviceArray);

  m.def("collect_tuning_results", TuningInfo::EnableCollect);

  m.def("max_tuning_duration_ms", TuningInfo::SetMaxTuningDurationMs);

  m.def("get_collected_tuning_results", []() {
    py::list ret;
    for (const auto& trs : TuningInfo::GetCollectedTuningResults()) {
      py::dict py_trs;
      py_trs["ep"] = trs.ep;
      py_trs["results"] = trs.results;
      py_trs["validators"] = trs.validators;
      ret.append(std::move(py_trs));
    }
    return ret;
  });

  // clang-format ill-format the following code below version 18
  // clang-format off
  m.def("is_composable_kernel_available", []() {
#ifdef USE_COMPOSABLE_KERNEL
    return true;
#else
    return false;
#endif
  });

  m.def("is_hipblaslt_available", []() {
#ifdef USE_HIPBLASLT
    return true;
#else
    return false;
#endif
  });

  m.def("is_float8_available", []() {
#ifndef DISABLE_FLOAT8_TYPES
    return true;
#else
    return false;
#endif
  });
  // clang-format on
}

}  // namespace onnxruntime
