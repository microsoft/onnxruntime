// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/providers/get_execution_providers.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

void CreateInferencePybindStateModule(py::module& m);

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  CreateInferencePybindStateModule(m);
  // move it out of shared method since training build has a little different behavior.
  m.def(
      "get_available_providers", []() -> const std::vector<std::string>& { return GetAvailableExecutionProviderNames(); },
      "Return list of available Execution Providers in this installed version of Onnxruntime. "
      "The order of elements represents the default priority order of Execution Providers "
      "from highest to lowest.");
}
}
}