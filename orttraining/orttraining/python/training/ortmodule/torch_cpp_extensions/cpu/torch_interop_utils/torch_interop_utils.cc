// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ctx_pool.h"
#include "custom_function_fw.h"
#include "custom_function_bw.h"

size_t get_custom_function_forward_runner() { return reinterpret_cast<size_t>(&custom_function_forward_runner); }
size_t get_custom_function_backward_runner() { return reinterpret_cast<size_t>(&custom_function_backward_runner); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clear_all_grad_fns", &clear_all_grad_fns, "Clear all grad_fn shared pointer references.");
  m.def("get_custom_function_forward_runner", &get_custom_function_forward_runner, "Get custom function forward runner.");
  m.def("get_custom_function_backward_runner", &get_custom_function_backward_runner, "Get custom function backward runner.");
}
