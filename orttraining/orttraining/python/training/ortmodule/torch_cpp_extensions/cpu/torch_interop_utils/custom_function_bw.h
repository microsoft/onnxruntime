// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

std::vector<PyObject*> custom_function_backward_runner(const char* func_name_char,
                                                       void* callback,
                                                       const std::vector<int64_t>& requires_grad_flags,
                                                       const std::vector<int64_t>& tensor_type_flags,
                                                       const bool is_training_mode,
                                                       const std::vector<int64_t>& inplace_map,
                                                       const char* kernel_invoke_id_char,
                                                       const bool safe_run_mode_enabled,
                                                       const std::vector<PyObject*>& args);
