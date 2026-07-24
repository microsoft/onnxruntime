// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include "core/framework/op_kernel.h"

namespace onnxruntime::cpu::registration_internal {

template <size_t N>
consteval bool IsKernelRegistrationTableValid(
    const BuildKernelCreateInfoFn (&function_table)[N],
    BuildKernelCreateInfoFn disabled_entry) {
  if (disabled_entry == nullptr || function_table[0] != disabled_entry) {
    return false;
  }

  // Reduced-operator builds may replace multiple registrations with the disabled entry.
  for (const auto build_kernel_create_info : function_table) {
    if (build_kernel_create_info == nullptr) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime::cpu::registration_internal
