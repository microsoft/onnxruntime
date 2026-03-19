// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/controlflow/scan.h"

namespace onnxruntime {
namespace contrib {

// Scan variant in the com.microsoft domain that uses variable-length output concatenation.
// Per-iteration scan outputs are collected and concatenated along axis 0, allowing the
// concatenation-axis dimension to vary across iterations.
// An optional output_lengths input at position 0 enables pre-allocation of the final output.
class ScanVarLen final : public Scan<9> {
 public:
  ScanVarLen(const OpKernelInfo& info)
      : Scan<9>(info, /*use_var_len_output=*/true, /*num_non_variadic_inputs=*/1) {
  }
};

}  // namespace contrib
}  // namespace onnxruntime
