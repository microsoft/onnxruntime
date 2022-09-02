// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/util.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

int CeilingDivision(int n, int m) {
  int r = (n - 1) / m + 1;
  return r;
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
