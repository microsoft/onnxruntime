// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

struct Fp4MmaConfig {
  int k_split;
  int col_tiles;
};

// Returns the tensor-core GEMV tiling selected for the given shape and device size.
Fp4MmaConfig PickFp4MmaConfig(int n, int k, int sm_count);

}  // namespace onnxruntime::contrib::cuda