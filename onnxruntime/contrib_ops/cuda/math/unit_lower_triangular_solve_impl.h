// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "core/common/status.h"

namespace onnxruntime::contrib::cuda {

Status UnitLowerTriangularSolveToColumn(cudaStream_t stream, const float* input, float* output,
                                        size_t count, int rows, int columns);
Status UnitLowerTriangularSolveFromColumn(cudaStream_t stream, const float* input, float* output,
                                          size_t count, int rows, int columns);
Status UnitLowerTriangularSolvePointers(cudaStream_t stream, float* a, float* b,
                                        const float** a_pointers, float** b_pointers,
                                        int batch, int rows, int columns);

}  // namespace onnxruntime::contrib::cuda
