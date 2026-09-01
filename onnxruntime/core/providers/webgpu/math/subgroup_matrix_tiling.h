// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <cstdint>

namespace onnxruntime {
namespace webgpu {

// Per-workgroup output tiling for one subgroup-matrix GEMM problem: the tile shape
// and split-K factor chosen by a vendor-specific policy (SubgroupMatrixTilingSelector
// in subgroup_matrix_matmul.h). The subgroup-matrix shape is not selected here - it
// comes from the device config the kernel was compiled for - but it constrains this:
// tile_m and tile_n must be whole multiples of the subgroup-matrix M and N.
//
// Which values are legal is a property of the kernel, not of this type, so there is
// no validity check here. The split-K partials live in workgroup memory, so a kernel
// that stages more there (the Conv kernel also holds its im2col A tile) affords less
// split-K for the same tile; the bounds live with the policy, in IsTilingValid in
// vendor/intel/math/subgroup_matrix_tiling_selector.cc.
//
// Deliberately its own header, and deliberately dependency-free beyond <cstdint>:
// some includers must not reach the program definitions (and the build-generated WGSL
// template headers behind them). Keep it that way; the includers that need it say why.
struct SubgroupMatrixTiling {
  uint32_t tile_m;   // output rows per workgroup
  uint32_t tile_n;   // output cols per workgroup
  uint32_t split_k;  // subgroups cooperating along K (1 = no split); also fixes the
                     // workgroup size, which is subgroup_size * split_k
};

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
