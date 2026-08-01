// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class MatMulBnb4Program final : public Program<MatMulBnb4Program> {
 public:
  MatMulBnb4Program(int64_t quant_type, int output_number)
      : Program{"MatMulBnb4"}, quant_type_{quant_type}, output_number_{output_number} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t quant_type_;
  // Number of output rows (M dimension) computed by each invocation. Reusing each dequantized
  // weight across multiple rows amortizes the relatively expensive dequantization.
  int output_number_;
};

// Shared-memory tiled variant used for larger M. Two schemes are selected by `components`:
//   * components == 4 (N % 4 == 0 && K % 4 == 0): a vec4 GEMM tiling modeled on the WebGPU
//     matmul_packed vec4 kernel. An 8x8 workgroup computes a 32x32 output block (each invocation
//     4 rows x one vec4 of columns); A is read vec4 along K, B is dequantized into a vec4 of 4
//     output columns, and the output is stored vec4.
//   * components == 1 (fallback): a scalar 16x16 tile, one output element per invocation.
class MatMulBnb4TileProgram final : public Program<MatMulBnb4TileProgram> {
 public:
  // Scalar-fallback tile size and the M threshold above which a tiled kernel is used.
  static constexpr int kTileSize = 16;
  // Vec4 GEMM tiling: 8x8 workgroup, 4 rows per invocation -> a 32x32 output block per workgroup.
  static constexpr int kGemmWorkgroup = 8;
  static constexpr int kGemmRowsPerThread = 4;
  static constexpr int kGemmTile = kGemmWorkgroup * kGemmRowsPerThread;  // 32

  MatMulBnb4TileProgram(int64_t quant_type, int components)
      : Program{"MatMulBnb4Tile"}, quant_type_{quant_type}, components_{components} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t quant_type_;
  // Vectorization scheme selector: 4 = vec4 GEMM tiling, 1 = scalar fallback (see class comment).
  int components_;
};

class MatMulBnb4 final : public WebGpuKernel {
 public:
  MatMulBnb4(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    block_size_ = info.GetAttr<int64_t>("block_size");
    quant_type_ = info.GetAttr<int64_t>("quant_type");
    ORT_ENFORCE(K_ > 0, "K must be positive, got ", K_);
    ORT_ENFORCE(N_ > 0, "N must be positive, got ", N_);
    ORT_ENFORCE(block_size_ > 0, "block_size must be positive, got ", block_size_);
    ORT_ENFORCE(quant_type_ == 0 || quant_type_ == 1,
                "Invalid quant_type, only 0 (FP4) and 1 (NF4) are supported.");
    // Only the forward case (transB=1) is supported on WebGPU; nodes with transB=0 are
    // filtered out and fall back to CPU in WebGpuExecutionProvider::GetCapability.
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t quant_type_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
