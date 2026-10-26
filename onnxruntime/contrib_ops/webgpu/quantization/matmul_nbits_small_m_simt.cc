// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_small_m_simt.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

namespace {

constexpr uint32_t kBN = 8;
constexpr uint32_t kBK = 512;
constexpr uint32_t kWorkgroupSize = 128;
// Nine vec4<f16> slots per block32 avoid a 64-byte shared-memory bank stride.
constexpr uint32_t kABlockStride = 9;
constexpr uint32_t kARowStride = kBK / 32 * kABlockStride;

uint32_t MatrixRows(const Tensor* a) {
  // MatMulComputeHelper flattens shared-weight batches into M; tile each matrix separately.
  const auto& shape = a->Shape();
  return shape.NumDimensions() < 2 ? 1u : narrow<uint32_t>(shape[shape.NumDimensions() - 2]);
}

uint32_t TileRows(uint32_t M) {
  return M <= 2 ? 2u : M <= 4 ? 4u
                              : 8u;
}

class MatMulNBitsSmallMSimtProgram final : public Program<MatMulNBitsSmallMSimtProgram> {
 public:
  MatMulNBitsSmallMSimtProgram(uint32_t bm, uint32_t a_components, bool has_zero_points, bool has_bias)
      : Program{"MatMulNBitsSmallMSimt"},
        bm_{bm},
        a_components_{a_components},
        has_zero_points_{has_zero_points},
        has_bias_{has_bias} {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulNBitsSmallMSimtProgram);

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const auto* zero_points = has_zero_points_ ? &shader.AddInput("zero_points", ShaderUsage::None) : nullptr;
    const auto* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::None) : nullptr;
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    constexpr uint32_t kKLanes = kWorkgroupSize / kBN;
    shader.AdditionalImplementation()
        << "const BM = " << bm_ << "u;\nconst BN = " << kBN << "u;\nconst BK = " << kBK
        << "u;\nconst K_LANES = " << kKLanes << "u;\n"
        << "var<workgroup> a_stage: array<vec4<f16>, " << bm_ * kARowStride << ">;\n"
        << "var<workgroup> partial: array<f32, " << bm_ * kWorkgroupSize << ">;\n";

    auto& body = shader.MainFunctionBody();
    body << R"(
  if (workgroup_idx >= uniforms.num_n_tile * uniforms.batch_count) { return; }
  let batch = workgroup_idx / uniforms.num_n_tile;
  let n = (workgroup_idx % uniforms.num_n_tile) * BN + local_idx / K_LANES;
  let k_lane = local_idx % K_LANES;
  let b_base = uniforms.weight_index * uniforms.N * uniforms.blocks;
  let scale_base = uniforms.weight_index * uniforms.N * uniforms.blocks;
)";
    // FP32 scalar accumulators avoid both FP16 overflow and private-memory arrays.
    for (uint32_t row = 0; row < bm_; ++row) {
      body << "  var acc" << row << " = 0.0f;\n";
    }
    body << R"(
  for (var kb = 0u; kb < uniforms.K; kb += BK) {
    for (var i = local_idx; i < BM * (BK / 4u); i += 128u) {
      let row = i / (BK / 4u);
      let k = kb + (i % (BK / 4u)) * 4u;
      var value = vec4<f16>(0.0h);
)";
    if (a_components_ == 4) {
      body << "      if (row < uniforms.M && k < uniforms.K) {\n"
           << "        value = " << a.GetByOffset("(batch * uniforms.M + row) * (uniforms.K / 4u) + k / 4u")
           << ";\n      }\n";
    } else {
      body << "      if (row < uniforms.M) {\n";
      for (uint32_t component = 0; component < 4; ++component) {
        const std::string k = "k + " + std::to_string(component) + "u";
        body << "        if (" << k << " < uniforms.K) { value[" << component << "] = "
             << a.GetByOffset("(batch * uniforms.M + row) * uniforms.K + " + k) << "; }\n";
      }
      body << "      }\n";
    }
    body << "      a_stage[row * " << kARowStride << "u + (i % (BK / 4u)) / 8u * "
         << kABlockStride << "u + i % 8u] = value;\n";
    body << R"(
    }
    workgroupBarrier();
    for (var block = k_lane; block < BK / 32u; block += K_LANES) {
      let k = kb + block * 32u;
      if (n < uniforms.N && k < uniforms.K) {
        let scale_block = k / 32u;
)";
    body << "        let scale = " << scales.GetByOffset("scale_base + n * uniforms.blocks + scale_block") << ";\n";
    if (has_zero_points_) {
      body << "        let zero_index = (uniforms.weight_index * uniforms.N + n) * uniforms.zero_stride + scale_block;\n"
           << "        let zero_word = " << zero_points->GetByOffset("zero_index / 8u") << ";\n"
           << "        let zero = f16((zero_word >> ((zero_index % 8u) * 4u)) & 15u);\n";
    } else {
      body << "        let zero = 8.0h;\n";
    }
    body << "        let packed = " << b.GetByOffset("b_base + n * uniforms.blocks + k / 32u") << ";\n";
    for (uint32_t word = 0; word < 4; ++word) {
      body << "        {\n          let bits = packed[" << word << "];\n"
           << "          let lo = vec4<f16>(unpack4xU8(bits & 0x0F0F0F0Fu)) - vec4<f16>(zero);\n"
           << "          let hi = vec4<f16>(unpack4xU8((bits >> 4u) & 0x0F0F0F0Fu)) - vec4<f16>(zero);\n"
           << "          let w0_half = vec4<f16>(lo.x, hi.x, lo.y, hi.y) * scale;\n"
           << "          let w1_half = vec4<f16>(lo.z, hi.z, lo.w, hi.w) * scale;\n"
           << "          let w0 = vec4<f32>(w0_half);\n          let w1 = vec4<f32>(w1_half);\n";
      for (uint32_t row = 0; row < bm_; ++row) {
        const uint32_t offset = row * kARowStride + word * 2;
        body << "          acc" << row << " += dot(vec4<f32>(a_stage[block * " << kABlockStride << "u + " << offset
             << "u]), w0) + dot(vec4<f32>(a_stage[block * " << kABlockStride << "u + " << offset + 1 << "u]), w1);\n";
      }
      body << "        }\n";
    }
    body << R"(
      }
    }
    workgroupBarrier();
  }
)";
    for (uint32_t row = 0; row < bm_; ++row) {
      body << "  partial[" << row * kWorkgroupSize << "u + local_idx] = acc" << row << ";\n";
    }
    body << R"(
  workgroupBarrier();
  for (var i = local_idx; i < uniforms.M * BN; i += 128u) {
    let row = i / BN;
    let col = (workgroup_idx % uniforms.num_n_tile) * BN + i % BN;
    if (col < uniforms.N) {
      var sum = 0.0f;
      for (var lane = 0u; lane < K_LANES; lane++) {
        sum += partial[row * 128u + (i % BN) * K_LANES + lane];
      }
)";
    if (has_bias_) {
      body << "      sum += f32(" << bias->GetByOffset("uniforms.weight_index * uniforms.N + col") << ");\n";
    }
    body << "      " << output.SetByOffset("(batch * uniforms.M + row) * uniforms.N + col", "f16(sum)")
         << "\n    }\n  }\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"blocks", ProgramUniformVariableDataType::Uint32},
      {"zero_stride", ProgramUniformVariableDataType::Uint32},
      {"num_n_tile", ProgramUniformVariableDataType::Uint32},
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"weight_index", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t bm_;
  uint32_t a_components_;
  bool has_zero_points_;
  bool has_bias_;
};

}  // namespace

bool CanApplySmallMSimtMatMulNBits(const Tensor* a, uint32_t N, uint32_t K,
                                   uint32_t block_size, int64_t nbits,
                                   bool has_weight_idx_indirect, uint32_t override_M,
                                   const onnxruntime::webgpu::ComputeContext& context) {
  const uint32_t M = MatrixRows(a);
  const auto& limits = context.DeviceLimits();
  const uint32_t shared_bytes = TileRows(M) * (kARowStride * 8 + kWorkgroupSize * 4);
  return M >= 2 && M <= 8 && K >= 4096 && N >= 4096 &&
         nbits == 4 && block_size == 32 && !has_weight_idx_indirect && override_M == 0 &&
         a->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
         context.HasFeature(wgpu::FeatureName::ShaderF16) &&
         limits.maxComputeWorkgroupSizeX >= kWorkgroupSize &&
         limits.maxComputeInvocationsPerWorkgroup >= kWorkgroupSize &&
         limits.maxComputeWorkgroupStorageSize >= shared_bytes;
}

Status ApplySmallMSimtMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  const Tensor* zero_points, const Tensor* bias,
                                  uint32_t N, uint32_t K, uint32_t weight_index,
                                  onnxruntime::webgpu::ComputeContext& context, Tensor* y) {
  const uint32_t M = MatrixRows(a);
  const uint32_t batch_count = narrow<uint32_t>(a->Shape().Size() / K / M);
  const uint32_t bm = TileRows(M);
  const uint32_t blocks = CeilDiv(K, 32u);
  const uint32_t zero_stride = CeilDiv(blocks, 2u) * 2u;
  const uint32_t a_components = K % 4 == 0 ? 4u : 1u;
  const uint32_t n_tiles = CeilDiv(N, kBN);
  MatMulNBitsSmallMSimtProgram program{bm, a_components, zero_points != nullptr, bias != nullptr};
  program.SetWorkgroupSize(kWorkgroupSize);
  program.SetDispatchGroupSize(n_tiles, batch_count);
  program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(a_components)});
  program.AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16});
  program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
  if (zero_points != nullptr) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {CeilDiv(zero_points->Shape().Size(), int64_t{4})}, 4});
  }
  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank});
  program.AddUniformVariables({M, N, K, blocks, zero_stride, n_tiles, batch_count, weight_index});
  program.CacheHint(bm, a_components, zero_points != nullptr, bias != nullptr);
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
