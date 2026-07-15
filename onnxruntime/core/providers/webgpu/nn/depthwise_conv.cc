// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/depthwise_conv.h"

#include <string>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Emit "let x{off_idx} = select(0, x[..], in_bounds)" for one input column read.
// The x load index is clamped so out-of-bounds accesses never produce impl-defined
// storage-buffer values that would corrupt the accumulator.
std::string EmitColLoad(const ShaderVariableHelper& x, int off_idx) {
  const std::string idx = std::to_string(off_idx);
  return "      let ic" + idx + " = iw_base + " + idx + ";\n" +
         "      let ic" + idx + "_c = u32(clamp(ic" + idx + ", 0, i32(W_in) - 1));\n" +
         "      let x" + idx + " = select(output_value_t(0), " +
         x.GetByIndices("x_indices_t(batch, iH, ic" + idx + "_c, oc_vec)") +
         ", ic" + idx + " >= 0 && ic" + idx + " < i32(W_in));\n";
}

}  // namespace

Status DepthwiseConv3x3Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x",
                                  ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias |
                                      ShaderUsage::UseIndicesTypeAlias);
  const auto& w = shader.AddInput("w",
                                  ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias |
                                      ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output",
                                        ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias |
                                            ShaderUsage::UseElementTypeAlias |
                                            ShaderUsage::UseShapeAndStride);
  const std::string apply_activation =
      GetActivationSnippet(activation_, "output_value_t", "output_element_t");

  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size") << "\n";

  body << "  let H_out = uniforms.output_shape[1];\n"
       << "  let W_out = uniforms.output_shape[2];\n"
       << "  let C_vec = uniforms.output_shape[3];\n"
       << "  let H_in  = uniforms.x_shape[1];\n"
       << "  let W_in  = uniforms.x_shape[2];\n"
       << "  let tiles_per_row = uniforms.tiles_per_row;\n"
       << "  let oc_vec = global_idx % C_vec;\n"
       << "  var t = global_idx / C_vec;\n"
       << "  let tile_w = t % tiles_per_row;\n"
       << "  t = t / tiles_per_row;\n"
       << "  let oh = t % H_out;\n"
       << "  let batch = t / H_out;\n"
       << "  let ow_base = tile_w * 4u;\n";

  // Load all 9 filter weights into registers once per thread.
  for (int kh = 0; kh < 3; ++kh) {
    for (int kw = 0; kw < 3; ++kw) {
      body << "  let wgt" << kh << kw << " = "
           << w.GetByIndices("w_indices_t(" + std::to_string(kh) + "u, " +
                             std::to_string(kw) + "u, 0u, oc_vec)")
           << ";\n";
    }
  }

  const uint32_t s = stride_;
  body << "  let ih_base = i32(oh * " << s << "u) - i32(uniforms.pads[0]);\n"
       << "  let iw_base = i32(ow_base * " << s << "u) - i32(uniforms.pads[1]);\n"
       << "  var v0 = output_value_t(0);\n"
       << "  var v1 = output_value_t(0);\n"
       << "  var v2 = output_value_t(0);\n"
       << "  var v3 = output_value_t(0);\n";

  // Number of unique input columns needed per row:
  //   stride 1: 6 (cols iw_base + [0..5]) — heavy reuse across 4 tiles
  //   stride 2: 9 (cols iw_base + [0..8]) — some reuse at tile boundaries
  const int num_cols = (s == 1) ? 6 : 9;

  for (int kh = 0; kh < 3; ++kh) {
    body << "  {\n"
         << "    let ih = ih_base + " << kh << ";\n"
         << "    if (ih >= 0 && ih < i32(H_in)) {\n"
         << "      let iH = u32(ih);\n";

    for (int c = 0; c < num_cols; ++c) {
      body << EmitColLoad(x, c);
    }

    // v_oi += sum over kw of x[oi*stride + kw] * wgt[kh][kw]
    for (int oi = 0; oi < 4; ++oi) {
      const int base = oi * static_cast<int>(s);
      body << "      v" << oi << " += x" << (base + 0) << " * wgt" << kh << "0"
           << " + x" << (base + 1) << " * wgt" << kh << "1"
           << " + x" << (base + 2) << " * wgt" << kh << "2;\n";
    }
    body << "    }\n"
         << "  }\n";
  }

  if (has_bias_) {
    const auto& b = shader.AddInput("b",
                                    ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
    body << "  let bias_v = " << b.GetByIndices("oc_vec") << ";\n"
         << "  v0 += bias_v;\n"
         << "  v1 += bias_v;\n"
         << "  v2 += bias_v;\n"
         << "  v3 += bias_v;\n";
  }

  // Apply activation snippet to each of the 4 accumulators.
  // The snippet reads/writes a local variable named `value`.
  for (int oi = 0; oi < 4; ++oi) {
    body << "  {\n"
         << "    var value = v" << oi << ";\n"
         << "    " << apply_activation << "\n"
         << "    v" << oi << " = value;\n"
         << "  }\n";
  }

  // Write 4 outputs with per-column width bounds check.
  body << "  let out_row_base = (batch * H_out + oh) * W_out * C_vec;\n";
  for (int oi = 0; oi < 4; ++oi) {
    const std::string oi_s = std::to_string(oi);
    body << "  {\n"
         << "    let ow = ow_base + " << oi << "u;\n"
         << "    if (ow < W_out) { "
         << output.SetByOffset("out_row_base + ow * C_vec + oc_vec", "v" + oi_s)
         << " }\n"
         << "  }\n";
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
