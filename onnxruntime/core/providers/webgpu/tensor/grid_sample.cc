// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/grid_sample.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status GridSampleProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& grid = shader.AddInput("grid", ShaderUsage::UseUniform);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform);

  // gs_denormalize: specialized per align_corners
  if (align_corners_) {
    shader.AdditionalImplementation()
        << "fn gs_denormalize(n: f32, length: u32) -> f32 {\n"
        << "  return (n + 1.0) * 0.5 * f32(length - 1u);\n"
        << "}\n";
  } else {
    shader.AdditionalImplementation()
        << "fn gs_denormalize(n: f32, length: u32) -> f32 {\n"
        << "  return ((n + 1.0) * f32(length) - 1.0) * 0.5;\n"
        << "}\n";
  }

  // gs_reflect: only needed for reflection padding mode
  if (padding_mode_ == 2) {
    shader.AdditionalImplementation()
        << "fn gs_reflect(v: f32, v_min: f32, v_max: f32) -> f32 {\n"
        << "  var fv = v;\n"
        << "  let range = v_max - v_min;\n"
        << "  if (fv < v_min) {\n"
        << "    let dv = v_min - fv;\n"
        << "    let n = i32(dv / range);\n"
        << "    let r = dv - f32(n) * range;\n"
        << "    fv = select(v_max - r, v_min + r, n % 2 == 0);\n"
        << "  } else if (fv > v_max) {\n"
        << "    let dv = fv - v_max;\n"
        << "    let n = i32(dv / range);\n"
        << "    let r = dv - f32(n) * range;\n"
        << "    fv = select(v_min + r, v_max - r, n % 2 == 0);\n"
        << "  }\n"
        << "  return fv;\n"
        << "}\n";
  }

  // gs_cubic_coeffs: only needed for bicubic mode
  if (mode_ == 2) {
    shader.AdditionalImplementation()
        << "fn gs_cubic_coeffs(t: f32) -> vec4<f32> {\n"
        << "  let ax = abs(t);\n"
        << "  let a = -0.75f;\n"
        << "  let c0 = ((a * (ax + 1.0) - 5.0 * a) * (ax + 1.0) + 8.0 * a) * (ax + 1.0) - 4.0 * a;\n"
        << "  let c1 = ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0;\n"
        << "  let c2 = ((a + 2.0) * (1.0 - ax) - (a + 3.0)) * (1.0 - ax) * (1.0 - ax) + 1.0;\n"
        << "  let c3 = ((a * (2.0 - ax) - 5.0 * a) * (2.0 - ax) + 8.0 * a) * (2.0 - ax) - 4.0 * a;\n"
        << "  return vec4<f32>(c0, c1, c2, c3);\n"
        << "}\n";
  }

  // gs_pixel: pixel fetch helper, specialized per padding_mode (and align_corners for reflection)
  // Returns f32 always; caller casts to output type.
  shader.AdditionalImplementation()
      << "fn gs_pixel(img_base: u32, r: i32, col: i32) -> f32 {\n";

  if (padding_mode_ == 0) {
    // zeros: out-of-bounds -> 0
    shader.AdditionalImplementation()
        << "  if (r < 0 || r >= i32(uniforms.H_in) || col < 0 || col >= i32(uniforms.W_in)) {\n"
        << "    return 0.0;\n"
        << "  }\n"
        << "  return f32(" << x.GetByOffset("img_base + u32(r) * uniforms.W_in + u32(col)") << ");\n";
  } else if (padding_mode_ == 1) {
    // border: clamp to nearest edge
    shader.AdditionalImplementation()
        << "  let cr = u32(clamp(r, 0, i32(uniforms.H_in) - 1));\n"
        << "  let cc = u32(clamp(col, 0, i32(uniforms.W_in) - 1));\n"
        << "  return f32(" << x.GetByOffset("img_base + cr * uniforms.W_in + cc") << ");\n";
  } else {
    // reflection: oscillating reflect, bounds depend on align_corners
    if (align_corners_) {
      // reflect within [0, length-1]
      shader.AdditionalImplementation()
          << "  let rr = i32(gs_reflect(f32(r),   0.0, f32(uniforms.H_in) - 1.0));\n"
          << "  let cc = i32(gs_reflect(f32(col), 0.0, f32(uniforms.W_in) - 1.0));\n";
    } else {
      // reflect within [-0.5, length-0.5]
      shader.AdditionalImplementation()
          << "  let rr = i32(gs_reflect(f32(r),   -0.5, f32(uniforms.H_in) - 0.5));\n"
          << "  let cc = i32(gs_reflect(f32(col), -0.5, f32(uniforms.W_in) - 0.5));\n";
    }
    shader.AdditionalImplementation()
        << "  let ur = u32(clamp(rr, 0, i32(uniforms.H_in) - 1));\n"
        << "  let uc = u32(clamp(cc, 0, i32(uniforms.W_in) - 1));\n"
        << "  return f32(" << x.GetByOffset("img_base + ur * uniforms.W_in + uc") << ");\n";
  }
  shader.AdditionalImplementation() << "}\n";

  // Main function body
  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
       // Decode global_idx -> (n, c, h_out, w_out)
       << "  let HW_out  = uniforms.H_out * uniforms.W_out;\n"
       << "  let CHW_out = uniforms.C * HW_out;\n"
       << "  let n       = global_idx / CHW_out;\n"
       << "  let rem     = global_idx % CHW_out;\n"
       << "  let c       = rem / HW_out;\n"
       << "  let hw      = rem % HW_out;\n"
       << "  let h_out   = hw / uniforms.W_out;\n"
       << "  let w_out   = hw % uniforms.W_out;\n"
       // Read normalized grid coords: grid is [N, H_out, W_out, 2], gx=x-coord (W), gy=y-coord (H)
       << "  let grid_base = ((n * uniforms.H_out + h_out) * uniforms.W_out + w_out) * 2u;\n"
       << "  let gx = f32(" << grid.GetByOffset("grid_base") << ");\n"
       << "  let gy = f32(" << grid.GetByOffset("grid_base + 1u") << ");\n"
       // Denormalize to image-space coordinates
       << "  let px = gs_denormalize(gx, uniforms.W_in);\n"
       << "  let py = gs_denormalize(gy, uniforms.H_in);\n"
       // Base flat offset for this (n, c) plane of X: [N, C, H_in, W_in]
       << "  let img_base = (n * uniforms.C + c) * uniforms.H_in * uniforms.W_in;\n";

  if (mode_ == 1) {
    // nearest: round to nearest integer
    body << "  let rx = i32(round(px));\n"
         << "  let ry = i32(round(py));\n"
         << "  let result = gs_pixel(img_base, ry, rx);\n";
  } else if (mode_ == 0) {
    // bilinear: 4-neighbor weighted interpolation
    body << "  let x1 = i32(floor(px));\n"
         << "  let y1 = i32(floor(py));\n"
         << "  let x2 = x1 + 1;\n"
         << "  let y2 = y1 + 1;\n"
         << "  let dx1 = px - f32(x1);\n"
         << "  let dx2 = 1.0 - dx1;\n"
         << "  let dy1 = py - f32(y1);\n"
         << "  let dy2 = 1.0 - dy1;\n"
         << "  let p11 = gs_pixel(img_base, y1, x1);\n"
         << "  let p12 = gs_pixel(img_base, y1, x2);\n"
         << "  let p21 = gs_pixel(img_base, y2, x1);\n"
         << "  let p22 = gs_pixel(img_base, y2, x2);\n"
         << "  let result = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (dx2 * p21 + dx1 * p22);\n";
  } else {
    // bicubic: 4x4 neighborhood with Robert Keys coefficients (alpha=-0.75)
    body << "  let x0 = i32(floor(px)) - 1;\n"
         << "  let y0 = i32(floor(py)) - 1;\n"
         << "  let dx = px - f32(x0 + 1);\n"
         << "  let dy = py - f32(y0 + 1);\n"
         << "  let cx = gs_cubic_coeffs(dx);\n"
         << "  let cy = gs_cubic_coeffs(dy);\n"
         << "  var rows: vec4<f32>;\n"
         << "  for (var i = 0i; i < 4i; i++) {\n"
         << "    let row = y0 + i;\n"
         << "    rows[i] = cx[0] * gs_pixel(img_base, row, x0    )\n"
         << "            + cx[1] * gs_pixel(img_base, row, x0 + 1)\n"
         << "            + cx[2] * gs_pixel(img_base, row, x0 + 2)\n"
         << "            + cx[3] * gs_pixel(img_base, row, x0 + 3);\n"
         << "  }\n"
         << "  let result = dot(cy, rows);\n";
  }

  body << "  " << y.SetByOffset("global_idx", "x_value_t(result)") << "\n";

  return Status::OK();
}

GridSample::GridSample(const OpKernelInfo& info) : WebGpuKernel(info) {
  // Accept both opset-16 names ("bilinear"/"bicubic") and opset-20+ names ("linear"/"cubic")
  std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
  if (mode_str == "bilinear" || mode_str == "linear") {
    mode_ = 0;
  } else if (mode_str == "nearest") {
    mode_ = 1;
  } else if (mode_str == "bicubic" || mode_str == "cubic") {
    mode_ = 2;
  } else {
    ORT_THROW("GridSample: unsupported mode \"", mode_str, "\"");
  }

  std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
  if (padding_mode_str == "zeros") {
    padding_mode_ = 0;
  } else if (padding_mode_str == "border") {
    padding_mode_ = 1;
  } else if (padding_mode_str == "reflection") {
    padding_mode_ = 2;
  } else {
    ORT_THROW("GridSample: unsupported padding_mode \"", padding_mode_str, "\"");
  }

  align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
}

Status GridSample::ComputeInternal(ComputeContext& context) const {
  const auto* X = context.Input<Tensor>(0);
  const auto* grid = context.Input<Tensor>(1);

  const auto& X_shape = X->Shape();
  const auto& grid_shape = grid->Shape();

  ORT_RETURN_IF_NOT(X_shape.NumDimensions() == 4, "X must be 4-D for opset 16");
  ORT_RETURN_IF_NOT(grid_shape.NumDimensions() == 4, "grid must be 4-D");
  ORT_RETURN_IF_NOT(grid_shape[3] == 2, "grid last dimension must be 2");

  const int64_t N = X_shape[0];
  const int64_t C = X_shape[1];
  const int64_t H_in = X_shape[2];
  const int64_t W_in = X_shape[3];

  ORT_RETURN_IF_NOT(grid_shape[0] == N, "grid batch size must match X batch size");

  const int64_t H_out = grid_shape[1];
  const int64_t W_out = grid_shape[2];

  TensorShape Y_shape{N, C, H_out, W_out};
  auto* Y = context.Output(0, Y_shape);

  const uint32_t output_size = onnxruntime::narrow<uint32_t>(Y_shape.Size());
  if (output_size == 0) {
    return Status::OK();
  }

  GridSampleProgram program{mode_, padding_mode_, align_corners_};
  program
      .AddInputs({{X, ProgramTensorMetadataDependency::TypeAndRank},
                  {grid, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({Y, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(mode_, padding_mode_, static_cast<int>(align_corners_))
      .AddUniformVariables({{output_size},
                            {static_cast<uint32_t>(C)},
                            {static_cast<uint32_t>(H_in)},
                            {static_cast<uint32_t>(W_in)},
                            {static_cast<uint32_t>(H_out)},
                            {static_cast<uint32_t>(W_out)}});

  return context.RunProgram(program);
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GridSample,
    kOnnxDomain,
    16, 19,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    GridSample);

}  // namespace webgpu
}  // namespace onnxruntime
