// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm_utils.h"

#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/matmul_utils.h"

namespace onnxruntime {
namespace webgpu {

// Some helper functions to handle the bias for GEMM and MatMul,
// which are used in the MatMulWriteFnSource function.
namespace {

void HandleMaybeHaveBiasForGEMM(ShaderHelper& shader,
                                const ShaderVariableHelper& output,
                                const ShaderVariableHelper* c,
                                int c_components,
                                int output_components,
                                bool c_is_scalar) {
  shader.AdditionalImplementation() << "    let coords = vec2(u32(row), u32(colIn));\n";

  if (c != nullptr) {
    shader.AdditionalImplementation() << "    value += output_element_t(uniforms.beta) * ";
    // We can be allowed to use broadcasting only when both components are equal.
    // There is only one case for c_components_ is not equal output_components.
    // I.g. the former is `1` and the latter is `4`.
    // That means the shape of `c` is either {M,1} or {1,1}
    if (c_components == output_components) {
      shader.AdditionalImplementation() << "output_value_t("
                                        << c->GetByOffset(c->BroadcastedIndicesToOffset("vec2(u32(row), u32(colIn))", output)) << ");\n";
    } else if (c_is_scalar) {
      shader.AdditionalImplementation() << "output_value_t(" << c->GetByOffset("0") << ");\n";
    } else {
      shader.AdditionalImplementation() << "output_value_t(" << c->GetByOffset("row") << ");\n";
    }
  }
  shader.AdditionalImplementation() << output.SetByIndices("coords", "value") << "\n";
}

void HandleMaybeBiasForMatMul(ShaderHelper& shader,
                              const ShaderVariableHelper& output,
                              const ShaderVariableHelper* bias,
                              std::string activation_snippet,
                              bool is_channels_last) {
  shader.AdditionalImplementation() << "    let coords = vec3(u32(batch), u32(row), u32(colIn));\n";
  if (bias != nullptr) {
    shader.AdditionalImplementation() << "    value = value + output_value_t(" << (is_channels_last ? bias->GetByOffset("colIn") : bias->GetByOffset("row")) << ");\n";
  }
  shader.AdditionalImplementation() << "    " << activation_snippet << "\n"
                                    << "    " << output.SetByIndices("coords", "value") << "\n";
}

void HandleMatMulWithSplitK(
    ShaderHelper& shader,
    bool is_gemm,
    const ShaderVariableHelper& output,
    ProgramVariableDataType output_variable_type) {
  if (is_gemm) {
    shader.AdditionalImplementation() << "    let coords = vec2(u32(row), u32(colIn));";
  } else {
    shader.AdditionalImplementation() << "    let coords = vec3(u32(batch), u32(row), u32(colIn));\n";
  }

  // With Split-K, the final output will be the sum of the sub-outputs from multiple workgroups,
  // so we must add them with atomic built-in functions. Because currently WebGPU doesn't support
  // atomic built-in functions on `f32` or `f16`, we implement the `atomicAdd` on `f32` and `f16`
  // with `atomicLoad` and `atomicCompareExchangeWeak`:
  // 1. Get `old_output_i32` from `output[offset]` with `atomicLoad`.
  // 2. Convert `old_output_i32` into `f32` (`old_output_f32`) or `vec2h` (`old_output_vec2h`).
  // 3. Add incoming `value` into `old_output_f32` or `old_output_vec2h`.
  // 4. Convert the result of step 3 into `i32` values.
  // 5. Try assigning the result of step 4 into `output[offset]` with `atomicCompareExchangeWeak`
  //    and `old_output_i32`. The assignment will fail if at this time `output[offset]` is not
  //    equal to `old_output_i32` (it is updated in another invocation). If the assignment fails
  //    we have to go to step 1 and repeat all the above steps.
  switch (output_variable_type) {
    case ProgramVariableDataType::Float32x4: {
      shader.AdditionalImplementation() << R"(
    let offset0 = i2o_output(coords) * 4u;
    for (var i = 0u; i < 4u; i++) {
        let offset = offset0 + i;
)";
      shader.AdditionalImplementation() << GenerateAtomicAddNonIntegerCode(output, "offset", "f32", "value[i]") << R"(
    }
)";
      break;
    }
    case ProgramVariableDataType::Float16x4: {
      shader.AdditionalImplementation() << R"(
    let offset0 = i2o_output(coords) * 2u;
    var vec2h_values : array<vec2h, 2>;
    vec2h_values[0] = value.xy;
    vec2h_values[1] = value.zw;
    for (var i = 0u; i < 2u; i++) {
        let offset = offset0 + i;
)";
      shader.AdditionalImplementation() << GenerateAtomicAddNonIntegerCode(output, "offset", "vec2h", "vec2h_values[i]") << R"(
    }
)";
      break;
    }
    default:
      break;
  }
}

// Compute `logical_workgroup_id` and `logical_global_id` because the dispatch workgroup size in
// `ProgramBase.SetDispatchGroupSize()` may be normalized in
// `ProgramManager::NormalizeDispatchGroupSize()`. In the shader we should always use
// `logical_workgroup_id` and `logical_global_id` instead of `workgroup_id` and `global_id`.
void InitializeLogicalWorkgroupIDAndGlobalID(ShaderHelper& shader) {
  shader.MainFunctionBody()
      << "  let logical_workgroup_id_z = workgroup_idx / (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y);\n"
      << "  let logical_workgroup_id_y = (workgroup_idx % (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y)) / uniforms.logical_dispatch_x;\n"
      << "  let logical_workgroup_id_x = (workgroup_idx % (uniforms.logical_dispatch_x * uniforms.logical_dispatch_y)) % uniforms.logical_dispatch_x;\n"
      << "  let logical_workgroup_id = vec3u(logical_workgroup_id_x, logical_workgroup_id_y, logical_workgroup_id_z);\n"
      << "  const workgroupSize = vec3u(workgroup_size_x, workgroup_size_y, workgroup_size_z);\n"
      << "  let logical_global_id = logical_workgroup_id * workgroupSize + local_id;\n";
}

}  // namespace

void MatMulReadFnSource(ShaderHelper& shader,
                        std::string_view function_name,
                        const ShaderVariableHelper& input,
                        const std::string& input_name,
                        const ShaderIndicesHelper* batch_dims,
                        std::string_view rows,
                        std::string_view components_per_row,
                        bool transpose) {
  const int components = input.NumComponents();
  const std::string data_type = "output_element_t";
  const std::string type_string = MakeScalarOrVectorType(components, data_type);

  shader.AdditionalImplementation()
      << "fn " << function_name << "(batch: i32, row: i32, colIn: i32 "
      << (batch_dims
              ? ", batch_indices: batch_dims_indices_t"
              : "")
      << ") -> " << type_string << " {\n "
      << "    var value = " << type_string << "(0);\n"
      << "    let col = colIn * " << components << ";\n";
  if (transpose) {
    shader.AdditionalImplementation() << "    if(row < i32(" << components_per_row << ") && col < i32(" << rows << ")) {\n";
  } else {
    shader.AdditionalImplementation() << "    if(row < i32(" << rows << ") && col < i32(" << components_per_row << ")) {\n";
  }

  const std::string input_indices = input_name + "_indices";
  shader.AdditionalImplementation() << "        var " << input_indices << ": " << input_name << "_indices_t" << ";\n";

  if (batch_dims) {
    shader.AdditionalImplementation() << ConvertOutputBatchIndicesToInputBatchIndices(input_name, input, input.Rank() - 2, batch_dims ? batch_dims->Rank() : 0, " batch_indices ") << "\n";
  }

  shader.AdditionalImplementation() << input.IndicesSet(input_indices, input.Rank() - 2, "u32(row)") << "\n"
                                    << input.IndicesSet(input_indices, input.Rank() - 1, "u32(colIn)") << "\n"
                                    << "        value = " << input.GetByIndices(input_indices) << ";\n"
                                    << "    }\n"
                                    << "    return value;\n"
                                    << "}\n\n";
}

void MatMulReadFnSource(ShaderHelper& shader,
                        const ShaderVariableHelper& a,
                        const ShaderVariableHelper& b,
                        const ShaderIndicesHelper* batch_dims,
                        bool transA,
                        bool transB) {
  MatMulReadFnSource(shader, "mm_readA", a, "a", batch_dims, "uniforms.dim_a_outer", "uniforms.dim_inner", transA);
  MatMulReadFnSource(shader, "mm_readB", b, "b", batch_dims, "uniforms.dim_inner", "uniforms.dim_b_outer", transB);
}

void MatMulWriteFnSource(ShaderHelper& shader,
                         const ShaderVariableHelper& output,
                         const ShaderVariableHelper* bias,
                         bool is_gemm,
                         int c_components,
                         bool c_is_scalar,
                         std::string activation_snippet,
                         bool is_channels_last,
                         bool use_split_k,
                         ProgramVariableDataType output_variable_type) {
  const int output_components = output.NumComponents();
  shader.AdditionalImplementation()
      << "fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: output_value_t) {\n";

  shader.AdditionalImplementation() << "  let col = colIn * " << output_components << ";\n";

  shader.AdditionalImplementation() << "  if(row < i32(uniforms.dim_a_outer) && col < i32(uniforms.dim_b_outer)) {\n"
                                    << "    var value = valueIn;\n";

  if (use_split_k) {
    // Set output when MatMul is performed with Split-K.
    // When Split-K is used in MatMul, the bias will be handled in `MatMulFillBiasOrZeroBeforeSplitKProgram`
    // instead of here, so `bias` and `is_channels_last` is not used for Split-K. Note that we
    // still need to handle `bias` (and `is_channels_last` in the future) in
    // `MatMulFillBiasOrZeroBeforeSplitKProgram`.
    ORT_ENFORCE(bias == nullptr, "Bias is not supported in MatMulProgram when Split-K is enabled.");
    if (!is_gemm) {
      ORT_ENFORCE(is_channels_last, "Only channels-last is supported in MatMulProgram when Split-K is enabled in non-GEMM ops.");
    }
    HandleMatMulWithSplitK(shader, is_gemm, output, output_variable_type);
  } else if (is_gemm) {
    HandleMaybeHaveBiasForGEMM(shader, output, bias, c_components, output_components, c_is_scalar);
  } else {
    HandleMaybeBiasForMatMul(shader, output, bias, activation_snippet, is_channels_last);
  }

  shader.AdditionalImplementation()
      << "  }\n"
      << "}\n\n";
}

Status MakeMatMulPackedVec4Source(ShaderHelper& shader,
                                  const InlinedVector<int64_t>& elements_per_thread,
                                  uint32_t workgroup_size_x,
                                  uint32_t workgroup_size_y,
                                  const std::string& data_type,
                                  const ShaderIndicesHelper* batch_dims,
                                  bool transpose_a,
                                  bool transpose_b,
                                  float alpha,
                                  bool need_handle_matmul,
                                  int output_components,
                                  uint32_t tile_inner,
                                  bool split_k,
                                  uint32_t split_dim_inner) {
  const std::string type_string = MakeScalarOrVectorType(4 /*components */, data_type);

  std::string write_data_to_sub_a_vec4_snippet =
      transpose_a ? std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, kStart + inputRow, globalRowStart / innerElementSize + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n"
                  : std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRow + innerRow, kStart / innerElementSize + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n";
  std::string write_data_to_sub_b_vec4_snippet =
      transpose_b ? std::string("mm_Bsub[inputRow][inputCol] = mm_readB(batch, globalColStart + tileRowB + innerRow, kStart / innerElementSize + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n"
                  : std::string("mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalCol") + (batch_dims ? ", batchIndices" : "") + ");\n";

  // elements per thread
  const auto elements_per_thread_x = elements_per_thread[0];
  const auto elements_per_thread_y = elements_per_thread[1];

  const auto tile_a_outer = workgroup_size_y * elements_per_thread_y;
  const auto tile_b_outer = workgroup_size_x * elements_per_thread_x;
  const auto tile_a_width = transpose_a ? tile_a_outer : tile_inner;
  const auto tile_a_height = transpose_a ? tile_inner : tile_a_outer;
  const auto inner_elements_size = tile_a_width / workgroup_size_x;
  const auto row_per_thread_b = tile_inner / workgroup_size_y;

  if (!((transpose_a && inner_elements_size == 4 && elements_per_thread[1] == 4) ||
        (!transpose_a && (inner_elements_size == 3 || inner_elements_size == 4))) &&
      tile_a_width % workgroup_size_x == 0 &&
      tile_inner % workgroup_size_y == 0 &&
      elements_per_thread_x == 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid matrix multiplication configuration inner_elements_size: ", inner_elements_size,
                           " must be 3 or 4. tile_a_width: ", tile_a_width, " must be divisible by WorkgroupSizeX: ",
                           workgroup_size_x, ". tile_inner: ", tile_inner, " must be divisible by WorkgroupSizeY: ",
                           workgroup_size_y, ". elements_per_thread_x: ", elements_per_thread_x, " must be 4.");
  }

  shader.AdditionalImplementation()
      << "var<workgroup> mm_Asub: array<array<vec" << inner_elements_size << "<" << data_type << ">, " << tile_a_width / inner_elements_size << ">, " << tile_a_height << ">;\n"
      << "var<workgroup> mm_Bsub: array<array<vec4<" << data_type << ">, " << tile_b_outer / elements_per_thread_x << ">, " << tile_inner << ">;\n"
      << "const rowPerThread = " << elements_per_thread_y << ";\n"
      << "const colPerThread = " << elements_per_thread_x << ";\n"
      << "const innerElementSize = " << inner_elements_size << ";\n"
      << "const tileInner = " << tile_inner << ";\n";

  InitializeLogicalWorkgroupIDAndGlobalID(shader);

  shader.MainFunctionBody()
      << "  let localRow = i32(local_id.y);\n"
      << "  let tileRow = localRow * rowPerThread;\n"
      << "  let tileCol = i32(local_id.x);\n"
      << "  let globalRow = i32(logical_global_id.y) * rowPerThread;\n"
      << "  let globalCol = i32(logical_global_id.x);\n"
      << "  let globalRowStart = i32(logical_workgroup_id.y) * " << tile_a_outer << ";\n"
      << "  let globalColStart = i32(logical_workgroup_id.x) * " << tile_b_outer << ";\n"
      << "  var acc: array<vec4<" << data_type << ">, rowPerThread>;\n";

  if (split_k) {
    // With Split-K, the original "workgroup" (with dispatch_z == 1 in API side) is split into
    // multiple ones, and in the current workgroup we only compute `kSplitK` elements starting from
    // `kSplitK * i32(logical_global_id.z)`.
    //
    //  For example: considering computing Y = (X * W + B) in one workgroup.
    //  Let kSplitK = 2, B = [d1, d2]
    //  Let X = [[a1 a1 b1 b1 c1 c1]  = [ A1 B1 C1 ], W = [[a2 a2]  = [ A2
    //           [a1 a1 b1 b1 c1 c1]]                      [a2 a2]      B2
    //                                                     [b2 b2]      C2 ]
    //                                                     [b2 b2]
    //                                                     [c2 c2]
    //                                                     [c2 c2]]
    //
    //  With Split-K:
    //  1. Initialize output Y with B in `MatMulFillBiasOrZeroBeforeSplitKProgram`:  Y = [[d1, d2]
    //                                                                                   [d1, d2]]
    //  2. Split the original 1 workgroup into 3 workgroups (now `dispatch_z = 3` in API side)
    //     Workgroup1: compute (A1 * A2)  Workgroup2: compute (B1 * B2)
    //     Workgroup3: compute (C1 * C2)
    //     In each workgroup:
    //     - `num_tiles` is computed with `kSplitK`, and `kStart` is computed with `logical_global_id.z`
    //     - When the computation in each workgroup is completed, add the result to Y with several
    //       atomic built-in functions in `HandleMatMulWithSplitK()`.
    shader.MainFunctionBody()
        << "const kSplitK = " << split_dim_inner << ";\n"
        << "  let num_tiles = (kSplitK - 1) / tileInner + 1;\n"
        << "  var kStart = kSplitK * i32(logical_global_id.z);\n"

        // When Split-K is used, `batch` should always be 0 and `logical_global_id.z` is used to indicate
        // the index of split-k instead of batch.
        << "  let batch = 0;\n"
        << "  let batchIndices = 0u;\n";
  } else {
    shader.MainFunctionBody()
        << "  let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
        << "  var kStart = 0;\n"
        << "  let batch = i32(logical_global_id.z);\n"
        << (nullptr != batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "");
  }

  // Loop over shared dimension.
  shader.MainFunctionBody() << "  let tileRowB = localRow * " << row_per_thread_b << ";\n";

  if (need_handle_matmul) {
    shader.MainFunctionBody() << "  for (var t = 0; t < i32(num_tiles); t = t + 1) {\n";

    // Load one tile of A into local memory.
    shader.MainFunctionBody()
        << "    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
        << "      let inputRow = tileRow + innerRow;\n"
        << "      let inputCol = tileCol;\n"
        << "      " << write_data_to_sub_a_vec4_snippet
        << "    }\n";

    // Load one tile of B into local memory.
    shader.MainFunctionBody()
        << "    for (var innerRow = 0; innerRow < " << row_per_thread_b << "; innerRow = innerRow + 1) {\n"
        << "      let inputRow = tileRowB + innerRow;\n"
        << "      let inputCol = tileCol;\n"
        << "     " << write_data_to_sub_b_vec4_snippet
        << "    }\n"
        << "    kStart = kStart + tileInner;\n"
        << "    workgroupBarrier();\n";

    if (transpose_a && transpose_b) {
      shader.MainFunctionBody()
          << "    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {\n"
          << "      let BCached0 = mm_Bsub[tileCol * innerElementSize][k];\n"
          << "      let BCached1 =  mm_Bsub[tileCol * innerElementSize + 1][k];\n"
          << "      let BCached2 =  mm_Bsub[tileCol * innerElementSize + 2][k];\n";
      if (inner_elements_size != 3) {
        shader.MainFunctionBody() << "      let BCached3 = mm_Bsub[tileCol * innerElementSize + 3][k];\n";
      }
      shader.MainFunctionBody()
          << "      let ACached0 = mm_Asub[k * innerElementSize][localRow];\n"
          << "      let ACached1 = mm_Asub[k * innerElementSize + 1][localRow];\n"
          << "      let ACached2 = mm_Asub[k * innerElementSize + 2][localRow];\n"
          << (inner_elements_size == 3 ? "" : "      let ACached3 = mm_Asub[k * innerElementSize + 3][localRow];\n")
          << "      for (var i = 0; i < rowPerThread; i = i + 1) {\n"
          << "             acc[i].x += ACached0[i] * BCached0.x + ACached1[i] * BCached0.y + ACached2[i] * BCached0.z + ACached3[i] * BCached0.w;\n"
          << "             acc[i].y += ACached0[i] * BCached1.x + ACached1[i] * BCached1.y + ACached2[i] * BCached1.z + ACached3[i] * BCached1.w;\n"
          << "             acc[i].z += ACached0[i] * BCached2.x + ACached1[i] * BCached2.y + ACached2[i] * BCached2.z + ACached3[i] * BCached2.w;\n"
          << "             acc[i].w += ACached0[i] * BCached3.x + ACached1[i] * BCached3.y + ACached2[i] * BCached3.z + ACached3[i] * BCached3.w;\n"
          << "      }\n";
    } else if (transpose_a && !transpose_b) {
      shader.MainFunctionBody()
          << "    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {\n"
          << "      let BCached0 = mm_Bsub[k * innerElementSize][tileCol];\n"
          << "      let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];\n"
          << "      let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];\n";
      if (inner_elements_size != 3) {
        shader.MainFunctionBody() << "      let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];\n";
      }
      shader.MainFunctionBody()
          << "      let ACached0 = mm_Asub[k * innerElementSize][localRow];\n"
          << "      let ACached1 = mm_Asub[k * innerElementSize + 1][localRow];\n"
          << "      let ACached2 = mm_Asub[k * innerElementSize + 2][localRow];\n"
          << (inner_elements_size == 3 ? "" : "      let ACached3 = mm_Asub[k * innerElementSize + 3][localRow];\n")
          << "      for (var i = 0; i < rowPerThread; i = i + 1) {\n"
          << "        let ACached = mm_Asub[tileCol][i];\n"
          << "        acc[i] = BCached0 * ACached0[i] + acc[i];\n"
          << "        acc[i] = BCached1 * ACached1[i] + acc[i];\n"
          << "        acc[i] = BCached2 * ACached2[i] + acc[i];\n"
          << "        " << (inner_elements_size == 3 ? "" : "acc[i] = BCached3 * ACached3[i] + acc[i];") << "\n"
          << "      }\n";
    } else if (!transpose_a && transpose_b) {
      shader.MainFunctionBody()
          << "    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {\n"
          << "      let BCached0 = mm_Bsub[tileCol * innerElementSize][k];\n"
          << "      let BCached1 =  mm_Bsub[tileCol * innerElementSize + 1][k];\n"
          << "      let BCached2 =  mm_Bsub[tileCol * innerElementSize + 2][k];\n";
      if (inner_elements_size != 3) {
        shader.MainFunctionBody() << "      let BCached3 = mm_Bsub[tileCol * innerElementSize + 3][k];\n";
      }
      shader.MainFunctionBody()
          << "      for (var i = 0; i < rowPerThread; i = i + 1) {\n"
          << "        let ACached = mm_Asub[tileRow + i][k];\n"
          << "        acc[i].x += dot(ACached, BCached0);\n"
          << "        acc[i].y += dot(ACached, BCached1);\n"
          << "        acc[i].z += dot(ACached, BCached2);\n"
          << "        " << (inner_elements_size == 3 ? "" : "acc[i].w += dot(ACached, BCached3);") << "\n"
          << "      }\n";
    } else {
      shader.MainFunctionBody()
          << "    for (var k = 0; k < tileInner / innerElementSize; k = k + 1) {\n"
          << "      let BCached0 = mm_Bsub[k * innerElementSize][tileCol];\n"
          << "      let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];\n"
          << "      let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];\n";
      if (inner_elements_size != 3) {
        shader.MainFunctionBody() << "      let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];\n";
      }
      shader.MainFunctionBody()
          << "      for (var i = 0; i < rowPerThread; i = i + 1) {\n"
          << "        let ACached = mm_Asub[tileRow + i][k];\n"
          << "        acc[i] = BCached0 * ACached.x + acc[i];\n"
          << "        acc[i] = BCached1 * ACached.y + acc[i];\n"
          << "        acc[i] = BCached2 * ACached.z + acc[i];\n"
          << "        " << (inner_elements_size == 3 ? "" : "acc[i] = BCached3 * ACached.w + acc[i];") << "\n"
          << "      }\n";
    }
    shader.MainFunctionBody()
        << "    }\n"
        << "    workgroupBarrier();\n"
        << "  }\n";  // main for loop

    // Calculate alpha * acc
    if (alpha != 1.0f) {
      shader.MainFunctionBody() << "  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
                                << "    acc[innerRow] = output_element_t(uniforms.alpha) * acc[innerRow];\n"
                                << "  }\n";
    }
  }

  // Write the results to the output buffer
  shader.MainFunctionBody() << "  for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n";
  if (output_components == 1) {
    shader.MainFunctionBody() << " for (var i = 0; i < innerElementSize; i = i + 1) {\n"
                              << "    mm_write(batch, globalRow + innerRow, globalCol * innerElementSize + i, acc[innerRow][i]);\n"
                              << "  }\n";
  } else {
    shader.MainFunctionBody() << "    mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);\n";
  }

  shader.MainFunctionBody() << "  }\n";

  return Status::OK();
}

Status MakeMatMulPackedSource(ShaderHelper& shader,
                              const InlinedVector<int64_t>& elements_per_thread,
                              uint32_t workgroup_size_x,
                              uint32_t workgroup_size_y,
                              const std::string& data_type,
                              const ShaderIndicesHelper* batch_dims,
                              bool transpose_a,
                              bool transpose_b,
                              float alpha,
                              bool need_handle_matmul,
                              uint32_t tile_inner,
                              bool split_k,
                              uint32_t split_dim_inner) {
  ORT_UNUSED_PARAMETER(split_k);
  ORT_UNUSED_PARAMETER(split_dim_inner);

  const auto elements_per_thread_x = elements_per_thread[0];
  const auto elements_per_thread_y = elements_per_thread[1];

  const auto tile_a_outer = workgroup_size_y * elements_per_thread_y;
  const auto tile_b_outer = workgroup_size_x * elements_per_thread_x;
  const auto tile_a_width = transpose_a ? tile_a_outer : tile_inner;
  const auto tile_a_height = transpose_a ? tile_inner : tile_a_outer;

  if (!(tile_a_height % workgroup_size_y == 0 && tile_a_width % workgroup_size_x == 0 && tile_inner % workgroup_size_y == 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "tile_a_height: ", tile_a_height, " must be divisible by WorkgroupSizeY: ", workgroup_size_y,
                           ", tile_a_width: ", tile_a_width, " must be divisible by WorkgroupSizeX: ", workgroup_size_x,
                           ", tile_inner: ", tile_inner, " must be divisible by WorkgroupSizeY: ", workgroup_size_y);
  }

  const auto row_per_thread_a = tile_a_height / workgroup_size_y;
  const auto col_per_thread_a = tile_a_width / workgroup_size_x;
  const auto row_per_thread_b = tile_inner / workgroup_size_y;
  std::string write_data_to_sub_a_snippet = transpose_a ? std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, kStart + inputRow, globalRowStart + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n"
                                                        : std::string("mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRowStart + inputRow, kStart + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n";
  std::string write_data_to_sub_b_snippet = transpose_b ? std::string("mm_Bsub[inputRow][inputCol] = mm_readB(batch, globalColStart + tileRowB + innerRow, kStart + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n"
                                                        : std::string("mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, globalColStart + inputCol") + (batch_dims ? ", batchIndices" : "") + ");\n";
  shader.AdditionalImplementation()
      << "var<workgroup> mm_Asub: array<array<" << data_type << ", " << tile_a_width << ">, " << tile_a_height << ">;\n"
      << "var<workgroup> mm_Bsub: array<array<" << data_type << ", " << tile_b_outer << ">, " << tile_inner << ">;\n"
      << "const rowPerThread = " << elements_per_thread_y << ";\n"
      << "const colPerThread = " << elements_per_thread_x << ";\n"
      << "const tileInner = " << tile_inner << ";\n";

  InitializeLogicalWorkgroupIDAndGlobalID(shader);

  shader.MainFunctionBody() << " let batch = i32(logical_global_id.z);\n"
                            << (nullptr != batch_dims ? "  let batchIndices = " + batch_dims->OffsetToIndices("u32(batch)") + ";\n" : "")
                            << " let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
                            << " var kStart = 0;\n"
                            << " var acc: array<array<" << data_type << ", colPerThread>, rowPerThread>;\n";

  shader.MainFunctionBody()
      << "let tileRow = i32(local_id.y) * rowPerThread;\n"
      << "let tileCol = i32(local_id.x) * colPerThread;\n"
      << "let globalRow = i32(logical_global_id.y) * rowPerThread;\n"
      << "let globalCol = i32(logical_global_id.x) * colPerThread;\n"
      << "let globalRowStart = i32(logical_workgroup_id.y) * " << tile_a_outer << ";\n"
      << "let globalColStart = i32(logical_workgroup_id.x) * " << tile_b_outer << ";\n"
      << "let tileRowA = i32(local_id.y) * " << row_per_thread_a << ";\n"
      << "let tileColA = i32(local_id.x) * " << col_per_thread_a << ";\n"
      << "let tileRowB = i32(local_id.y) * " << row_per_thread_b << ";\n";

  if (need_handle_matmul) {
    // Loop over shared dimension.
    shader.MainFunctionBody()
        << "for (var t = 0; t < i32(num_tiles); t = t + 1) {\n";

    // Load one tile of A into local memory.
    shader.MainFunctionBody()
        << "  for (var innerRow = 0; innerRow < i32(" << row_per_thread_a << "); innerRow = innerRow + 1) {\n"
        << "    for (var innerCol = 0; innerCol < i32(" << col_per_thread_a << "); innerCol = innerCol + 1) {\n"
        << "      let inputRow = tileRowA + innerRow;\n"
        << "      let inputCol = tileColA + innerCol;\n"
        << "      " << write_data_to_sub_a_snippet << "\n"
        << "    }\n"
        << "  }\n";

    // Load one tile of B into local memory.
    shader.MainFunctionBody()
        << "  for (var innerRow = 0; innerRow < i32(" << row_per_thread_b << "); innerRow = innerRow + 1) {\n"
        << "    for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n"
        << "      let inputRow = tileRowB + innerRow;\n"
        << "      let inputCol = tileCol + innerCol;\n"
        << "           " << write_data_to_sub_b_snippet << "\n "
        << "    }\n"
        << "  }\n"
        << "  kStart = kStart + tileInner;\n"
        << "  workgroupBarrier();\n";

    // Compute acc values for a single thread.
    shader.MainFunctionBody()
        << "var BCached: array<" << data_type << ", colPerThread>;\n"
        << "  for (var k = 0; k < tileInner; k = k + 1) {\n"
        << "    for (var inner = 0; inner < i32(colPerThread); inner = inner + 1) {\n";
    if (transpose_b) {
      shader.MainFunctionBody() << "      BCached[inner] = mm_Bsub[tileCol + inner][k];\n";
    } else {
      shader.MainFunctionBody()
          << "      BCached[inner] = mm_Bsub[k][tileCol + inner];\n";
    }
    shader.MainFunctionBody() << "    }\n"
                              << "    for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {\n";
    if (transpose_a) {
      shader.MainFunctionBody() << "      let ACached = mm_Asub[k][tileRow + innerRow];\n";
    } else {
      shader.MainFunctionBody() << "      let ACached = mm_Asub[tileRow + innerRow][k];\n";
    }
    shader.MainFunctionBody() << "      for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n"
                              << "        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];\n"
                              << "      }\n"
                              << "    }\n"
                              << "  }\n"
                              << "  workgroupBarrier();\n"
                              << "}\n";

    // Calculate alpha * acc
    if (alpha != 1.0f) {
      shader.MainFunctionBody() << "for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {\n"
                                << "  for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n"
                                << "    acc[innerRow][innerCol] = output_element_t(uniforms.alpha) * acc[innerRow][innerCol];\n"
                                << "  }\n"
                                << "}\n";
    }
  }
  // Write the results to the output buffer
  shader.MainFunctionBody()
      << "for (var innerRow = 0; innerRow < i32(rowPerThread); innerRow = innerRow + 1) {\n"
      << "  for (var innerCol = 0; innerCol < i32(colPerThread); innerCol = innerCol + 1) {\n"
      << "    mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);\n"
      << "  }\n"
      << "}\n";
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
