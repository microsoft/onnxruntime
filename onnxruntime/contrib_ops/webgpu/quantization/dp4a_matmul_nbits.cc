// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {
namespace {

constexpr std::string_view commonFunctions = R"ADDNL_FN(
        fn DequantizedFrom4BitsTo8Bits(in: vec2<u32>) -> vec4<u32>
        {
            var out = vec4<u32>(0);
            var value_lower = vec4<i32>(unpack4xU8(in[0] & 0x0F0F0F0Fu)) - vec4<i32>(8);
            var value_upper = vec4<i32>(unpack4xU8((in[0] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
            out[0] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
            out[1] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
            value_lower = vec4<i32>(unpack4xU8(in[1] & 0x0F0F0F0Fu)) - vec4<i32>(8);
            value_upper = vec4<i32>(unpack4xU8((in[1] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
            out[2] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
            out[3] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
            return out;
        }

        // Scaled dot product of 8 packed unsigned integers.
        fn SDP8AI(a1:vec4<u32>, b1:vec4<u32>, a2:vec4<u32>, b2:vec4<u32>, scale:output_element_t) -> output_element_t
        {
            var local_sum = dot4I8Packed(a1[0], b1[0]);
            local_sum += dot4I8Packed(a1[1], b1[1]);
            local_sum += dot4I8Packed(a1[2], b1[2]);
            local_sum += dot4I8Packed(a1[3], b1[3]);
            local_sum += dot4I8Packed(a2[0], b2[0]);
            local_sum += dot4I8Packed(a2[1], b2[1]);
            local_sum += dot4I8Packed(a2[2], b2[2]);
            local_sum += dot4I8Packed(a2[3], b2[3]);
            return output_element_t(local_sum) * scale;
        }
  )ADDNL_FN";

}  // namespace

Status DP4AMatMulQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("scales", ShaderUsage::UseUniform);
  shader.MainFunctionBody() << R"MAIN_FN(
        var local_a : array<vec4<input_a_element_t>, 32>;
        var max_value:vec4<input_a_element_t> = vec4<input_a_element_t>(0);
        for (var idx:u32=0;idx<32;idx+=1)
        {
            local_a[idx] = input_a[workgroup_idx*32 + idx];
            max_value = max(max_value, abs(local_a[idx]));
        }
        var scale = max(max_value.x, max_value.y);
        scale = max(scale, max_value.z);
        scale = max(scale, max_value.w);
        for (var idx:u32=0;idx<32;idx+=1)
        {
            output[workgroup_idx*32+idx] = pack4x8snorm(vec4<f32>(local_a[idx]/scale));
        }
        // 127 is the max value of signed int8 [-127,127] used by pack4x8snorm for 1.0f.
        scales[workgroup_idx] = scale/127;
    )MAIN_FN";
  return Status::OK();
}

Status DP4AMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  // This shader implements co-operative matrix multiply. The key idea here is to
  // assume there is a primitive for medium size matrix multiply a subgroup can perform,
  // using all its lanes and pooling all its registers to keep the values in registry.
  //
  // The entire workgroup which has N subgroups first loads a tile into shared memory,
  // Then each subgroup loads a subtile from shared memory into registers and uses
  // the medium size matrix multiply primitive to perform the math.
  // The values for tile/subtile size are chosen to conform to the resource limits
  // of an alderlake/tiger lake gpu. A tile is 64x64, workgroup is 256 threads -
  // therefore there are 16 subgroups and 16 lanes in each subgroup.
  // K the hidden dimension is paged in from RAM at k tile size which is 64.
  // All this puts the shared memory requirement slightly above 16KB.
  // WebGPU limit is 16KB, output is moved to registers instead of SHM to make
  // everything fit in shared memory.
  //
  // Each subgroup performs a 16 x 64 x 16 multiply which is implemented with
  // subgroup shuffle as a placeholder for the day the medium matrix mul primitive
  // becomes available in WGSL. The registry requirements is ~2KB per subgroup, on
  // Alderlake/Tigerlake subgroup has 8KB of registry space pooling the
  // 512B of registry from each lane.
  //
  // The medium size matmul is implemented using dot4I8Packed, so the inputs for
  // this shader require A to be int8 quantized with block size 64. B is regular
  // matmulnbits input with block size 32.

  shader.AdditionalImplementation() << commonFunctions
                                    << "  const block_size = " << block_size_ << ";";

  shader.AdditionalImplementation() << R"ADDNL_FN(
        const tile_size = 64;
        const subtile_size = 16;
        const tile_size_k =  32;
        const vec_factor = 4;
        const u32_factor = 4;
        const tile_size_k_vec = 2;

        // Shared memory
        var<workgroup> tile_A : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;                     // 64 x 32
        var<workgroup> scale_A : array<output_element_t, tile_size>;                                     // 64 x 1
        var<workgroup> tile_B : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;                     // 64 x 32
        var<workgroup> scale_B : array<output_element_t, tile_size>;                                     // 64 x 1

        fn loadSHMA(a_global_base:u32, kidx_v:u32, row: u32, col: u32)
        {
            let a_global = a_global_base + row;
            if (a_global >= uniforms.M)
            {
                return;
            }
            tile_A[col][row] = input_a[a_global*uniforms.K16+kidx_v+col];
            if (col == 0)
            {
                // kidx_v - covers 16 values of k
                scale_A[row] = scales_a[a_global*(uniforms.K/128) + kidx_v/8];
            }
        }

        fn loadSHMB(b_global_base:u32, kidx_v:u32, row: u32, col: u32)
        {
            let b_global = b_global_base + row;
            if (b_global >= uniforms.N)
            {
                return;
            }

            let b_value = input_b[b_global*uniforms.K16+kidx_v+col];
            tile_B[col][row] = DequantizedFrom4BitsTo8Bits(b_value);
            if (col == 0)
            {
                // kidx_v - each kidx_v covers 16 values of k
                scale_B[row] = scales_b[b_global*(uniforms.K/block_size) + kidx_v/(block_size/16)];
            }
        }
    )ADDNL_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
        // During the load phase we use all 256 threads to load 64 rows of A/B.
        // For each row we load tile_size_k_vec (2) vectorized elements, which are 32 elements of K.
        let a_global_base = u32(workgroup_idx / uniforms.num_N_tile) * tile_size;
        let b_global_base = (workgroup_idx % uniforms.num_N_tile) * tile_size;
        let load_AorB = u32(local_idx/128);
        let load_row = u32((local_idx%128)/2);
        let load_col = u32(local_idx%2);

        // During the compute phase, we have the 64x64 tile split into
        // subtiles of 16x16. We have a grid of 4x4 subtiles.
        let subtile_id = u32(local_idx / subtile_size);
        let subtile_idx = u32(subtile_id / 4);
        let subtile_idy = u32(subtile_id % 4);
        let base_A = subtile_idx * 16;
        let base_B = subtile_idy * 16;
        // For each subtile we have 16 threads assigned.
        let a_idx = u32(local_idx % subtile_size);

        var lane_output1: vec4<output_element_t>;
        var lane_output2: vec4<output_element_t>;
        var lane_output3: vec4<output_element_t>;
        var lane_output4: vec4<output_element_t>;
        // K's vectrorization is 16 items per index. See input_a/input_b.
        // tile_size_k_vec - is the k tile size in vectorized space (1/16). That is
        // k tile size is 32. In vectorized space that is 32/16 = 2.
        for (var kidx_v:u32 = 0; kidx_v < uniforms.K16; kidx_v+=tile_size_k_vec)
        {
            // Load Phase: Populate shared memory for the workgroup.
            if (load_AorB == 0)
            {
                loadSHMA(a_global_base, kidx_v, load_row, load_col);
            }
            else
            {
                loadSHMB(b_global_base, kidx_v, load_row, load_col);
            }
            workgroupBarrier();

            // Compute phase: Perform matmul for this subtile 16 x 32 x 16.
            // Step 1: Load from shared memory into registers across entire subgroup.
            var own_a0: vec4<u32> = tile_A[0][base_A + a_idx];
            var own_a1: vec4<u32> = tile_A[1][base_A + a_idx];
            var own_scale_a: output_element_t = scale_A[base_A + a_idx];
            if (sg_size == 16)
            {
                var own_b0: vec4<u32> = tile_B[0][base_B + sg_id];
                var own_b1: vec4<u32> = tile_B[1][base_B + sg_id];
                var own_scale_b: output_element_t  = scale_B[base_B + sg_id];
                // Step 2: Access registers across the subgroup using subgroupShuffle and perform the matmul.
                lane_output1[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 0), own_a1, subgroupShuffle(own_b1, 0), subgroupShuffle(own_scale_b, 0) * own_scale_a);
                lane_output1[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 1), own_a1, subgroupShuffle(own_b1, 1), subgroupShuffle(own_scale_b, 1) * own_scale_a);
                lane_output1[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 2), own_a1, subgroupShuffle(own_b1, 2), subgroupShuffle(own_scale_b, 2) * own_scale_a);
                lane_output1[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 3), own_a1, subgroupShuffle(own_b1, 3), subgroupShuffle(own_scale_b, 3) * own_scale_a);

                lane_output2[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 4), own_a1, subgroupShuffle(own_b1, 4), subgroupShuffle(own_scale_b, 4) * own_scale_a);
                lane_output2[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 5), own_a1, subgroupShuffle(own_b1, 5), subgroupShuffle(own_scale_b, 5) * own_scale_a);
                lane_output2[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 6), own_a1, subgroupShuffle(own_b1, 6), subgroupShuffle(own_scale_b, 6) * own_scale_a);
                lane_output2[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 7), own_a1, subgroupShuffle(own_b1, 7), subgroupShuffle(own_scale_b, 7) * own_scale_a);

                lane_output3[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 8), own_a1, subgroupShuffle(own_b1, 8), subgroupShuffle(own_scale_b, 8) * own_scale_a);
                lane_output3[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 9), own_a1, subgroupShuffle(own_b1, 9), subgroupShuffle(own_scale_b, 9) * own_scale_a);
                lane_output3[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 10), own_a1, subgroupShuffle(own_b1, 10), subgroupShuffle(own_scale_b, 10) * own_scale_a);
                lane_output3[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 11), own_a1, subgroupShuffle(own_b1, 11), subgroupShuffle(own_scale_b, 11) * own_scale_a);

                lane_output4[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 12), own_a1, subgroupShuffle(own_b1, 12), subgroupShuffle(own_scale_b, 12) * own_scale_a);
                lane_output4[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 13), own_a1, subgroupShuffle(own_b1, 13), subgroupShuffle(own_scale_b, 13) * own_scale_a);
                lane_output4[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 14), own_a1, subgroupShuffle(own_b1, 14), subgroupShuffle(own_scale_b, 14) * own_scale_a);
                lane_output4[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 15), own_a1, subgroupShuffle(own_b1, 15), subgroupShuffle(own_scale_b, 15) * own_scale_a);
            }
            else
            {
                // Code for other subgroup sizes, simply doesnt use subgroups at all.
                // Relies on reads from single location tile_B[][base_B + col] by all
                // being optimized by the hardware.
                lane_output1[0] += SDP8AI(own_a0, tile_B[0][base_B + 0], own_a1, tile_B[1][base_B + 0],  own_scale_a * scale_B[base_B + 0]);
                lane_output1[1] += SDP8AI(own_a0, tile_B[0][base_B + 1], own_a1, tile_B[1][base_B + 1],  own_scale_a * scale_B[base_B + 1]);
                lane_output1[2] += SDP8AI(own_a0, tile_B[0][base_B + 2], own_a1, tile_B[1][base_B + 2],  own_scale_a * scale_B[base_B + 2]);
                lane_output1[3] += SDP8AI(own_a0, tile_B[0][base_B + 3], own_a1, tile_B[1][base_B + 3],  own_scale_a * scale_B[base_B + 3]);

                lane_output2[0] += SDP8AI(own_a0, tile_B[0][base_B + 4], own_a1, tile_B[1][base_B + 4],  own_scale_a * scale_B[base_B + 4]);
                lane_output2[1] += SDP8AI(own_a0, tile_B[0][base_B + 5], own_a1, tile_B[1][base_B + 5],  own_scale_a * scale_B[base_B + 5]);
                lane_output2[2] += SDP8AI(own_a0, tile_B[0][base_B + 6], own_a1, tile_B[1][base_B + 6],  own_scale_a * scale_B[base_B + 6]);
                lane_output2[3] += SDP8AI(own_a0, tile_B[0][base_B + 7], own_a1, tile_B[1][base_B + 7],  own_scale_a * scale_B[base_B + 7]);

                lane_output3[0] += SDP8AI(own_a0, tile_B[0][base_B + 8], own_a1, tile_B[1][base_B + 8],  own_scale_a * scale_B[base_B + 8]);
                lane_output3[1] += SDP8AI(own_a0, tile_B[0][base_B + 9], own_a1, tile_B[1][base_B + 9],  own_scale_a * scale_B[base_B + 9]);
                lane_output3[2] += SDP8AI(own_a0, tile_B[0][base_B + 10], own_a1, tile_B[1][base_B + 10],  own_scale_a * scale_B[base_B + 10]);
                lane_output3[3] += SDP8AI(own_a0, tile_B[0][base_B + 11], own_a1, tile_B[1][base_B + 11],  own_scale_a * scale_B[base_B + 11]);

                lane_output4[0] += SDP8AI(own_a0, tile_B[0][base_B + 12], own_a1, tile_B[1][base_B + 12],  own_scale_a * scale_B[base_B + 12]);
                lane_output4[1] += SDP8AI(own_a0, tile_B[0][base_B + 13], own_a1, tile_B[1][base_B + 13],  own_scale_a * scale_B[base_B + 13]);
                lane_output4[2] += SDP8AI(own_a0, tile_B[0][base_B + 14], own_a1, tile_B[1][base_B + 14],  own_scale_a * scale_B[base_B + 14]);
                lane_output4[3] += SDP8AI(own_a0, tile_B[0][base_B + 15], own_a1, tile_B[1][base_B + 15],  own_scale_a * scale_B[base_B + 15]);
            }
            workgroupBarrier();
        }

        let a_global = a_global_base + base_A + a_idx;
        let b_global = b_global_base + base_B;
        let output_idx = ((a_global) * uniforms.N + b_global)/4;
        // This creates a shader requirement that uniforms.N % 16 == 0
        if (a_global < uniforms.M && b_global < uniforms.N)
        {
            output[output_idx] = lane_output1;
            output[output_idx+1] = lane_output2;
            output[output_idx+2] = lane_output3;
            output[output_idx+3] = lane_output4;
        }
    )MAIN_FN";

  return Status::OK();
}

// scale_A components = 1, b components = 4, output components = 1
Status DP4AMatMulNBitsSmallMProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  // This algorithm works to compute dot product of k parallelly, by processing k at each step amongst tile_size_k_vec threads,
  // and utilizing the remaining threads in the workgroup to process additional rows of b in parallel (such that the values in shared memory for A can be reused).
  // For each load of k, the tile_size_k_vec threads also reload B tile_size/num_concurrent_b_rows times to compute partial dot products of other B rows
  // in order to complete all tile_size b rows in this workgroup and also reusing the loaded in register values of a.

  // 1. Each workgroup handles tile_size_k_vec (16) * k_vectorization_in_b (32) columns (total 512) and num_concurrent_b_rows of matrix B at a time,
  // iterating over the columns to compute a partial dot product.
  // 2. Uses vec4 vectorization where each K represents 32 elements of matrix B
  constexpr uint32_t tile_size_k_vec = 16;

  // 1. Workgroup Responsibility:
  //    - Processes one row of matrix A
  //    - Handles tile_size rows of matrix B
  //
  // 2. Computation Process:
  //    - Reads [tile_size][tile_size_k_vec] block of B data at a time
  //    - Each thread within workgroup computes dot products of 32 A*B elements since each K represents 32 elements of matrix B
  //    - Stores intermediate results in shared memory (inter_results)
  //    - Iterates through columns accumulating results in inter_results
  //    - Performs final reduction sum in inter_results for output
  shader.AdditionalImplementation() << "const tile_size = " << tile_size_ << "u;\n"
                                    << "const tile_size_k_vec = " << tile_size_k_vec << "u;\n"
                                    // sub_tile_size is the number of concurrent b rows processed by the workgroup.
                                    << "const sub_tile_size = " << WorkgroupSizeX() / tile_size_k_vec << "u;\n";
  shader.AdditionalImplementation() << commonFunctions
                                    << R"ADDNL_FN(
    // Shared memory
    // Need 2 * tile_size_k_vec (32) to store a tile_A since b is quantized as 4 bits and a is quantized as 8 bits.
    var<workgroup> tile_A : array<vec4<u32>, 32>;
    // Need 4 scales value since each tile_A includes 512 (4x4x32) scalars and the block_size is 128.
    var<workgroup> scale_A : array<output_element_t, 4>;
    var<workgroup> inter_results: array<array<output_element_t, tile_size_k_vec>, tile_size>;
    fn loadSHMA(a_global: u32, kidx_v: u32, col: u32)
    {
      let k_offset = kidx_v + col;
      if (k_offset >= uniforms.K16) {
        return;
      }

      tile_A[col] = input_a[a_global*uniforms.K16+k_offset];
      if (col < 4)
      {
        // kidx_v - covers 16 values of k in input_a
        scale_A[col] = scales_a[a_global*(uniforms.K/128) + kidx_v/8 + col];
      }
    }
  )ADDNL_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
    let a_global = u32(workgroup_idx / uniforms.num_N_tile);
    let b_global_base = (workgroup_idx % uniforms.num_N_tile) * tile_size;
    // Handle each workgroup threads as a block of [sub_tile_size][tile_size_k_vec]
    let local_col = local_idx % tile_size_k_vec;
    let local_row = local_idx / tile_size_k_vec;
    for (var kidx_v:u32 = 0; kidx_v < uniforms.K32; kidx_v += tile_size_k_vec)
    {
      // Load Phase: Populate shared memory for the workgroup.
      if (local_idx < 32)
      {
        loadSHMA(a_global, kidx_v * 2, local_idx);
      }
      workgroupBarrier();
      var own_a: vec4<u32> = tile_A[local_col * 2];
      var own_a1: vec4<u32> = tile_A[local_col * 2 + 1];
      var own_scale_a = scale_A[local_col / 4];
      var own_b = vec4<u32>(0);
      var own_b1 = vec4<u32>(0);
      let k_offset = kidx_v + local_col;
      // calculate intermediate results into inter_results.
      for (var row_offset = 0u; row_offset < tile_size; row_offset += sub_tile_size) {
        let b_global = b_global_base + row_offset + local_row;
        if (b_global < uniforms.N && k_offset < uniforms.K32)
        {
          let b_offset = b_global * uniforms.K32 + k_offset;
          let b_value = input_b[b_offset];
          own_b = DequantizedFrom4BitsTo8Bits(b_value.xy);
          own_b1 = DequantizedFrom4BitsTo8Bits(b_value.zw);

          // k_offset - covers 32 values of k in input_b
          let own_scale_b = scales_b[b_global * uniforms.K / uniforms.block_size + k_offset * 32 / uniforms.block_size];
          inter_results[row_offset + local_row][local_col] += SDP8AI(own_a, own_b, own_a1, own_b1, own_scale_a * own_scale_b);
        }
      }
      workgroupBarrier();
    }

    if (local_idx < tile_size) {
      // Do reduce sum to get final output.
      var output_value = output_element_t(0);
      for (var b = 0u; b < tile_size_k_vec; b++) {
        output_value += inter_results[local_idx][b];
      }
      let b_global =  b_global_base + local_idx;
      let output_idx = a_global * uniforms.N + b_global;
      if (b_global < uniforms.N) {
        output[output_idx] = output_value;
      }
    }
  )MAIN_FN";

  return Status::OK();
}

Status ApplyDP4AMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  uint32_t M,
                                  uint32_t N,
                                  uint32_t K,
                                  uint32_t block_size,
                                  uint32_t min_M_for_tile_optimization,
                                  onnxruntime::webgpu::ComputeContext& context,
                                  Tensor* y) {
  constexpr uint32_t kVec4Components = 4;
  constexpr uint32_t kVec2Components = 2;
  constexpr uint32_t kU32Components = 4;

  constexpr uint32_t kBlockSizeA = 128;
  DP4AMatMulQuantizeProgram quantize_program;
  quantize_program.SetWorkgroupSize(1);
  quantize_program.SetDispatchGroupSize(M * K / kBlockSizeA, 1, 1);
  TensorShape a_quant_shape{1, M, K / kU32Components};
  Tensor a_quant = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), a_quant_shape);
  TensorShapeVector a_scales_dims({1, 1, M, K / kBlockSizeA});
  Tensor a_scale = context.CreateGPUTensor(a->DataType(), a_scales_dims);
  quantize_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)}})
      .AddOutputs({{&a_quant, ProgramTensorMetadataDependency::Rank, a_quant.Shape(), 1},
                   {&a_scale, ProgramTensorMetadataDependency::Rank, a_scale.Shape(), 1}});
  ORT_RETURN_IF_ERROR(context.RunProgram(quantize_program));

  if (M < min_M_for_tile_optimization) {
    constexpr uint32_t kTileSize = 32;
    DP4AMatMulNBitsSmallMProgram mul_program{kTileSize};
    uint32_t num_N_tile = (N + kTileSize - 1) / kTileSize;
    mul_program.SetWorkgroupSize(128);
    mul_program.SetDispatchGroupSize(M * num_N_tile);
    mul_program.AddInputs({{&a_quant, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)},
                           {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, 1},
                           {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components * kU32Components)},
                           {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddUniformVariables({M, N, K, K / 16, K / 32, block_size, num_N_tile})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, 1});
    return context.RunProgram(mul_program);
  }

  constexpr uint32_t kTileSize = 64;
  TensorShape reshaped_y_shape{1, M, N / kVec4Components};
  uint32_t num_M_tile = (M + kTileSize - 1) / kTileSize;
  uint32_t num_N_tile = (N + kTileSize - 1) / kTileSize;
  DP4AMatMulNBitsProgram mul_program{block_size};
  mul_program.SetWorkgroupSize(256);
  mul_program.SetDispatchGroupSize(num_M_tile * num_N_tile);
  mul_program.AddInputs({{&a_quant, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)},
                         {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, 1},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec2Components * kU32Components)},
                         {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
      .AddUniformVariables({{static_cast<uint32_t>(M)},
                            {static_cast<uint32_t>(N)},
                            {static_cast<uint32_t>(K)},
                            {static_cast<uint32_t>(K / 8)},
                            {static_cast<uint32_t>(K / 16)},
                            {num_N_tile}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, static_cast<int>(kVec4Components)})
      .CacheHint("Block" + std::to_string(block_size));
  return context.RunProgram(mul_program);
}

bool CanApplyDP4AMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                   uint64_t accuracy_level,
                                   uint32_t block_size,
                                   uint32_t batch_count,
                                   uint32_t N,
                                   uint32_t K,
                                   uint32_t components_k,
                                   bool has_zero_points) {
  // macOS - Avoid using dp4a on Metal, as it does not appear to have native dp4a support.
  // https://github.com/gpuweb/gpuweb/issues/2677#issuecomment-1713292226
  bool use_dp4a = context.HasFeature(wgpu::FeatureName::Subgroups) &&
                  context.AdapterInfo().backendType != wgpu::BackendType::Metal;
  return (accuracy_level == 4 && block_size % 32 == 0 &&
          batch_count == 1 && components_k == 4 && K % 128 == 0 && N % 16 == 0 &&
          !has_zero_points && use_dp4a);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
