from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_DENSE_TILE_PARALLEL3"
CLASS = "DenseTileParallel3PackedProgram"
NAME = "DenseTileParallel3Packed"
SHARED_BYTES = 15360


def make_program():
    program = ORIGINAL_PROGRAM.replace("DenseColumns8DoubleBuffer", "DenseTileParallel3Packed")
    program = replace_once(program, "tile_a: array<vec4<f16>, 512>",
                           "tile_a: array<vec4<f16>, 1536>")
    program = replace_once(program, "partials: array<f16, 256>",
                           "tile_sums: array<u32, 768>")
    program = replace_once(
        program,
        "let stream = local_idx % 32u;\n"
        "let column_group = local_idx / 32u;\n"
        "for (var i = local_idx; i < 256u; i += 128u) { partials[i] = f16(0); }\n"
        "for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {\n"
        "  for (var i = local_idx; i < 256u; i += 128u) {",
        "let logical_idx = local_idx % 128u;\n"
        "let tile_worker = local_idx / 128u;\n"
        "let stream = logical_idx % 32u;\n"
        "let column_group = logical_idx / 32u;\n"
        "for (var wave = 0u; wave * 96u < uniforms.blocks; wave++) {\n"
        "  let tile = wave * 3u + tile_worker;\n"
        "  let buffer = (wave % 2u) * 3u + tile_worker;\n"
        "  for (var i = logical_idx; i < 256u; i += 128u) {")
    if program.count("(tile % 2u)") != 3:
        raise RuntimeError("Unexpected original double-buffer offsets")
    program = program.replace("(tile % 2u)", "buffer")
    program = replace_once(
        program,
        "  let block = tile * 32u + stream;\n"
        "  if (block < uniforms.blocks) {",
        "  let block = tile * 32u + stream;\n"
        "  var sum0 = f16(0);\n"
        "  var sum1 = f16(0);\n"
        "  if (block < uniforms.blocks) {")
    for column in (0, 1):
        program = replace_once(
            program, ' << ";\\n"\n' + f'         << "    var sum{column} = f16(0);\\n";',
            ' << ";\\n";')
        program = replace_once(
            program,
            f'    code << "    partials[(column_group + {column * 4}u) * 32u + stream] += sum{column};\\n";\n',
            "")
    program = replace_once(
        program,
        "  }\n}\nworkgroupBarrier();\nif (local_idx < 8u) {",
        "  }\n"
        "  tile_sums[tile * 128u + logical_idx] = bitcast<u32>(vec2<f16>(sum0, sum1));\n"
        "}\nworkgroupBarrier();\n"
        "if (local_idx < 128u) {\n"
        "  var partial0 = f16(0);\n"
        "  var partial1 = f16(0);\n"
        "  for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {\n"
        "    if (tile * 32u + stream < uniforms.blocks) {\n"
        "      let pair = bitcast<vec2<f16>>(tile_sums[tile * 128u + logical_idx]);\n"
        "      partial0 += pair.x;\n"
        "      partial1 += pair.y;\n"
        "    }\n"
        "  }\n"
        "  tile_sums[logical_idx] = bitcast<u32>(vec2<f16>(partial0, partial1));\n"
        "}\nworkgroupBarrier();\nif (local_idx < 8u) {")
    return replace_once(
        program, "    value += partials[local_idx * 32u + stream];",
        "    let pair = bitcast<vec2<f16>>(tile_sums[(local_idx % 4u) * 32u + stream]);\n"
        "    value += pair[local_idx / 4u];")


PROGRAM = make_program()


def transform(source):
    source = replace_once(source, ORIGINAL_PROGRAM, PROGRAM + "\n" + ORIGINAL_PROGRAM)
    source = replace_once(
        source,
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {",
        "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {\n"
        f'  static const int tile_parallel3 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);\n'
        f'  ORT_RETURN_IF_NOT(tile_parallel3 == 0 || tile_parallel3 == 1, "{FLAG} must be 0 or 1");')
    source = replace_once(
        source, "  auto compute = [&]() {",
        "  const bool use_tile_parallel3 = tile_parallel3 == 1 && columns8 &&\n"
        "      (K_ == 2560 || K_ == 6144) &&\n"
        '      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&\n'
        "      context.DeviceLimits().maxComputeWorkgroupSizeX >= 384 &&\n"
        "      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 384 &&\n"
        f"      context.DeviceLimits().maxComputeWorkgroupStorageSize >= {SHARED_BYTES};\n"
        "  auto compute = [&]() {")
    begin = source.index("    if (columns8) {")
    end = source.index("\n    }", begin) + len("\n    }")
    branch = source[begin:end]
    selected = replace_once(branch, "if (columns8)", "if (use_tile_parallel3)")
    selected = replace_once(selected, "DenseColumns8DoubleBufferProgram program;", CLASS + " program;")
    selected = replace_once(selected, ".SetWorkgroupSize(128)", ".SetWorkgroupSize(384)")
    source = source[:begin] + selected + "\n" + source[begin:]
    return replace_once(source, "columns8 ? 7 :", "columns8 ? (use_tile_parallel3 ? 8 : 7) :")
