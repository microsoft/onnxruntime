from pathlib import Path

from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_FULL_A"
PROGRAM = "QmoeFC1FullABitcastF16"


def transform_program(source):
    begin = source.index('         << "  for (var id = local_idx;')
    end = source.index('         << "  for (var offset =', begin)
    load = source[begin:end]
    source = source[:begin] + ''';
    if (!full_a_) {
      code ''' + load + ''';
    }
    code ''' + source[end:]
    anchor = '         << "for (var kidx = 0u;'
    begin = source.index(anchor)
    source = source[:begin] + ''';
    if (full_a_) {
      code << "for (var id = local_idx; id < 640u; id += 128u) {\\n"
           << "  tile_a[id] = " << a.GetByOffset("id") << ";\\n"
           << "}\\nworkgroupBarrier();\\n";
    }
    code ''' + source[begin:]
    original = ("        sum += dot(tile_a[lane * 8u + i * 2u], w0) + "
                "dot(tile_a[lane * 8u + i * 2u + 1u], w1);\\n")
    candidate = original.replace("tile_a[", "tile_a[kidx / 4u + ")
    source = replace_once(source, f'<< "{original}"', f'<< (full_a_ ? "{candidate}" : "{original}")')
    source = replace_once(source,
        '         << "  workgroupBarrier();\\n"\n         << "}\\n"',
        '         << (full_a_ ? "" : "  workgroupBarrier();\\n")\n'
        '         << "}\\n"\n         << (full_a_ ? "workgroupBarrier();\\n" : "")')
    source = replace_once(source, '<< "var<workgroup> tile_a: array<vec4<f16>, 256>;\\n"',
                          '<< (full_a_ ? "var<workgroup> tile_a: array<vec4<f16>, 640>;\\n" : '
                          '"var<workgroup> tile_a: array<vec4<f16>, 256>;\\n")')
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool full_a)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{full_a ? (bitcast ? "QmoeFC1FullABitcastF16" : '
                          '"QmoeFC1FullAF16") : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, full_a_{full_a} {}")
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool full_a_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc1_full_a = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_full_a == 0 || fc1_full_a == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_full_a);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        '''const bool use_fc1_full_a = fc1_full_a == 1 &&
          context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
          context.DeviceLimits().maxComputeWorkgroupStorageSize >= 5632;
      QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1,
                                 use_fc1_full_a};''')
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc1_activation.inc").read_text()))


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc1-full-a/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
