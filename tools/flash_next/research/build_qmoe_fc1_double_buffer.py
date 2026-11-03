import json
import os
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_QMOE_FC1_DOUBLE_BUFFER"
STORAGE_BYTES = 4608
ROOT = Path(__file__).parent
SOURCE = Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc")
HEADER = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe/qmoe_fc1_activation.inc")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-fc1-double-buffer/build-v1")


HELPER = f'''
int QmoeFc1DoubleBufferMode() {{
  static const int mode = [] {{
    const int value = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(value == 0 || value == 1, "{FLAG} must be 0 or 1");
    return value;
  }}();
  return mode;
}}

'''


def transform_program(source):
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool double_buffer)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{double_buffer ? "QmoeFC1SwiGLUDoubleBufferF16" : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, double_buffer_{double_buffer} {}")
    source = replace_once(source,
        '<< "var<workgroup> tile_a: array<vec4<f16>, 256>;\\n"',
        '<< (double_buffer_ ? "var<workgroup> tile_a: array<vec4<f16>, 512>;\\n" : '
        '"var<workgroup> tile_a: array<vec4<f16>, 256>;\\n")')
    source = replace_once(source,
        '<< "for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\\n"',
        '<< "for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\\n"\n'
        '         << (double_buffer_ ? "  let buffer = ((kidx / 1024u) % 2u) * 256u;\\n" : "")')
    source = replace_once(source, '<< "      tile_a[id] = "',
                          '<< (double_buffer_ ? "      tile_a[buffer + id] = " : "      tile_a[id] = ")')
    source = replace_once(source, '<< "    } else { tile_a[id] = vec4<f16>(0); }\\n"',
                          '<< (double_buffer_ ? "    } else { tile_a[buffer + id] = vec4<f16>(0); }\\n" : '
                          '"    } else { tile_a[id] = vec4<f16>(0); }\\n")')
    dot = "        sum += dot(tile_a[lane * 8u + i * 2u], w0) + dot(tile_a[lane * 8u + i * 2u + 1u], w1);\\n"
    source = replace_once(source, f'<< "{dot}"',
                          f'<< (double_buffer_ ? "{dot.replace("tile_a[", "tile_a[buffer + ")}" : "{dot}")')
    source = replace_once(source,
        '<< "  workgroupBarrier();\\n"\n         << "}\\n"\n         << "if (local_idx < 8u',
        '<< (double_buffer_ ? "" : "  workgroupBarrier();\\n")\n'
        '         << "}\\n"\n'
        '         << (double_buffer_ ? "workgroupBarrier();\\n" : "")\n'
        '         << "if (local_idx < 8u')
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool double_buffer_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    if source.count(validation) != 2:
        raise ValueError("Unexpected QMoE selector surfaces")
    source = source.replace(validation, validation + "\n  const int fc1_double_buffer = QmoeFc1DoubleBufferMode();")
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_double_buffer);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        f'''const bool use_fc1_double_buffer = fc1_double_buffer == 1 && expert_bitcast == 1 &&
          context.DeviceLimits().maxComputeWorkgroupStorageSize >= {STORAGE_BYTES};
      QmoeFC1SwiGLUProgram program{{fc1_experts_bias_optional != nullptr, expert_bitcast == 1,
                                  use_fc1_double_buffer}};''')
    source = replace_once(source, ".CacheHint(fc1_experts_bias_optional != nullptr);",
                          ".CacheHint(fc1_experts_bias_optional != nullptr, use_fc1_double_buffer);")
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        HELPER + transform_program(HEADER.read_text() if program_source is None else program_source))


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(original_k_streams=32, output_columns=8, threads=128,
                        activation_buffers=2, shared_storage_bytes=STORAGE_BYTES,
                        barriers_per_k_tile=1, final_partial_barrier=True,
                        no_reduction_or_normalization_changes=True,
                        independent_original_arithmetic_qualified=False,
                        natural_model_timing_performed=False))
    report["source_sha256"][os.path.relpath(HEADER, OUTPUT)] = sha(HEADER)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
