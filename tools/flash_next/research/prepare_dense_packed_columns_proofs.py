from pathlib import Path

from build_qmoe_node_replay import replace_once

ROOT = Path(__file__).parent


def base(source):
    return (source.replace("dense-lane8", "dense-packed-columns")
            .replace("dense_lane8", "dense_packed_columns")
            .replace("ORT_WEBGPU_DENSE_LANE8", "ORT_WEBGPU_DENSE_PACKED_COLUMNS")
            .replace("MatMulNBitsLane8F16", "MatMulNBitsPackedColumnsF16")
            .replace('"lane8"', '"packed_columns"')
            .replace("'lane8'", "'packed_columns'")
            .replace("lane8", "packed_columns"))


def model(source):
    result = base(source)
    result = replace_once(result, "    prerequisites = []",
        '''    candidate_native_path = native_path
    candidate_native = native
    if args.mode == "control":
        native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
        native = json.loads(native_path.read_text())
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/"
                        "libonnxruntime_providers_webgpu.so")
        assert sha(provider) == native["provider_sha256"] == parent["provider_sha256"]
    prerequisites = []''')
    result = replace_once(result,
        'assert proof["status"] == expected and proof["provider_sha256"] == native["provider_sha256"]',
        '''gate_provider = (candidate_native["provider_sha256"] if "-screen-" in path.name or
                         "-profile" in path.name else native["provider_sha256"])
        assert proof["status"] == expected and proof["provider_sha256"] == gate_provider''')
    result = replace_once(result, "build_path, template_path, controls_path, *prerequisites, *profiles,",
                          "build_path, candidate_native_path, template_path, controls_path, *prerequisites, *profiles,")
    return replace_once(result, "experiment_flag=FLAG,",
                        'original_provider_control=args.mode == "control", experiment_flag=FLAG,')


def screen(source):
    return (source.replace("hc_projection_packed16", "dense_packed_columns")
            .replace("unroll", "packed_columns"))


if __name__ == "__main__":
    generated = dict(
        native=base((ROOT / "run_dense_lane8_native.py").read_text()),
        model=model((ROOT / "run_dense_lane8_model.py").read_text()),
        screen=screen((ROOT / "run_hc_projection_packed16_screen.py").read_text()))
    for kind, source in generated.items():
        target = ROOT / f"run_dense_packed_columns_{kind}.py"
        with target.open("x") as stream:
            stream.write(source)
        print(target.name)
