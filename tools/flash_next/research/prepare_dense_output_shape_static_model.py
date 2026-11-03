from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    source = source.replace("dense-output-shape-v2", "dense-output-shape-static-v1")
    source = source.replace("qwen38-int4-dense-output-shape/build-v2",
                            "qwen38-int4-dense-output-shape-static/build-v1")
    source = source.replace("build_dense_output_shape import", "build_dense_output_shape_static import")
    source = source.replace("run_dense_output_shape_model", "run_dense_output_shape_static_model")
    source = source.replace("dense_output_shape_changing_inputs_finite_bitwise_qualified",
                            "dense_output_shape_session_static_changing_inputs_finite_bitwise_qualified")
    source = source.replace("dense_output_shape_rank1_fallback_rank234_finite_bitwise_qualified",
                            "dense_output_shape_session_static_rank1_fallback_rank234_finite_bitwise_qualified")
    source = source.replace("MatMulNBitsOutputShapeExact", "MatMulNBitsStaticOutputShapeExact")
    return source.replace("ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATS_FILE",
                          "ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATIC_STATS_FILE")


def model_source(source):
    source = base(source)
    anchor = '''    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest, name
    prerequisites = []'''
    source = replace_once(source, anchor, '''    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest, name
    candidate_native = native
    candidate_native_path = native_path
    original_native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    original_native = json.loads(original_native_path.read_text())
    assert original_native["provider_sha256"] == parent["provider_sha256"]
    if args.mode == "control":
        native = original_native
        native_path = original_native_path
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/"
                        "libonnxruntime_providers_webgpu.so")
        assert sha(provider) == native["provider_sha256"]
    prerequisites = []''')
    source = replace_once(source,
        '        assert proof["status"] == expected and proof["provider_sha256"] == native["provider_sha256"]',
        '''        gate_provider = (candidate_native["provider_sha256"] if "-screen-" in path.name or
                         "coverage" in path.name else native["provider_sha256"])
        assert proof["status"] == expected and proof["provider_sha256"] == gate_provider''')
    source = replace_once(source,
        '                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],',
        '                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],\n'
        '                  original_provider_control=args.mode == "control", constructor_scoped_experiment=True,')
    source = replace_once(source, "               build_path, ranks_path, template_path, controls_path,",
                          "               build_path, ranks_path, candidate_native_path, original_native_path,\n"
                          "               template_path, controls_path,")
    return source


def screen_source(source):
    source = base(source)
    source = replace_once(source, "    provider = None", "    providers = {}")
    source = replace_once(source,
        '            provider = proof["provider_sha256"] if provider is None else provider\n'
        '            assert proof["provider_sha256"] == raw["plugin_sha256"] == provider',
        '            providers.setdefault(mode, proof["provider_sha256"])\n'
        '            assert proof["provider_sha256"] == raw["plugin_sha256"] == providers[mode]\n'
        '            assert proof["original_provider_control"] == (mode == "control")')
    source = replace_once(source, '    manifest["status"] = "completed"',
        '''    assert providers["control"] != providers["shape"]
    assert providers["control"] == "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"
    provider = providers["shape"]
    manifest["status"] = "completed"''')
    source = replace_once(source, 'context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,',
        'context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,\n'
        '                  original_baseline_provider_sha256=providers["control"],')
    return source.replace("bracketed_separate_load_runtime_flag_screen_not_same_load_windows",
                          "bracketed_separate_load_original_provider_net_screen")


def main():
    pairs = (
        ("run_dense_output_shape_model.py", "run_dense_output_shape_static_model.py", model_source),
        ("run_dense_output_shape_screen.py", "run_dense_output_shape_static_screen.py", screen_source),
    )
    for old, new, prepare in pairs:
        target = ROOT / new
        if target.exists():
            raise FileExistsError(target)
        with target.open("x") as stream:
            stream.write(prepare((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
