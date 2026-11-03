from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("dense-replay-scratch-v1", "ready-replay-key-trim-v2")
            .replace("qwen38-int4-dense-replay-scratch/build-v1",
                     "qwen38-int4-ready-replay-key-trim/build-v2")
            .replace("build_dense_replay_scratch", "build_ready_replay_key_trim")
            .replace("run_dense_replay_scratch", "run_ready_replay_key")
            .replace("scratch", "trim"))


def model_source(source):
    source = base(source)
    source = replace_once(source,
        '    command[-1] = f"export {FLAG}={int(mode == \'trim\')}; " + command[-1]',
        '''    command[-1] = (
        f"export {FLAG}={int(mode == 'trim')}; "
        f"export ORT_WEBGPU_READY_REPLAY_KEY_STATS_FILE=/evidence/webgpu-perf-oct20/"
        f"{prefix.name}-key-stats.jsonl; " + command[-1])''')
    start, end = source.index("def verify_coverage("), source.index("\ndef main():")
    source = source[:start] + '''def verify_coverage(rows):
    dense = [row for row in rows if row.get("operator") == "MatMulNBitsExact"]
    assert len(dense) == 171, len(dense)
    assert all(row["hits"] > 0 and row["binding_hits"] > 0 and
               row["rebinds"] + row["binding_hits"] == row["hits"] for row in dense)
    return dict(actual_dense_replay_instances=len(dense),
                dense_replay_hits=sum(row["hits"] for row in dense),
                dense_binding_hits=sum(row["binding_hits"] for row in dense),
                actual_replay_path_coverage_only=True, not_a_standard_tps_claim=True)

''' + source[end:]
    source = replace_once(source,
        '    assert native["status"] == "dense_replay_trim_changing_inputs_finite_bitwise_qualified"\n'
        '    assert native["finite_bitwise_comparisons"] == 116582784',
        '''    assert native["status"] == "ready_replay_key_trim_process_modes_and_invalid_selector_qualified"
    assert native["finite_bitwise_comparisons"] == 233165568
    assert native["process_modes_tested"] == [0, 1]
    assert native["invalid_selector_rejected_in_fresh_process"]
    for name, digest in native["source_sha256"].items():
        assert sha(Path(name)) == digest, name''')
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
        '    log = prefix.with_suffix(".log").read_text()',
        '''    key_path = ROOT / (prefix.name + "-key-stats.jsonl")
    key_coverage = None
    if args.mode == "trim":
        rows = [json.loads(line) for line in key_path.read_text().splitlines()]
        assert rows and all(row["operator"] == "ReadyReplayWithoutKey" and
                            row["commands"] > 0 and row["key_bytes_avoided"] > 0
                            for row in rows), rows
        key_coverage = dict(commands=sum(row["commands"] for row in rows),
                            key_bytes_avoided=sum(row["key_bytes_avoided"] for row in rows))
        if args.phase == "coverage":
            assert key_coverage["commands"] >= result["dense_replay_hits"]
        statistics.append(key_path)
    else:
        assert not key_path.exists()
    log = prefix.with_suffix(".log").read_text()''')
    source = replace_once(source,
        "               build_path, template_path, controls_path, *prerequisites, *statistics,",
        "               build_path, candidate_native_path, original_native_path,\n"
        "               template_path, controls_path, *prerequisites, *statistics,")
    return replace_once(source,
        '                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],',
        '                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],\n'
        '                  original_provider_control=args.mode == "control", process_static_experiment=True,\n'
        '                  ready_replay_key_coverage=key_coverage,')


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
        '''    assert providers["control"] != providers["trim"]
    assert providers["control"] == "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"
    provider = providers["trim"]
    manifest["status"] = "completed"''')
    source = replace_once(source, 'context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,',
        'context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,\n'
        '                  original_baseline_provider_sha256=providers["control"],')
    return source.replace("bracketed_separate_load_runtime_flag_screen_not_same_load_windows",
                          "bracketed_separate_load_original_provider_net_screen")


def main():
    for old, new, prepare in (
            ("run_dense_replay_scratch_model.py", "run_ready_replay_key_model.py", model_source),
            ("run_dense_replay_scratch_screen.py", "run_ready_replay_key_screen.py", screen_source)):
        target = ROOT / new
        if target.exists():
            raise FileExistsError(target)
        with target.open("x") as stream:
            stream.write(prepare((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
