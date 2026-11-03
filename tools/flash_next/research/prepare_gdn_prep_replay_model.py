from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("ready-replay-key-trim", "gdn-prep-node-replay")
            .replace("build_ready_replay_key_trim", "build_gdn_prep_node_replay")
            .replace("run_ready_replay_key", "run_gdn_prep_replay")
            .replace("trim", "prep"))


def model_source(source):
    source = base(source)
    start = source.index("    command[-1] = (")
    end = source.index('    if phase == "coverage":', start)
    source = source[:start] + '''    command[-1] = f"export {FLAG}={int(mode == 'prep')}; " + command[-1]
''' + source[end:]
    source = replace_once(source,
        '    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"',
        '    native_path = ROOT / f"{VARIANT}-native-gpu7-20261101.qualification.json"')
    start = source.index('    assert native["status"] ==')
    end = source.index('    for name, digest in native["source_sha256"].items():', start)
    source = source[:start] + '''    assert native["status"] == "gdn_prep_replay_changing_gpu_inputs_finite_bitwise_qualified"
    assert native["finite_bitwise_comparisons"] == 181043200
    assert native["constructor_selector_modes"] == [0, 1]
    assert native["distinct_live_gpu_banks"] == 3
    assert native["invalid_constructor_selector_rejected"]
    negative_path = ROOT / "gdn-prep-node-replay-v2-negative-gpu7-20261101.qualification.json"
    negative = json.loads(negative_path.read_text())
    assert negative["status"] == "gdn_prep_replay_original_layout_and_fresh_selector_rejections_qualified"
    assert negative["provider_sha256"] == native["provider_sha256"]
    for name, digest in negative["source_sha256"].items():
        assert sha(Path(name)) == digest, name
''' + source[end:]
    source = replace_once(source, '    assert sha(build_path) == native["build_manifest_sha256"]',
                          '    assert sha(build_path) == native["source_sha256"][str(build_path)]')
    start = source.index("def verify_coverage(")
    end = source.index("\ndef main():", start)
    source = source[:start] + '''def verify_coverage(rows):
    dense = [row for row in rows if row.get("operator") == "MatMulNBitsExact"]
    prep = [row for row in rows if row.get("operator") == "GdnProjPrepF16Exact"]
    assert len(dense) == 171, len(dense)
    assert len(prep) == 36, len(prep)
    assert all(row["hits"] > 0 and row["binding_hits"] > 0 and
               row["rebinds"] + row["binding_hits"] == row["hits"] for row in dense + prep)
    return dict(actual_dense_replay_instances=len(dense),
                dense_replay_hits=sum(row["hits"] for row in dense),
                dense_binding_hits=sum(row["binding_hits"] for row in dense),
                actual_gdn_prep_replay_instances=len(prep),
                gdn_prep_replay_hits=sum(row["hits"] for row in prep),
                gdn_prep_binding_hits=sum(row["binding_hits"] for row in prep),
                actual_replay_path_coverage_only=True, not_a_standard_tps_claim=True)

''' + source[end:]
    start = source.index("    key_path =")
    end = source.index('    log = prefix.with_suffix(".log").read_text()', start)
    source = source[:start] + source[end:]
    source = replace_once(source,
        "build_path, candidate_native_path, original_native_path,",
        "build_path, candidate_native_path, original_native_path, negative_path,")
    source = replace_once(source, "process_static_experiment=True,", "constructor_static_experiment=True,")
    return replace_once(source, "                  ready_replay_key_coverage=key_coverage,\n", "")


def main():
    for old, new, transform in (
        ("run_ready_replay_key_model.py", "run_gdn_prep_replay_model.py", model_source),
        ("run_ready_replay_key_screen.py", "run_gdn_prep_replay_screen.py", base)):
        target = ROOT / new
        with target.open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
