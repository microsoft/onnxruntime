from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def transform(source):
    source = source.replace("dense-replay-scratch", "dense-recent-binding")
    source = source.replace("dense_replay_scratch", "dense_recent_binding")
    source = source.replace("scratch", "recent").replace("ScratchExact", "RecentExact")
    return source


def model_source(source):
    source = transform(source)
    source = replace_once(source,
        "dense_recent_binding_changing_inputs_finite_bitwise_qualified",
        "dense_recent_binding_fast_path_changing_inputs_finite_bitwise_qualified")
    source = replace_once(source, '    build_path = DIRECTORY / "build-manifest.json"', '''
    bank_path = ROOT / f"{VARIANT}-bank3-native-gpu7-20261031.qualification.json"
    bank = json.loads(bank_path.read_text())
    assert bank["status"] == "dense_recent_binding_distinct_bank_miss_path_finite_bitwise_qualified"
    assert bank["finite_bitwise_comparisons"] == 116582784
    assert bank["provider_sha256"] == native["provider_sha256"]
    bank_directory = DIRECTORY.parent / "binding-proof-v1"
    assert sha(bank_directory / "build-manifest.json") == bank["build_manifest_sha256"]
    assert sha(bank_directory / "native_compare") == bank["comparator_sha256"]
    bank_build = json.loads((bank_directory / "build-manifest.json").read_text())
    for name, digest in bank_build["source_sha256"].items():
        assert sha(bank_directory / name) == digest, name
    build_path = DIRECTORY / "build-manifest.json"''')
    source = replace_once(source, "    return dict(actual_private_dense_replay_instances=len(dense),",
        '''    fast_hits = sum(row["recent_binding_hits"] for row in dense)
    binding_hits = sum(row["binding_hits"] for row in dense)
    assert 0 < fast_hits <= binding_hits
    return dict(recent_binding_hits=fast_hits, binding_hits=binding_hits,
                recent_binding_fraction=fast_hits / binding_hits,
                actual_private_dense_replay_instances=len(dense),''')
    source = replace_once(source, "               build_path, template_path, controls_path,",
                          "               build_path, bank_path, template_path, controls_path,")
    return source


def main():
    pairs = (
        ("run_dense_replay_scratch_model.py", "run_dense_recent_binding_model.py", model_source),
        ("run_dense_replay_scratch_screen.py", "run_dense_recent_binding_screen.py", transform),
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
