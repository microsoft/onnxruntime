from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def transform(source):
    source = source.replace("dense-replay-scratch-v1", "dense-output-shape-v2")
    source = source.replace("qwen38-int4-dense-replay-scratch/build-v1",
                            "qwen38-int4-dense-output-shape/build-v2")
    source = source.replace("dense_replay_scratch", "dense_output_shape")
    source = source.replace("scratch", "shape").replace("ScratchExact", "OutputShapeExact")
    return source


def model_source(source):
    source = transform(source)
    source = replace_once(source, '    build_path = DIRECTORY / "build-manifest.json"', '''
    ranks_path = ROOT / f"{VARIANT}-ranks-native-gpu7-20261031.qualification.json"
    ranks = json.loads(ranks_path.read_text())
    assert ranks["status"] == "dense_output_shape_rank1_fallback_rank234_finite_bitwise_qualified"
    assert ranks["finite_bitwise_comparisons"] == 24543744
    assert ranks["provider_sha256"] == native["provider_sha256"]
    for name, digest in ranks["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    for name, digest in ranks["fixture_sha256"].items():
        assert sha(Path(name)) == digest, name
    build_path = DIRECTORY / "build-manifest.json"''')
    source = replace_once(source, '        command[-1] = command[-1].replace(marker, "--tokens 36 --repetitions 1")',
        '''        command[-1] = command[-1].replace(marker, "--tokens 36 --repetitions 1")
        assert "ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATS_FILE" not in command[-1]
        command[-1] = ("export ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATS_FILE="
                       f"/evidence/webgpu-perf-oct20/{prefix.name}-shape-stats.jsonl; " + command[-1])''')
    source = replace_once(source, "        statistics.append(stats_path)",
        '''        shape_path = ROOT / (prefix.name + "-shape-stats.jsonl")
        shape_rows = [json.loads(s) for s in shape_path.read_text().splitlines()]
        assert len(shape_rows) == 172, len(shape_rows)
        assert all(row["fast_shape_hits"] >= 0 for row in shape_rows)
        assert sum(row["fast_shape_hits"] > 0 for row in shape_rows) >= 171
        result["observed_shape_kernels"] = len(shape_rows)
        result["fast_output_shape_instances"] = sum(row["fast_shape_hits"] > 0 for row in shape_rows)
        result["fast_output_shape_calls"] = sum(row["fast_shape_hits"] for row in shape_rows)
        statistics.extend([stats_path, shape_path])''')
    source = replace_once(source, "               build_path, template_path, controls_path,",
                          "               build_path, ranks_path, template_path, controls_path,")
    return source


def main():
    pairs = (
        ("run_dense_replay_scratch_model.py", "run_dense_output_shape_model.py", model_source),
        ("run_dense_replay_scratch_screen.py", "run_dense_output_shape_screen.py", transform),
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
