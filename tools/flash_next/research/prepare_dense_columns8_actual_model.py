from pathlib import Path

from build_qmoe_node_replay import replace_once
from prepare_dense_columns16_model_proofs import model_source, screen_source


ROOT = Path(__file__).parent


def actual_source():
    source = (ROOT / "run_dense_columns8_batch.py").read_text()
    source = source.replace("dense-columns8-batch-v1", "dense-columns8-actual-batch-v1")
    source = source.replace('"synthetic-batch-v1"', '"actual-batch-v1"')
    source = source.replace('"dense-columns16-v1-outcome.json"', '"dense-paired-matrix-reset-actual-v1-outcome.json"')
    source = source.replace("dense_columns16_original_provider_synthetic_shapes_supported",
                            "packed_dense_original_provider_actual_weights_supported")
    source = source.replace("qwen38-int4-dense-columns16/fixtures-v1",
                            "qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1")
    source = replace_once(source, "for k, n in CASES]", "for k, n in CASES if n != 1]")
    source = source.replace("columns8_batched_synthetic_and_default_supported",
                            "columns8_batched_actual_and_default_supported")
    return source.replace("actual_model_weights_used=False", "actual_model_weights_used=True")


def adapt(source):
    return source.replace("columns16", "columns8").replace("Columns16", "Columns8").replace("COLUMNS16", "COLUMNS8")


def model():
    source = adapt(model_source())
    source = source.replace('"dense-columns8-actual-v1-outcome.json"', '"dense-columns8-actual-batch-v1-outcome.json"')
    source = source.replace('"dense-columns8-v1-outcome.json"', '"dense-columns8-batch-v1-outcome.json"')
    start = source.index('    assert actual["status"]')
    end = source.index("    for proof in (actual, synthetic, invalid):", start)
    source = source[:start] + '''    assert actual["status"] == "columns8_batched_actual_and_default_supported"
    assert synthetic["status"] == "columns8_batched_synthetic_and_default_supported"
    assert invalid["status"] == "fresh_invalid_columns8_selector_rejected"
    assert [phase["phase"] for phase in actual["phases"]] == ["capture", "default", "candidate"]
    assert [phase["phase"] for phase in synthetic["phases"]] == ["capture", "default", "candidate"]
    assert all(len(phase["shapes"]) == 8 and all(row["mismatches"] == 0 for row in phase["shapes"])
               for phase in actual["phases"])
    assert all(len(phase["shapes"]) == 9 and all(row["mismatches"] == 0 for row in phase["shapes"])
               for phase in synthetic["phases"])
    assert all(row["selected_dispatches"] == (64 if row["n"] >= 2560 else 0)
               for row in actual["phases"][-1]["shapes"])
''' + source[end:]
    source = replace_once(source, '    assert counts[selected] == 159 and counts[original] == 0, counts',
        '''    assert counts[selected] == (158 if mode == "columns8" else 159), counts
    assert counts[original] == (1 if mode == "columns8" else 0), counts''')
    return source


def invalid():
    source = adapt((ROOT / "run_dense_columns16_invalid.py").read_text())
    source = source.replace("run_dense_columns8_synthetic", "run_dense_columns8_batch")
    source = source.replace('"dense-columns8-actual-v1-outcome.json"', '"dense-columns8-actual-batch-v1-outcome.json"')
    source = source.replace("dense_columns8_original_provider_actual_weights_supported",
                            "columns8_batched_actual_and_default_supported")
    source = source.replace('BUILD.parent / "actual-fixtures-v1/k640-n2560.onnx"',
        'Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1/k640-n2560.onnx")')
    return source


if __name__ == "__main__":
    for name, source in (
        ("run_dense_columns8_actual_batch.py", actual_source()),
        ("run_dense_columns8_invalid.py", invalid()),
        ("run_dense_columns8_model.py", model()),
        ("run_dense_columns8_screen.py", adapt(screen_source())),
    ):
        target = ROOT / name
        with target.open("x") as stream:
            stream.write(source)
        print(target.name)
