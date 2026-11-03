import os
from pathlib import Path

from build_qmoe_node_replay import replace_once
import prepare_int8_lm_head_full_a_native as original
from prepare_int8_lm_head_full_a_native import MODEL, PROOF, region_sha


ROOT = Path(__file__).parent
DATA = original.REPO / "fixtures-v2"
RUNNER = ROOT / "run_int8_lm_head_full_a_native_v2.py"


def runner_source():
    source = (ROOT / "run_int8_lm_head_full_a_native.py").read_text()
    source = replace_once(source,
        "from prepare_int8_lm_head_full_a_native import DATA, MODEL, PROOF, region_sha",
        "from prepare_int8_lm_head_full_a_native_v2 import DATA, MODEL, PROOF, region_sha")
    source = replace_once(source, 'REFERENCES = PROOF / "original-references"',
                          'REFERENCES = PROOF / "original-references-v2"')
    return source.replace("int8-lm-head-full-a-v1-native", "int8-lm-head-full-a-v2-native")


if __name__ == "__main__":
    if RUNNER.exists():
        raise FileExistsError(RUNNER)
    original.DATA = DATA
    original.prepare_fixtures()
    link = DATA / "model.onnx.data"
    assert link.is_symlink() and link.samefile(MODEL / "model.onnx.data")
    link.unlink()
    link.symlink_to(os.path.relpath(MODEL / "model.onnx.data", DATA))
    assert link.samefile(MODEL / "model.onnx.data")
    with RUNNER.open("x") as stream:
        stream.write(runner_source())
    print(RUNNER)
