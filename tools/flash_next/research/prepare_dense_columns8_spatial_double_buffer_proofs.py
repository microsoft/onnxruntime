import ast
from pathlib import Path

from prepare_dense_columns16_spatial_double_buffer_proofs import batch_source as parent_source
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
GUARD_HELPER = '''
UUID = "GPU-68170cb3-c9ea-a588-7558-ae82635d9173"


def run_guard(command, prefix):
    from run_fp16_dot4_tree_probe import GUARD
    for suffix in (".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    child = subprocess.run(["python3", str(GUARD), "--gpu-index", "4",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    valid = ((proof["status"] == "completed" and child.returncode == 0)
             or (proof["status"] == "failed" and child.returncode == 2))
    if (not valid or proof.get("exit_code") != child.returncode or proof["gpu_uuid"] != UUID):
        raise RuntimeError(f"GPU4 proof child did not complete: {prefix.with_suffix('.log')}")
    return child.returncode, prefix.with_suffix(".log").read_text()

'''


def batch_source(actual=False):
    source = parent_source(actual)
    for before, after in (
        ("dense-columns16-spatial-double-buffer", "dense-columns8-spatial-double-buffer"),
        ("ORT_WEBGPU_DENSE_COLUMNS16_SPATIAL_DOUBLE_BUFFER", "ORT_WEBGPU_DENSE_COLUMNS8_SPATIAL_DOUBLE_BUFFER"),
        ("DenseColumns16SpatialDoubleBuffer", "DenseColumns8SpatialDoubleBuffer"),
        ("dense_columns16_spatial_double_buffer", "dense_columns8_spatial_double_buffer"),
        ("columns16_spatial_double_buffer_", "columns8_spatial_double_buffer_"),
        ("-gpu7", "-gpu4"),
    ):
        source = source.replace(before, after)
    source = replace_once(source,
        "from run_dense_columns16_synthetic import CASES, FLAGS, run_guard, UUID, workspace",
        "from run_dense_columns16_synthetic import CASES, FLAGS, workspace")
    source = replace_once(source, "ROOT = Path(__file__).parent", GUARD_HELPER + "ROOT = Path(__file__).parent")
    source = replace_once(source, "args = [workspace(binary), mode, workspace(library),",
                          'args = ["taskset", "-c", "56-63", workspace(binary), mode, workspace(library),')
    ast.parse(source)
    return source


if __name__ == "__main__":
    for actual in (False, True):
        name = ("run_dense_columns8_spatial_double_buffer_actual_batch.py" if actual
                else "run_dense_columns8_spatial_double_buffer_batch.py")
        with (ROOT / name).open("x") as stream:
            stream.write(batch_source(actual))
        print(name)
