import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def fixture_source(source):
    source = source.replace("actual-fixtures-v1", "actual-fixtures-v2")
    source = source.replace("hc-up-zero-tail-actual-v1-generation.json", "hc-up-zero-tail-actual-v2-generation.json")
    source = source.replace("import os\n", "")
    source = replace_once(source, "alias.symlink_to(os.path.relpath(source, DATA))", "alias.hardlink_to(source)")
    return replace_once(
        source, '"no_model_weight_copies": True, "quality_qualified": False',
        '"no_model_weight_copies": True, "external_data_packaging": "in-directory original-inode hardlinks",\n'
        '              "quality_qualified": False')


def controller_source(source):
    source = source.replace("actual-screen-v1", "actual-screen-v2")
    source = source.replace('VARIANT = "hc-up-zero-tail-v1"', 'VARIANT = "hc-up-zero-tail-v2"')
    source = source.replace("hc-up-zero-tail-actual-v1-generation.json", "hc-up-zero-tail-actual-v2-generation.json")
    return replace_once(source, "from prepare_hc_up_zero_tail_actual import range_sha",
                         "from prepare_hc_up_zero_tail_actual_v2 import range_sha")


def main():
    parents = (ROOT / "prepare_hc_up_zero_tail_actual.py", ROOT / "run_hc_up_zero_tail_v1.py")
    outputs = (ROOT / "prepare_hc_up_zero_tail_actual_v2.py", ROOT / "run_hc_up_zero_tail_v2.py")
    report = ROOT / "hc-up-zero-tail-v2-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh hardlink-packaged HC-up generation required")
    for parent, output, transform in zip(parents, outputs, (fixture_source, controller_source), strict=True):
        source = transform(parent.read_text())
        compile(source, str(output), "exec")
        with output.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "ORT correctly rejects external symlinks escaping a model directory. Package immutable original-inode hardlinks inside the diagnostic directory; do not disable path validation, copy weights, or change shader/native scope.",
                   "source_sha256": {str(path): sha(path) for path in (Path(__file__), *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
