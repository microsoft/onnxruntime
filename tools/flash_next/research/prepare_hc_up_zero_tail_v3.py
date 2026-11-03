import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def fixture_source(source):
    source = source.replace("actual-fixtures-v2", "actual-fixtures-v3")
    source = source.replace("hc-up-zero-tail-actual-v2-generation.json", "hc-up-zero-tail-actual-v3-generation.json")
    source = replace_once(source, "from pathlib import Path", "from pathlib import Path\nimport subprocess")
    return replace_once(
        source, "                alias.hardlink_to(source)",
        '                try:\n'
        '                    alias.hardlink_to(source)\n'
        '                except PermissionError:\n'
        '                    subprocess.run(["sudo", "-n", "ln", str(source), str(alias)], check=True)')


def controller_source(source):
    source = source.replace("actual-screen-v2", "actual-screen-v3")
    source = source.replace('VARIANT = "hc-up-zero-tail-v2"', 'VARIANT = "hc-up-zero-tail-v3"')
    source = source.replace("hc-up-zero-tail-actual-v2-generation.json", "hc-up-zero-tail-actual-v3-generation.json")
    return replace_once(source, "from prepare_hc_up_zero_tail_actual_v2 import range_sha",
                         "from prepare_hc_up_zero_tail_actual_v3 import range_sha")


def main():
    parents = (ROOT / "prepare_hc_up_zero_tail_actual_v2.py", ROOT / "run_hc_up_zero_tail_v2.py")
    outputs = (ROOT / "prepare_hc_up_zero_tail_actual_v3.py", ROOT / "run_hc_up_zero_tail_v3.py")
    report = ROOT / "hc-up-zero-tail-v3-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh protected-hardlink HC-up generation required")
    for parent, output, transform in zip(parents, outputs, (fixture_source, controller_source), strict=True):
        source = transform(parent.read_text())
        compile(source, str(output), "exec")
        with output.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Original read-only weights are root-owned and Linux protects hardlink creation. Use the existing repository sudo-n ln permission fallback, verify original inode/ranges, and leave both previous failed preparations sealed.",
                   "source_sha256": {str(path): sha(path) for path in (Path(__file__), *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
