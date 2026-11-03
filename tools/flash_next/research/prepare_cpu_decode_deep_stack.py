import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def specialize(source):
    source = replace_once(source, 'VARIANT = "cpu-decode-sample-v2"',
                          'VARIANT = "cpu-decode-deep-stack-v1"')
    return replace_once(source, '"dwarf,8192"', '"dwarf,32768"')


if __name__ == "__main__":
    source = ROOT / "run_cpu_decode_sample_v2.py"
    output = ROOT / "run_cpu_decode_deep_stack_v1.py"
    with output.open("x") as stream:
        stream.write(specialize(source.read_text()))
    report = dict(status="deeper_user_stack_diagnostic_generated",
                  only_changes=["fresh evidence prefix", "DWARF stack bytes 8192 to 32768"],
                  no_performance_candidate_retry=True, headline_eligible=False,
                  source_sha256={str(path): sha(path) for path in (Path(__file__), source, output)})
    with (ROOT / "deep-stack-v1-generation.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
