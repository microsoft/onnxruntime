from pathlib import Path

from build_qmoe_node_replay import replace_once
from prepare_int8_lm_head_full_a_native_v3 import runner_source as previous_source


ROOT = Path(__file__).parent
RUNNER = ROOT / "run_int8_lm_head_full_a_native_v4.py"


def runner_source():
    source = previous_source().replace("int8-lm-head-full-a-v3-native", "int8-lm-head-full-a-v4-native")
    source = replace_once(source, 'REFERENCES = PROOF / "original-references-v3"',
                          'REFERENCES = PROOF / "original-references-v4"')
    return replace_once(source, '            assert counts["MatMulNBits"] == 64 - selected_count, (row, counts)',
        '            if row["fp32"]:\n'
        '                assert selected_count == 0\n'
        '                assert counts["DP4AMatMulQuantize"] == 64, (row, counts)\n'
        '                assert counts["DP4AMatMulNBitsSmallMProgram"] == 64, (row, counts)\n'
        '            else:\n'
        '                assert counts["MatMulNBits"] == 64 - selected_count, (row, counts)')


if __name__ == "__main__":
    with RUNNER.open("x") as stream:
        stream.write(runner_source())
    print(RUNNER)
