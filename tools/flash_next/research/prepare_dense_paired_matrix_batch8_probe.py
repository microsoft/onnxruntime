from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "run_dense_paired_matrix_reset.py").read_text()
    source = source.replace("dense_paired_matrix_reset", "dense_paired_matrix_batch8")
    source = source.replace("dense-paired-matrix-reset", "dense-paired-matrix-batch8")
    source = source.replace("DensePairedMatrixReset", "DensePairedMatrixBatch8")
    cases_start = source.index("CASES = ")
    cases_end = source.index("\nFLAGS =", cases_start)
    source = source[:cases_start] + "CASES = ((640, 2560),)" + source[cases_end:]
    source = source.replace("dense-paired-matrix-batch8-v1-", "dense-paired-matrix-batch8-probe-v1-")
    source = replace_once(source,
        'ROOT / "test_dense_paired_matrix_batch8.py", ROOT / "dense_paired_matrix_compare.cpp",',
        'ROOT / "test_dense_paired_matrix_batch8.py", ROOT / "dense_paired_matrix_compare.cpp",\n'
        '                ROOT / "prepare_dense_paired_matrix_batch8_probe.py",')
    output = ROOT / "run_dense_paired_matrix_batch8_probe.py"
    with output.open("x") as stream:
        stream.write(source)
    print(output.name)
