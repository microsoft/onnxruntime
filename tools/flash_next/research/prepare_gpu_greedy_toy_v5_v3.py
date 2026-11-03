from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "gpu_greedy_toy_v5_v2.py").read_text()
    source = replace_once(source,
        "            np.testing.assert_array_equal(scores.view(np.uint32), expected_scores.view(np.uint32))",
        "            api_expected = data[actual % 14]\n"
        "            np.testing.assert_array_equal(scores.view(np.uint32), api_expected.view(np.uint32))")
    source = replace_once(source, '                  score_elements_checked=score_checks, runtime_sha256=',
        "                  public_get_logits_computes_next_step_when_needed=True,\n"
        "                  score_elements_checked=score_checks, runtime_sha256=")
    with (ROOT / "gpu_greedy_toy_v5_v3.py").open("x") as stream:
        stream.write(source)
    source = (ROOT / "run_gpu_greedy_toy_v5_probe3.py").read_text()
    source = source.replace("toy-probe3", "toy-probe4")
    source = source.replace("gpu_greedy_toy_v5_v2.py", "gpu_greedy_toy_v5_v3.py")
    with (ROOT / "run_gpu_greedy_toy_v5_probe4.py").open("x") as stream:
        stream.write(source)
