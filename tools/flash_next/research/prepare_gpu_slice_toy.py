from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "gpu_greedy_toy_v5_v5.py").read_text()
    source = replace_once(source, "gen.append_tokens([14 + case])", "gen.append_tokens([28, 14 + case])")
    source = replace_once(source, "gen.append_tokens([27])", "gen.append_tokens([28, 27])")
    source = replace_once(source, "            score_checks += n", """            score_checks += n

            initial = generator(case)
            copied = np.asarray(initial.get_logits(), dtype=np.float32).reshape(-1, n)[0]
            expected_nan = np.isnan(data[case])
            np.testing.assert_array_equal(np.isnan(copied), expected_nan)
            copied_bits, source_bits = copied.view(np.uint32), data[case].view(np.uint32)
            np.testing.assert_array_equal(copied_bits[~expected_nan], source_bits[~expected_nan])
            differences = int(np.count_nonzero(copied_bits[expected_nan] != source_bits[expected_nan]))
            if dtype == np.float32 and differences:
                raise RuntimeError("F32 initial sliced scores changed NaN payloads")
            nan_payload_differences += differences
            score_checks += n""")
    source = replace_once(source, '                  fused_flag=os.environ["ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH"],',
        '                  fused_flag=os.environ["ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH"],\n'
        '                  gpu_slice_flag=os.environ["ORT_GENAI_WEBGPU_GPU_SLICE_COPY"],')
    with (ROOT / "gpu_slice_toy.py").open("x") as stream:
        stream.write(source)
