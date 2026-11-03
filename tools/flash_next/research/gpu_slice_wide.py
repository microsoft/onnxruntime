import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpu_slice_toy import fixtures


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    import onnxruntime_genai as og

    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    og.register_execution_provider_library("webgpu", str(args.plugin))
    folder = Path("/workspace/qwen38-int4-gpu-greedy-argmax/toy-v5-v2/float32-248320")
    config = og.Config(str(folder))
    config.overlay(json.dumps(dict(model=dict(context_length=16512))))
    model = og.Model(config)
    table = fixtures(248320, np.float32)
    for case in (1, 5, 8, 9):
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=16400, min_length=16400, do_sample=False)
        generator = og.Generator(model, params)
        prompt = np.full(16384, 28, dtype=np.int32)
        prompt[-1] = 14 + case
        generator.append_tokens(prompt)
        scores = np.asarray(generator.get_logits(), dtype=np.float32).reshape(-1)
        np.testing.assert_array_equal(scores.view(np.uint32), table[case].view(np.uint32))
        print(json.dumps(dict(case=case, exact_score_elements=len(scores))), flush=True)
        del generator
    mapped = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if line.endswith("/libonnxruntime-genai.so")}
    if len(mapped) != 1:
        raise RuntimeError("Expected one native runtime")
    runtime = Path(next(iter(mapped)))
    sources = (Path(__file__), args.plugin, folder / "model.onnx", folder / "genai_config.json")
    result = dict(status="wide_uint32_slice_scores_bitwise_exact", cases=4, score_elements_checked=993280,
        source_bytes=16384 * 248320 * 4, copied_bytes=248320 * 4,
        source_start_word=(16384 - 1) * 248320, source_word_count=16384 * 248320,
        runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
        source_and_evidence_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_and_evidence_sha256"}), flush=True)
