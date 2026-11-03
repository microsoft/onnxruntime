import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/home/sayanshaw/repos/qwen38-int4-gpu-greedy-argmax/toy-v5-v2")


def fixtures(n, dtype):
    values = np.zeros((14, n), dtype=dtype)
    values[1, [1, 5]] = 3
    values[2] = -4
    values[2, 7] = -2
    values[3, 0] = np.nan
    values[3, 7] = 8
    values[4, 1] = np.nan
    values[4, 7] = 8
    values[5] = -1
    values[5, 1] = -0.0
    values[5, 2] = 0.0
    values[6, [4, 6]] = np.inf
    values[7] = -np.inf
    values[8] = np.nan
    values[9, -1] = np.nan
    values[9, 5] = np.nextafter(dtype(0), dtype(1))
    values[10] = -np.inf
    values[10, 2] = np.finfo(dtype).min
    values[11, -1] = 1000
    values[11, 4] = 900
    values[12, 26] = 4
    values[12, 5] = 3
    values[13, 29] = 4
    values[13, 5] = 3
    return values


def prepare():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ROOT.mkdir(parents=True, exist_ok=False)
    for dtype in (np.float16, np.float32):
        for n in (32, 2049, 248320):
            folder = ROOT / f"{np.dtype(dtype).name}-{n}"
            folder.mkdir()
            data = fixtures(n, dtype)
            graph = helper.make_graph([
                helper.make_node("Mod", ["input_ids", "period"], ["indices"], fmod=0),
                helper.make_node("Gather", ["table", "indices"], ["logits"], axis=0),
            ], "greedy_behavior_fixture",
                [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"])],
                [helper.make_tensor_value_info("logits", TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT,
                                              ["batch", "sequence", n])],
                [numpy_helper.from_array(data, "table"), numpy_helper.from_array(np.array(14, dtype=np.int64), "period")])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
            onnx.checker.check_model(model)
            onnx.save(model, folder / "model.onnx")
            config = dict(model=dict(type="llama", vocab_size=n, context_length=128,
                bos_token_id=16, eos_token_id=[0, n - 1], pad_token_id=0,
                decoder=dict(filename="model.onnx", num_hidden_layers=0, num_attention_heads=1,
                    num_key_value_heads=1, head_size=1, hidden_size=1,
                    inputs=dict(input_ids="input_ids"), outputs=dict(logits="logits"),
                    session_options=dict(intra_op_num_threads=1, inter_op_num_threads=1,
                        provider_options=[dict(webgpu=dict(enableGraphCapture="0", enableInt64="1",
                                                           validationMode="full"))]))),
                search=dict(max_length=8, min_length=8, do_sample=False, num_beams=1,
                            past_present_share_buffer=False))
            (folder / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n")


def scalar_argmax(values):
    best, index = float(values[0]), 0
    for i in range(1, len(values)):
        if float(values[i]) > best:
            best, index = float(values[i]), i
    return index


def run(models, output, plugin):
    import faulthandler
    faulthandler.enable()
    import onnxruntime_genai as og

    og.register_execution_provider_library("webgpu", str(plugin))
    checks, score_checks = [], 0
    for folder in models:
        config = json.loads((folder / "genai_config.json").read_text())
        n = config["model"]["vocab_size"]
        dtype = np.float16 if folder.name.startswith("float16") else np.float32
        data = fixtures(n, dtype).astype(np.float32)
        print(json.dumps(dict(stage="model", model=folder.name)), flush=True)
        model = og.Model(str(folder))

        def generator(case, **options):
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=8, min_length=8, do_sample=False, **options)
            print(json.dumps(dict(stage="generator", case=case)), flush=True)
            gen = og.Generator(model, params)
            # Input IDs modulo fourteen select the independent score fixture.
            gen.append_tokens([14 + case])
            return gen

        for case in range(14):
            gen = generator(case)
            gen.generate_next_token()
            actual = int(gen.get_next_tokens()[0])
            expected_scores = data[case].copy()
            expected_scores[[0, n - 1]] = np.finfo(np.float32).min
            expected = scalar_argmax(expected_scores)
            if actual != expected:
                raise RuntimeError(f"Toy ID mismatch: {folder.name} case={case} expected={expected} actual={actual}")
            scores = np.asarray(gen.get_logits(), dtype=np.float32).reshape(-1, n)[0]
            api_expected = data[actual % 14]
            np.testing.assert_array_equal(scores.view(np.uint32), api_expected.view(np.uint32))
            checks.append(dict(model=folder.name, case=case, token_id=actual))
            score_checks += n

        for options in (dict(repetition_penalty=2.0), dict(no_repeat_ngram_size=1)):
            gen = generator(12, **options)
            gen.generate_next_token()
            actual = int(gen.get_next_tokens()[0])
            if actual != 5:
                raise RuntimeError(f"CPU penalty/cache authority mismatch: {folder.name} options={options} actual={actual}")
            checks.append(dict(model=folder.name, cpu_penalty=options, token_id=actual))

        gen = generator(1)
        gen.generate_next_token()
        gen.get_logits()
        replacement = np.zeros((1, n), dtype=np.float32)
        replacement[0, 5] = 100
        gen.set_logits(replacement)
        gen.generate_next_token()
        if int(gen.get_next_tokens()[0]) != 5:
            raise RuntimeError(f"Stale cached token after CPU API score replacement: {folder.name}")
        checks.append(dict(model=folder.name, api_replacement=True, token_id=5))

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=8, min_length=8, do_sample=True, top_k=1, top_p=1.0)
        gen = og.Generator(model, params)
        gen.append_tokens([27])
        gen.generate_next_token()
        if int(gen.get_next_tokens()[0]) != 29:
            raise RuntimeError(f"CPU sampling/cache authority mismatch: {folder.name}")
        checks.append(dict(model=folder.name, sampling_top1=True, token_id=29))

    mapped = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if line.endswith("/libonnxruntime-genai.so")}
    if len(mapped) != 1:
        raise RuntimeError(f"Expected one mapped runtime: {mapped}")
    native = Path(next(iter(mapped)))
    result = dict(status="toy_runtime_ids_scores_cpu_authority_exact", checks=checks,
                  public_get_logits_computes_next_step_when_needed=True,
                  score_elements_checked=score_checks, runtime_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                  fused_flag=os.environ["ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH"],
                  source_and_evidence_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), plugin, *[p for folder in models for p in
                                                       (folder / "model.onnx", folder / "genai_config.json")]]})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], ids_checked=len(checks),
                         score_elements_checked=score_checks)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--models", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plugin", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        if not args.models or args.output is None or args.plugin is None:
            parser.error("--models, --output and --plugin are required")
        run(args.models, args.output, args.plugin)
