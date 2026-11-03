from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "run_gpu_greedy_argmax_initial_model.py").read_text()
    source = source.replace("gpu-greedy-argmax/build-v3", "gpu-greedy-argmax/build-v4")
    source = source.replace("gpu-greedy-argmax-v3-", "gpu-greedy-argmax-v4-")
    source = replace_once(source, '    phases = []',
        '    mask_path = ROOT / "gpu-greedy-argmax-v4-mask-native-outcome.json"\n'
        '    mask = json.loads(mask_path.read_text())\n'
        '    if mask["status"] != "gpu_min_length_cast_where_argmax_cpu_ids_exact":\n'
        '        raise RuntimeError("GPU min-length native qualification missing")\n'
        '    for path, digest in mask["source_and_evidence_sha256"].items():\n'
        '        if sha(Path(path)) != digest:\n'
        '            raise RuntimeError(f"Changed GPU mask evidence: {path}")\n'
        '    sources.extend((mask_path, ROOT / "prepare_gpu_greedy_argmax_v4_model.py"))\n'
        '    phases = []')
    source = replace_once(source, '        fallbacks = log.count("WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK")',
        '        fallbacks = log.count("WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK")\n'
        '        masks = re.findall(r"^WEBGPU_GREEDY_MIN_LENGTH rows=1 vocabulary=248320$", log, re.M)')
    source = replace_once(source, "        if len(selected) != expected or fallbacks != 0:",
        "        if len(selected) != expected or len(masks) != expected or fallbacks != 0:")
    source = replace_once(source, "selected_argmax_calls=len(selected), cpu_fallback_calls=fallbacks,",
        "selected_argmax_calls=len(selected), selected_min_length_masks=len(masks), cpu_fallback_calls=fallbacks,")
    with (ROOT / "run_gpu_greedy_argmax_initial_model_v4.py").open("x") as stream:
        stream.write(source)
