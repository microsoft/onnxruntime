import json
from pathlib import Path

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT


if __name__ == "__main__":
    pairs = (
        ("benchmark_context_flag_cache.py", "benchmark_bind_group_inline_key.py"),
        ("run_context_flag_cache_model.py", "run_bind_group_inline_key_model.py"),
        ("test_context_flag_cache_model.py", "test_bind_group_inline_key_model.py"),
    )
    report_path = ROOT / "bind-group-inline-key-v1-model-generation.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    if any((ROOT / output).exists() for _, output in pairs):
        raise FileExistsError("A generated inline-key model source already exists")
    generated = {}
    for source_name, output_name in pairs:
        original = (ROOT / source_name).read_text()
        source = original
        for before, after in (
            ("context_flag_cache", "bind_group_inline_key"),
            ("context-flag-cache", "bind-group-inline-key"),
            ("ContextFlagCache", "BindGroupInlineKey"),
            ("ORT_WEBGPU_STATIC_CONTEXT_FLAGS", "ORT_WEBGPU_BIND_GROUP_INLINE_KEY"),
            ("static-context selector", "inline-key selector"),
        ):
            source = source.replace(before, after)
        if source == original:
            raise RuntimeError(f"No specialization occurred: {source_name}")
        compile(source, str(ROOT / output_name), "exec")
        generated[output_name] = source
    for name, source in generated.items():
        with (ROOT / name).open("x") as stream:
            stream.write(source)
    paths = [Path(__file__), *[ROOT / name for pair in pairs for name in pair]]
    report = dict(status="inline_key_model_controller_specialized",
        sealed_context_controller_unchanged=True, original_request_implementation_unchanged=True,
        source_and_evidence_sha256={str(path): sha(path) for path in paths})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
