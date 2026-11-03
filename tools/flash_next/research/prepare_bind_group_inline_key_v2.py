import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import ROOT


if __name__ == "__main__":
    pairs = (
        ("build_bind_group_inline_key.py", "build_bind_group_inline_key_v2.py"),
        ("run_bind_group_inline_key_native.py", "run_bind_group_inline_key_native_v2.py"),
        ("run_bind_group_inline_key_model.py", "run_bind_group_inline_key_model_v2.py"),
        ("test_bind_group_inline_key_model.py", "test_bind_group_inline_key_model_v2.py"),
        ("queue_bind_group_inline_key_model.py", "queue_bind_group_inline_key_model_v2.py"),
    )
    report_path = ROOT / "bind-group-inline-key-v2-model-generation.json"
    if report_path.exists() or any((ROOT / output).exists() for _, output in pairs):
        raise FileExistsError("A v2 specialization already exists")
    generated = {}
    for source_name, output_name in pairs:
        source = (ROOT / source_name).read_text()
        source = source.replace("qwen38-int4-bind-group-inline-key/build-v1",
                                "qwen38-int4-bind-group-inline-key/build-v2")
        source = source.replace("bind-group-inline-key-v1", "bind-group-inline-key-v2")
        if source_name == "build_bind_group_inline_key.py":
            source = replace_once(source,
                '    source = replace_once(source, "struct WebGpuContextFactory::BindGroupCache {",\n'
                '                          HELPERS + "struct WebGpuContextFactory::BindGroupCache {")',
                '    signature = "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {"\n'
                '    source = replace_once(source, signature, HELPERS + signature +\n'
                '                          "\\n  (void)BindGroupInlineKeyEnabled();")')
            source = replace_once(source, "process_static_selector=True, stack_binding_capacity=32,",
                "process_static_selector=True, invalid_selector_validated_before_program_work=True,\n"
                "        stack_binding_capacity=32,")
        if source_name in ("run_bind_group_inline_key_model.py", "queue_bind_group_inline_key_model.py"):
            source = source.replace("run_bind_group_inline_key_native", "run_bind_group_inline_key_native_v2")
        if source_name in ("test_bind_group_inline_key_model.py", "queue_bind_group_inline_key_model.py"):
            source = source.replace("run_bind_group_inline_key_model", "run_bind_group_inline_key_model_v2")
        compile(source, str(ROOT / output_name), "exec")
        generated[output_name] = source
    for name, source in generated.items():
        with (ROOT / name).open("x") as stream:
            stream.write(source)
    paths = [Path(__file__), ROOT / "benchmark_bind_group_inline_key.py",
             *[ROOT / name for pair in pairs for name in pair]]
    report = dict(status="inline_key_model_controller_specialized",
        sealed_v1_sources_unchanged=True, original_request_implementation_unchanged=True,
        early_validation_before_program_work=True,
        source_and_evidence_sha256={str(path): sha(path) for path in paths})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
