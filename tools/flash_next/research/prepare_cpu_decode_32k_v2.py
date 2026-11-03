import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import ROOT


if __name__ == "__main__":
    original = ROOT / "run_cpu_decode_32k.py"
    output = ROOT / "run_cpu_decode_32k_v2.py"
    report_path = ROOT / "cpu-decode-32k-retained-v2-generation.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("32K v2 diagnostic specialization already exists")
    source = replace_once(original.read_text(), 'VARIANT = "cpu-decode-32k-retained-v1"',
                          'VARIANT = "cpu-decode-32k-retained-v2"')
    source = replace_once(source, "from nvml_decode_clocks import sample_clocks",
        "from nvml_decode_clocks import sample_clocks\nfrom cpu_decode_diagnostic_checks import verify_two_requests")
    source = replace_once(source,
        'parent.verify_tokens(raw, [s for s in references if s["context"] == 32768], 32768, 256)',
        'verify_two_requests(raw, [s for s in references if s["context"] == 32768], 32768)')
    source = replace_once(source, 'root / "nvml_decode_clocks.py", *markers.values()',
        'root / "nvml_decode_clocks.py", root / "cpu_decode_diagnostic_checks.py", *markers.values()')
    compile(source, str(output), "exec")
    with output.open("x") as stream:
        stream.write(source)
    paths = (Path(__file__), original, output, ROOT / "cpu_decode_diagnostic_checks.py")
    report = dict(status="cpu_32k_retained_diagnostic_specialized",
        sealed_2k_diagnostic_unchanged=True, original_request_implementation_unchanged=True,
        complete_warmup_and_measured_prefixes_required=True, measured_requests=1, gpu_index=7,
        diagnostic_only=True, source_and_evidence_sha256={str(path): sha(path) for path in paths})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
