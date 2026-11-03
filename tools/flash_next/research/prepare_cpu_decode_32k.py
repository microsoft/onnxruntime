import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import ROOT


if __name__ == "__main__":
    original = ROOT / "run_cpu_decode_sample_v2.py"
    output = ROOT / "run_cpu_decode_32k.py"
    report_path = ROOT / "cpu-decode-32k-retained-v1-generation.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("32K diagnostic specialization already exists")
    source = original.read_text()
    source = replace_once(source, 'VARIANT = "cpu-decode-sample-v2"', '''VARIANT = "cpu-decode-32k-retained-v1"
TEMPLATE = "gpu-slice-combined32k-v1-context32768-retained-flag1-trial0-trace0-gpu7"


def verify_placement(raw, guard):
    cpus = list(range(48, 56))
    if guard["gpu_index"] != 7 or guard["gpu_uuid"] != "GPU-0e70e969-bbfc-80c5-d582-88be134632ea":
        raise RuntimeError("Wrong GPU7 diagnostic adapter")
    if (raw["cpu_affinity"] != cpus or raw["worker_affinity_override"]["target_cpus"] != cpus or
            not raw["thread_cpu_affinity"] or any(mask != cpus for mask in raw["thread_cpu_affinity"].values())):
        raise RuntimeError("Diagnostic threads escaped the matched GPU7 CPU mask")
''')
    source = source.replace("parent.TEMPLATE", "TEMPLATE")
    source = source.replace("fused_slice_original_prefixes_qualified", "decode_stack_original_prefixes_qualified")
    source = source.replace('["taskset", "-c", "16-23", "bash", "-lc"]',
                            '["taskset", "-c", "48-55", "bash", "-lc"]')
    source = replace_once(source, '"--gpu-index", "1"', '"--gpu-index", "7"')
    source = replace_once(source, 'state["gpu_index"] != 1', 'state["gpu_index"] != 7')
    source = replace_once(source, "gpu_index=1,", "gpu_index=7,")
    source = replace_once(source, "parent.verify_placement(raw, state)", "verify_placement(raw, state)")
    source = replace_once(source,
        'parent.verify_tokens(raw, [s for s in references if s["context"] == 2048], 2048, 256)',
        'parent.verify_tokens(raw, [s for s in references if s["context"] == 32768], 32768, 256)')
    source = replace_once(source, "import run_cpu_decode_sample as parent",
                          "import run_cpu_decode_sample as parent\nfrom nvml_decode_clocks import sample_clocks")
    source = replace_once(source, '    command[-1] = f"export ORT_WEBGPU_CPU_PERF_MARKER_PREFIX=',
        '    command[-1] = parent.replace_once(command[-1], "--repetitions 3", "--repetitions 1")\n'
        '    command[-1] = f"export ORT_WEBGPU_CPU_PERF_MARKER_PREFIX=')
    source = replace_once(source, "            perf = subprocess.Popen(perf_command, stdout=log, stderr=log)",
        "            perf = subprocess.Popen(perf_command, stdout=log, stderr=log)\n"
        "            clocks = []\n"
        "            next_clock = time.monotonic()")
    source = replace_once(source, '            while not markers["stop"].exists():',
        '            while not markers["stop"].exists():\n'
        '                if time.monotonic() >= next_clock:\n'
        '                    clocks.append(sample_clocks(state["gpu_uuid"]))\n'
        '                    next_clock = time.monotonic() + .1')
    source = replace_once(source, '        interval = f\'{start["monotonic_ns"]',
        '        clock_path = Path(str(prefix) + "-clocks.json")\n'
        '        clock_path.write_text(json.dumps(clocks, indent=2) + "\\n")\n'
        '        interval = f\'{start["monotonic_ns"]')
    source = replace_once(source, 'Path(str(prefix) + ".perf.data"), report_path, *markers.values())',
        'Path(str(prefix) + ".perf.data"), report_path, clock_path, root / "nvml_decode_clocks.py", *markers.values())')
    compile(source, str(output), "exec")
    with output.open("x") as stream:
        stream.write(source)
    paths = (Path(__file__), original, output, ROOT / "nvml_decode_clocks.py")
    report = dict(status="cpu_32k_retained_diagnostic_specialized",
        sealed_2k_diagnostic_unchanged=True, original_request_implementation_unchanged=True,
        warmups=1, measured_requests=1, gpu_index=7, diagnostic_only=True,
        source_and_evidence_sha256={str(path): sha(path) for path in paths})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
