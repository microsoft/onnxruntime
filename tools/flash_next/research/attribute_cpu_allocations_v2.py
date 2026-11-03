import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).parent
IMAGE = "sha256:254075f70048e478f1603e693129432708cfe8da130870f0fc852f4909c7f454"
LIBRARIES = {
    "opt/conda/lib/libstdc++.so.6.0.34":
        "722f1edfb4497bfa7ffbd0b6b1d951123b2e77521e133f2d0b8b9b77dbe71411",
    "opt/conda/lib/libgcc_s.so.1":
        "bba953006263120beb520a58fe9a9cf98d3ec712cf7cb1464d58b974e571a1be",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def allocation_samples(text, interval):
    begin, end = interval
    records = []
    sample_count = 0
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if not lines or "cpu-clock:u:" not in lines[0]:
            continue
        match = re.search(r"\s(\d+\.\d+):", lines[0])
        if match is None or not begin <= float(match[1]) <= end:
            raise RuntimeError("CPU event outside the strict marker interval")
        sample_count += 1
        frames = []
        for line in lines[1:]:
            match = re.match(r"\s*[0-9a-f]+ (.+) \((.*)\)$", line)
            if match:
                frames.append(re.sub(r"\+0x[0-9a-f]+$", "", match[1]))
        if "malloc" not in frames:
            continue
        callers = frames[frames.index("malloc") + 1:]
        allocation_site = next(
            (frame for frame in callers if not frame.startswith("operator new")),
            None,
        )
        records.append({
            "timestamp": float(re.search(r"\s(\d+\.\d+):", lines[0])[1]),
            "allocation_site": allocation_site,
            "dawn_ancestor": next(
                (frame for frame in callers if frame.startswith("dawn::")), None),
            "ort_ancestor": next(
                (frame for frame in callers if "onnxruntime::" in frame), None),
            "callers": callers,
        })
    return sample_count, records


def main():
    symbol_root = ROOT / "cpu-decode-symbol-root-v2"
    for relative, expected in LIBRARIES.items():
        if sha(symbol_root / relative) != expected:
            raise RuntimeError(f"Approved-image symbol library changed: {relative}")
    output = ROOT / "cpu-allocation-attribution-symbols-v2.json"
    if output.exists():
        raise FileExistsError(output)
    evidence = [Path(__file__), *(symbol_root / relative for relative in LIBRARIES)]
    reports = {}
    for name in ("cpu-decode-sample-v2", "cpu-decode-32k-retained-v2"):
        outcome_path = ROOT / f"{name}-outcome.json"
        outcome = json.loads(outcome_path.read_text())
        data = ROOT / f"{name}.perf.data"
        expected = outcome["source_and_evidence_sha256"][str(data)]
        if sha(data) != expected or outcome["samples"] < 100:
            raise RuntimeError("Changed or insufficient qualified capture")
        original_report = ROOT / f"{name}.perf-report.txt"
        if "# Total Lost Samples: 0" not in original_report.read_text():
            raise RuntimeError("Original CPU capture is not zero-loss qualified")
        command = [
            "perf", "script", "-i", str(data), "--symfs", str(symbol_root),
            "--time", outcome["interval"],
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stderr.strip():
            raise RuntimeError(result.stderr)
        callchains = ROOT / f"{name}-callchains-symbols-v2.txt"
        if callchains.read_text() != result.stdout:
            raise RuntimeError("Derived callchains differ from fresh strict-window decode")
        count, allocations = allocation_samples(
            result.stdout, tuple(map(float, outcome["interval"].split(","))))
        if count != outcome["samples"] or not allocations:
            raise RuntimeError("Strict CPU sample count changed")
        if any(record["allocation_site"] is None for record in allocations):
            raise RuntimeError("Allocation callers remain truncated")
        dawn_count = sum(record["dawn_ancestor"] is not None for record in allocations)
        reports[name] = {
            "user_cpu_samples": count,
            "malloc_samples": len(allocations),
            "recovered_allocation_callers": len(allocations),
            "malloc_with_dawn_ancestor": dawn_count,
            "dawn_malloc_percent_of_user_cpu_samples": 100 * dawn_count / count,
            "perf_command": command,
            "allocations": allocations,
        }
        evidence.extend((outcome_path, data, original_report, callchains))
    report = {
        "status": "qualified_capture_offline_allocation_callers_recovered",
        "diagnostic_only": True,
        "performance_qualified": False,
        "quality_qualified": False,
        "approved_image_id": IMAGE,
        "symbol_libraries_sha256": LIBRARIES,
        "captures": reports,
        "limitations": [
            "No new model run or timing gain.",
            "Only strict-window, zero-loss original captures are attributed.",
            "Sampled user CPU fractions are not wall-time fractions or predicted TPS gains.",
            "Capture build IDs do not independently verify each mapped DSO.",
            "Source and copied libraries match the current pinned approved image.",
            "The separate lossy 32-KiB-stack capture remains rejected.",
            "Deep callers can still exceed the original 8-KiB stack snapshot.",
        ],
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({name: {key: value for key, value in row.items()
                            if key not in ("allocations", "perf_command")}
                      for name, row in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
