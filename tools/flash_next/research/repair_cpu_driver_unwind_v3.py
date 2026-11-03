import collections
import json
from pathlib import Path
import re
import shutil
import subprocess

from attribute_cpu_allocations_v2 import IMAGE, LIBRARIES, sha
from run_fc2_rank_waves_native import ROOT, guarded


DRIVER = "usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.580.159.04"
DRIVER_SHA = "3c9340e2af8c764d8bd024cc895d41b5bb244103d2879501081f7677bfe3d557"
DRIVER_BUILD_ID = "701a056de23ed91e89c48916a3f5847dd8da607e"
NAMES = ("cpu-decode-sample-v2", "cpu-decode-32k-retained-v2")


def samples(text, interval):
    result = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if not lines or "cpu-clock:u:" not in lines[0]:
            continue
        match = re.search(r"\s(\d+\.\d+):", lines[0])
        if match is None or not interval[0] <= float(match[1]) <= interval[1]:
            raise RuntimeError("CPU event outside the strict marker interval")
        frames = []
        for line in lines[1:]:
            match = re.fullmatch(r"\s*([0-9a-f]+) (.+) \((.*)\)", line)
            if not match:
                raise RuntimeError(f"Unparsed callchain line: {line}")
            frames.append({
                "address": match[1],
                "symbol": re.sub(r"\+0x[0-9a-f]+$", "", match[2]),
                "dso": match[3],
            })
        result.append({"header": lines[0], "frames": frames})
    return result


def summarize(records):
    leaves = collections.Counter()
    driver_callers = collections.Counter()
    empty = 0
    unknown_libc = 0
    for record in records:
        frames = record["frames"]
        if not frames:
            empty += 1
            continue
        leaf = frames[0]
        leaves[(leaf["symbol"], leaf["dso"])] += 1
        unknown_libc += int(
            leaf["symbol"] == "[unknown]" and leaf["dso"].endswith("/libc.so.6"))
        if any(frame["dso"].endswith("/" + Path(DRIVER).name) for frame in frames):
            caller = next(
                (frame["symbol"] for frame in frames
                 if frame["symbol"].startswith("dawn::")), None)
            driver_callers[caller] += 1
    return {
        "strict_samples": len(records),
        "header_only_samples": empty,
        "unknown_libc_leaf_samples": unknown_libc,
        "leaf_counts": [
            {"symbol": symbol, "dso": dso, "samples": count}
            for (symbol, dso), count in leaves.most_common()
        ],
        "driver_stack_first_dawn_ancestor": [
            {"symbol": symbol, "samples": count}
            for symbol, count in driver_callers.most_common()
        ],
    }


def main():
    old_root = ROOT / "cpu-decode-symbol-root-v2"
    new_root = ROOT / "cpu-decode-symbol-root-v3"
    output = ROOT / "cpu-driver-offline-unwind-v3.json"
    destinations = [new_root, output]
    destinations += [
        ROOT / f"{name}-callchains-symbols-v3.txt" for name in NAMES
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Fresh symbol-root/decode output required")
    for relative, expected in LIBRARIES.items():
        if sha(old_root / relative) != expected:
            raise RuntimeError(f"Changed approved-image symbol library: {relative}")
    inputs = [Path(__file__), ROOT / "attribute_cpu_allocations_v2.py",
              ROOT / "run_fc2_rank_waves_native.py",
              ROOT / "cpu-driver-symbol-probe-v2.log",
              ROOT / "cpu-driver-symbol-probe-v2.guard.json"]
    captures = {}
    for name in NAMES:
        outcome_path = ROOT / f"{name}-outcome.json"
        outcome = json.loads(outcome_path.read_text())
        data = ROOT / f"{name}.perf.data"
        if sha(data) != outcome["source_and_evidence_sha256"][str(data)]:
            raise RuntimeError("Changed original raw capture")
        report = ROOT / f"{name}.perf-report.txt"
        if "# Total Lost Samples: 0" not in report.read_text():
            raise RuntimeError("Only original zero-loss captures are admitted")
        if outcome["samples"] < 100:
            raise RuntimeError("Insufficient original strict samples")
        captures[name] = outcome
        inputs.extend((outcome_path, data, report,
                       ROOT / f"{name}-callchains-symbols-v2.txt"))
    old_files = {
        str(path.relative_to(old_root)): sha(path)
        for path in old_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    old_symlinks = {
        str(path.relative_to(old_root)): str(path.readlink())
        for path in old_root.rglob("*") if path.is_symlink()
    }
    shutil.copytree(old_root, new_root, symlinks=True)
    if (new_root / DRIVER).exists():
        raise FileExistsError(new_root / DRIVER)
    child_code = f"""
from pathlib import Path
import hashlib
import re
import shutil
import subprocess
p = Path('/{DRIVER}')
q = Path('/evidence/webgpu-perf-oct20/cpu-decode-symbol-root-v3/{DRIVER}')
assert q.resolve().is_relative_to(
    Path('/evidence/webgpu-perf-oct20/cpu-decode-symbol-root-v3').resolve())
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
assert digest(p) == {DRIVER_SHA!r}, 'Mapped-driver source differs from probe'
notes = subprocess.run(['readelf', '-n', str(p)], check=True,
                       capture_output=True, text=True).stdout
assert re.search(r'Build ID: ([0-9a-f]+)', notes)[1] == {DRIVER_BUILD_ID!r}
q.parent.mkdir(parents=True, exist_ok=True)
with p.open('rb') as source, q.open('xb') as target:
    shutil.copyfileobj(source, target, 1024 * 1024)
assert digest(q) == {DRIVER_SHA!r}, 'Copied driver bytes differ'
print('PASS CPU-only exact probed driver copy', q.stat().st_size)
"""
    prefix = ROOT / "cpu-driver-symbol-copy-v3"
    child, guard = guarded(
        prefix, ["taskset", "-c", "0-7", "python3", "-c", child_code])
    child.check_returncode()
    if guard["status"] != "completed" or sha(new_root / DRIVER) != DRIVER_SHA:
        raise RuntimeError("Driver-symbol copy did not complete")
    for relative, expected in old_files.items():
        if sha(old_root / relative) != expected or sha(new_root / relative) != expected:
            raise RuntimeError(f"Changed inherited symbol file: {relative}")
    for relative, target in old_symlinks.items():
        if (str((old_root / relative).readlink()) != target
                or str((new_root / relative).readlink()) != target):
            raise RuntimeError(f"Changed inherited symbol symlink: {relative}")
    inputs.extend((prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"),
                   new_root / DRIVER))
    reports = {}
    for name, outcome in captures.items():
        interval = tuple(map(float, outcome["interval"].split(",")))
        old = samples((ROOT / f"{name}-callchains-symbols-v2.txt").read_text(), interval)
        command = ["perf", "script", "-i", str(ROOT / f"{name}.perf.data"),
                   "--symfs", str(new_root), "--time", outcome["interval"]]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stderr.strip():
            raise RuntimeError(result.stderr)
        new = samples(result.stdout, interval)
        if (len(new) != outcome["samples"]
                or [row["header"] for row in old] != [row["header"] for row in new]):
            raise RuntimeError("Strict CPU sample identities changed")
        if any(a["frames"] and not b["frames"] for a, b in zip(old, new)):
            raise RuntimeError("Previously decoded stack became empty")
        chains = ROOT / f"{name}-callchains-symbols-v3.txt"
        with chains.open("x") as stream:
            stream.write(result.stdout)
        inputs.append(chains)
        reports[name] = {
            "before": summarize(old),
            "after": summarize(new),
            "recovered_header_only_samples": sum(
                not a["frames"] and bool(b["frames"]) for a, b in zip(old, new)),
            "perf_command": command,
        }
        data = ROOT / f"{name}.perf.data"
        if sha(data) != outcome["source_and_evidence_sha256"][str(data)]:
            raise RuntimeError("Raw capture changed during offline decode")
    report = {
        "status": "original_zero_loss_captures_offline_driver_decode_completed",
        "diagnostic_only": True,
        "performance_qualified": False,
        "quality_qualified": False,
        "approved_image_id": IMAGE,
        "driver_sha256": DRIVER_SHA,
        "driver_build_id_observed_in_current_guarded_environment": DRIVER_BUILD_ID,
        "inherited_symbol_file_sha256": old_files,
        "inherited_symbol_symlinks": old_symlinks,
        "captures": reports,
        "limitations": [
            "No new model load, GPU computation, or measured speedup.",
            "Driver is runtime-injected, not independently supplied by the pinned image.",
            "Raw captures do not independently establish each mapped DSO build ID.",
            "Current injected driver path/version/hash/build ID are recorded, not historical proof.",
            "Original 8-KiB user-stack snapshots can truncate deep callers.",
            "Sampled user CPU fractions are not wall-time shares or predicted TPS gains.",
            "Unknown libc symbols are not assigned function identities.",
            "The separate lossy 32-KiB capture remains excluded.",
        ],
        "source_and_evidence_sha256": {str(path): sha(path) for path in inputs},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        name: {key: value for key, value in row.items() if key != "perf_command"}
        for name, row in reports.items()
    }, indent=2))


if __name__ == "__main__":
    main()
