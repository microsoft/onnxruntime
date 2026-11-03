import collections
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from attribute_cpu_allocations_v2 import IMAGE, sha
from repair_cpu_driver_unwind_v3 import ROOT, NAMES, samples, summarize


LIBC = "usr/lib/x86_64-linux-gnu/libc.so.6"


def ifunc_ranges(entries, frames_text):
    fdes = [
        (int(a, 16), int(b, 16))
        for a, b in re.findall(r"\bFDE\b[^\n]*\bpc=([0-9a-f]+)\.\.([0-9a-f]+)",
                              frames_text)
    ]
    aliases = collections.defaultdict(list)
    unmatched = []
    for row in entries:
        if not row["inside_verified_libc"]:
            continue
        matching = [pair for pair in fdes if pair[0] == row["elf_address"]]
        if len(matching) != 1:
            unmatched.append(row)
            continue
        aliases[matching[0]].append(row["name"])
    return [
        {"begin": begin, "end": end, "aliases": sorted(names)}
        for (begin, end), names in sorted(aliases.items())
    ], unmatched


def libc_base(mapping_lines):
    bases = set()
    executable = []
    for line in mapping_lines:
        if not line.endswith("r-xp /usr/lib/x86_64-linux-gnu/libc.so.6"):
            continue
        match = re.search(
            r"\[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0) ", line)
        if match is None:
            raise RuntimeError("Unparsed executable libc mapping")
        begin, length, offset = (int(value, 16) for value in match.groups())
        bases.add(begin - offset)
        executable.append((begin, begin + length))
    if len(bases) != 1:
        raise RuntimeError("Missing or ambiguous executable libc load bias")
    return bases.pop(), executable


def main():
    old_root = ROOT / "cpu-decode-symbol-root-v3"
    new_root = ROOT / "cpu-decode-symbol-root-v5"
    output = ROOT / "cpu-libc-ifunc-attribution-v2.json"
    for path in (new_root, output, ROOT / "cpu-libc-ifunc-v2-spec.json",
                 ROOT / "cpu-libc-ifunc-v2-probe.json",
                 ROOT / "cpu-libc-ifunc-v2-cpu.log"):
        if path.exists():
            raise FileExistsError(path)
    driver_report_path = ROOT / "cpu-driver-offline-unwind-v3.json"
    driver_report = json.loads(driver_report_path.read_text())
    for path, digest in driver_report["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed parent diagnostic evidence: {path}")
    libc = old_root / LIBC
    expected_libc = driver_report["inherited_symbol_file_sha256"][LIBC]
    if sha(libc) != expected_libc:
        raise RuntimeError("Changed exact-image libc")
    dynsyms = subprocess.run(["readelf", "--dyn-syms", "--wide", str(libc)],
                             check=True, capture_output=True, text=True).stdout
    names = sorted({
        fields[7].split("@")[0]
        for line in dynsyms.splitlines()
        if len(fields := line.split()) >= 8 and fields[3] == "IFUNC"
    })
    if not names:
        raise RuntimeError("No exact-image exported IFUNCs")
    loads = subprocess.run(["readelf", "-lW", str(libc)], check=True,
                           capture_output=True, text=True).stdout
    executable_loads = [
        line.split() for line in loads.splitlines()
        if line.strip().startswith("LOAD") and "E" in line.split()[6:-1]
    ]
    if (len(executable_loads) != 1
            or int(executable_loads[0][1], 16) != int(executable_loads[0][2], 16)):
        raise RuntimeError("ELF executable file/vaddr relation is not identity")
    shutil.copytree(old_root, new_root, symlinks=True, copy_function=os.link)
    spec = ROOT / "cpu-libc-ifunc-v2-spec.json"
    with spec.open("x") as stream:
        json.dump({"ifunc_names": names, "libc_sha256": expected_libc}, stream)
    helper = ROOT / "probe_pinned_libc_ifunc_v2.py"
    command = [
        "sudo", "-n", "docker", "-H",
        "unix:///home/sayanshaw/docker-qwen38.sock", "run", "--rm",
        "--runtime=runc", "--network=none", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--memory=1g", "--memory-swap=1g",
        "--cpuset-cpus=8-15", "--pids-limit=64", "--user=664503:664503",
        "-e", "NVIDIA_VISIBLE_DEVICES=void",
        "-v", f"{ROOT.parent}:/evidence", IMAGE, "python3",
        "/evidence/webgpu-perf-oct20/probe_pinned_libc_ifunc_v2.py",
    ]
    log = ROOT / "cpu-libc-ifunc-v2-cpu.log"
    with log.open("x") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    probe_path = ROOT / "cpu-libc-ifunc-v2-probe.json"
    probe = json.loads(probe_path.read_text())
    if (probe["libc_sha256"] != expected_libc
            or sha(new_root / "opt/conda/bin/python3.11") != probe["python_sha256"]):
        raise RuntimeError("Wrong pinned-image library/executable provenance")
    fde = subprocess.run(["readelf", "--debug-dump=frames", str(libc)], check=True,
                         capture_output=True, text=True).stdout
    ranges, unmatched = ifunc_ranges(probe["selected_ifuncs"], fde)
    mapping_path = ROOT / "cpu-libc-driver-mmaps-v1.txt"
    mappings = mapping_path.read_text()
    evidence = [Path(__file__), helper, spec, log, probe_path, mapping_path,
                driver_report_path, libc, new_root / "opt/conda/bin/python3.11",
                ROOT / "cpu-libc-ifunc-v2-generation.json"]
    reports = {}
    for name in NAMES:
        outcome_path = ROOT / f"{name}-outcome.json"
        outcome = json.loads(outcome_path.read_text())
        data = ROOT / f"{name}.perf.data"
        if sha(data) != outcome["source_and_evidence_sha256"][str(data)]:
            raise RuntimeError("Changed original raw capture")
        interval = tuple(map(float, outcome["interval"].split(",")))
        old_path = ROOT / f"{name}-callchains-symbols-v3.txt"
        old = samples(old_path.read_text(), interval)
        result = subprocess.run([
            "perf", "script", "-i", str(data), "--symfs", str(new_root),
            "--time", outcome["interval"],
        ], check=True, capture_output=True, text=True)
        if result.stderr.strip():
            raise RuntimeError(result.stderr)
        rows = samples(result.stdout, interval)
        if (len(rows) != outcome["samples"]
                or [r["header"] for r in rows] != [r["header"] for r in old]):
            raise RuntimeError("Strict CPU identities changed")
        path = ROOT / f"{name}-callchains-symbols-v5.txt"
        with path.open("x") as stream:
            stream.write(result.stdout)
        section = mappings.split(name + "\n", 1)[1].split("cpu-decode-", 1)[0]
        base, executable = libc_base(section.splitlines())
        counts = collections.Counter()
        unresolved = collections.Counter()
        environment = collections.Counter()
        allocator_ancestry = 0
        leaf_records = []
        for row in rows:
            frames = row["frames"]
            symbols = [frame["symbol"] for frame in frames]
            if "getenv" in symbols:
                index = symbols.index("getenv")
                caller = next((
                    symbol for symbol in symbols[index + 1:]
                    if not any(skip in symbol for skip in (
                        "ParseEnvironmentVariable", "PosixEnv::GetEnvironmentVar",
                        "GetEnvironmentVarOrDefault"))), None)
                environment[caller] += 1
            if (not frames or frames[0]["symbol"] != "[unknown]"
                    or not frames[0]["dso"].endswith("/libc.so.6")):
                continue
            address = int(frames[0]["address"], 16)
            if not any(begin <= address < end for begin, end in executable):
                raise RuntimeError("Unknown libc leaf is outside its executable mapping")
            offset = address - base
            matched = [r for r in ranges if r["begin"] <= offset < r["end"]]
            aliases = matched[0]["aliases"] if len(matched) == 1 else []
            if aliases:
                counts[tuple(aliases)] += 1
            else:
                unresolved[hex(offset)] += 1
                allocator_ancestry += int(any(
                    symbol in ("malloc", "cfree", "free", "realloc", "calloc")
                    for symbol in symbols[1:]))
            leaf_records.append({
                "header": row["header"], "elf_address": hex(offset),
                "selected_ifunc_implementation_aliases": aliases,
                "fde": matched[0] if len(matched) == 1 else None,
                "callers": symbols[1:6],
            })
        reports[name] = {
            "before": summarize(old),
            "after": summarize(rows),
            "matched_ifunc_leaf_samples": sum(counts.values()),
            "implementation_alias_counts": [
                {"aliases": list(aliases), "samples": count}
                for aliases, count in counts.most_common()],
            "unassigned_unknown_libc_leaf_samples": sum(unresolved.values()),
            "unassigned_with_observed_allocator_ancestor": allocator_ancestry,
            "unassigned_offsets": dict(unresolved),
            "getenv_samples": sum(environment.values()),
            "first_getenv_callers": [
                {"symbol": symbol, "samples": count}
                for symbol, count in environment.most_common()],
            "unknown_libc_leaf_records": leaf_records,
        }
        evidence.extend((outcome_path, data, old_path, path))
        if sha(data) != outcome["source_and_evidence_sha256"][str(data)]:
            raise RuntimeError("Raw capture changed during decode")
    report = {
        "status": "strict_cpu_libc_selected_ifunc_and_caller_diagnostic",
        "diagnostic_only": True, "performance_qualified": False,
        "quality_qualified": False, "approved_image_id": IMAGE,
        "cpu_helper_command": command, "matched_ifunc_fdes": ranges,
        "selected_libc_ifuncs_without_unique_entry_fde": unmatched,
        "captures": reports,
        "limitations": [
            "Implementation aliases do not identify which public API was called.",
            "Only FDE ranges beginning exactly at a verified selected IFUNC entry are assigned.",
            "Current CPU IFUNC selection is observed, not independent historical dispatch proof.",
            "Unmatched internal libc addresses remain unnamed; allocator ancestry is not a function identity.",
            "Mapped-DSO build IDs are not independently established by the original captures.",
            "Original 8-KiB snapshots can truncate deep callers.",
            "Sampled user CPU fractions are not wall time or predicted gains.",
            "No new model load, GPU computation, installs or measured speedup.",
        ],
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for name, row in reports.items():
        print(name, "samples", row["after"]["strict_samples"], "empty",
              row["after"]["header_only_samples"], "IFUNC-matched",
              row["matched_ifunc_leaf_samples"], "unassigned",
              row["unassigned_unknown_libc_leaf_samples"], "getenv",
              row["getenv_samples"])
        print(row["implementation_alias_counts"])


if __name__ == "__main__":
    main()
