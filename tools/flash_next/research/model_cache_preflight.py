import json
import os
from pathlib import Path
import subprocess
import sys


DATA_FILES = ("engram.data", "model.onnx.data")


def snapshot_model_cache(model):
    paths = [model / name for name in DATA_FILES]
    sizes = {str(path): path.stat().st_size for path in paths}
    # Unprivileged mincore can report every page resident for root-owned read-only files.
    output = subprocess.check_output(
        ["sudo", "-n", "fincore", "--json", "--bytes", "--output", "RES,SIZE,FILE",
         *map(str, paths)], text=True)
    rows = json.loads(output)["fincore"]
    if len(rows) != len(paths) or {row["file"] for row in rows} != set(sizes):
        raise ValueError("Cache probe did not return exactly the requested model data files")
    page_size = os.sysconf("SC_PAGE_SIZE")
    for row in rows:
        if (type(row["res"]) is not int or type(row["size"]) is not int or
                row["size"] != sizes[row["file"]] or row["size"] <= 0 or
                not 0 <= row["res"] <= ((row["size"] + page_size - 1) // page_size) * page_size):
            raise ValueError(f"Invalid model cache-residency record: {row}")
    return rows


def all_resident(rows):
    return all(row["res"] >= row["size"] for row in rows)


def prepare_model_cache(model, prefix):
    evidence = prefix.with_name(prefix.name + "-cache.json")
    report = dict(status="checking", evidence_path=str(evidence),
                  privileged_residency_probe=True, warming_excluded_from_timing=True,
                  file_cache_is_not_pinned=True, warmed=False,
                  before=snapshot_model_cache(model))
    with evidence.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if not all_resident(report["before"]):
        report["status"] = "warming"
        report["warming_log"] = str(prefix.with_name(prefix.name + "-cache-warm.log"))
        evidence.write_text(json.dumps(report, indent=2) + "\n")
        with Path(report["warming_log"]).open("x") as log:
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("warm_int4_cpu_data.py")),
                 "--model", str(model)], stdout=log, stderr=subprocess.STDOUT,
                timeout=600, check=True)
        report["warmed"] = True
    report["after_preflight"] = snapshot_model_cache(model)
    ready = all_resident(report["after_preflight"])
    report["status"] = "ready" if ready else "failed_residency"
    evidence.write_text(json.dumps(report, indent=2) + "\n")
    if not ready:
        raise RuntimeError(f"Model data not fully resident after cache preflight: {evidence}")
    return report
