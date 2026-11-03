import argparse
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=int, nargs="+", default=[64, 128, 256, 96, 32, 64])
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--kind", choices=("gdn-view", "prefill-dense"), default="gdn-view")
    args = parser.parse_args()
    summary = args.output_prefix.with_suffix(".sweep.json")
    if summary.exists():
        raise FileExistsError(summary)
    rows = []
    for index, pending in enumerate(args.pending):
        prefix = args.output_prefix.with_name(args.output_prefix.name + f"-step{index}-pending{pending}")
        log = prefix.with_suffix(".driver.log")
        if args.kind == "prefill-dense":
            command = ["python3", str(ROOT / "run_prefill_dense_model.py"), "--chunk", "128",
                       "--phase", "standard", "--pending", str(pending), "--output-prefix", str(prefix)]
        else:
            command = ["python3", str(ROOT / "run_int4_candidate.py"), "--kind", "gdn-view",
                       "--phase", "standard", "--pending", str(pending), "--output-prefix", str(prefix)]
        with log.open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads(prefix.with_suffix(".qualification.json").read_text())
        metrics = report["result"] if args.kind == "prefill-dense" else report["summary"]
        rows.append(dict(step=index, pending=pending, median_decode_tps=metrics["median_decode_tps"],
                         aggregate_decode_tps=metrics["decode_tps"], evidence=str(prefix)))
        summary.write_text(json.dumps(dict(
            status="completed" if len(rows) == len(args.pending) else "in_progress", rows=rows,
            scope="Separate-load tuning rows with repeated endpoint control; not matched causal evidence",
            quality_qualified=False, headline_eligible=False), indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
