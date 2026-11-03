import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill2048_rows4_model import STAMP, VARIANT


def main():
    output = ROOT / f"{VARIANT}-context2048-screen-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    rows, sources = [], {}
    for full in (True, False):
        cases = {}
        for control in (True, False):
            case = "control1024" if control else "candidate2048"
            prefix = ROOT / f"{VARIANT}-{case}-gpu7-{'full-' if full else ''}standard2048-trial0-{STAMP}"
            path = prefix.with_suffix(".qualification.json")
            proof = json.loads(path.read_text())
            assert proof["status"] == "original_sequences_qualified" and proof["zero_major_faults"]
            assert proof["original_output_prefixes_match"]
            assert proof["context"] == 2048 and proof["full_validation"] == full
            assert proof["control"] == control and proof["chunk"] == (1024 if control else 2048)
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest, name
                sources[name] = digest
            sources[path.name] = sha(path)
            cases["control" if control else "candidate"] = proof
        first, selected = cases["control"]["result"], cases["candidate"]["result"]
        rows.append(dict(full_validation=full, control=first, candidate=selected,
                         observed_ttft_ratio=selected["mean_ttft_seconds"] / first["mean_ttft_seconds"],
                         observed_decode_ratio=selected["median_decode_tps"] / first["median_decode_tps"]))
    report = dict(status="initial_separate_load_ttft_screen_not_promotion_qualified",
                  rejected=any(row["observed_ttft_ratio"] >= 1 or row["observed_decode_ratio"] < 0.99
                               for row in rows),
                  context=2048, provider_sha256=cases["candidate"]["provider_sha256"], rows=rows,
                  original_output_prefixes_match=True, zero_major_faults=True, default_off=True,
                  comparison_kind="separate_load_chunk2048_vs_qualified_chunk1024_not_isolated_route_gain",
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={**sources, Path(__file__).name: sha(Path(__file__))})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "rejected", "rows")}), flush=True)


if __name__ == "__main__":
    main()
