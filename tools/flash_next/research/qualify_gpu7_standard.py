import argparse
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, identities, load, sha, snapshot
from resume_gpu7_8k import validate_gpu7


GPU_UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192), required=True)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if args.filtered and not args.wide:
        parser.error("--filtered requires --wide")
    label = "exact-wide-filtered" if args.filtered else "exact-wide" if args.wide else "quickgelu"
    prefix = ROOT / f"{label}-gpu7-warm-{args.context}-20261028"
    if args.output_prefix:
        prefix = args.output_prefix
    output = prefix.with_suffix(".qualification.json")
    if output.exists() or any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".log")):
        raise FileExistsError(prefix)
    control = load(ROOT / "int4-pending64-standard-contexts.json")
    baseline_identity = identities(control)
    identity = baseline_identity.copy()
    invocation = load(ROOT / "quickgelu-mul-candidate-32k-20261027.guard.json")["command"].copy()
    invocation[-1] = invocation[-1].replace(
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd", GPU_UUID
    ).replace("quickgelu-mul-candidate-32k-20261027", prefix.name).replace(
        "--contexts 32768", f"--contexts {args.context}"
    )
    if args.wide:
        native = load(ROOT / ("exact-wide-filtered-gpu7-native-20261028.qualification.json"
                              if args.filtered else "exact-wide-gpu7-native-20261028.qualification.json"))
        assert native["status"] == "native_finite_bitwise_qualified_not_model_qualified"
        build = Path("/home/sayanshaw/repos/qwen38-int4-exact-wide-gpu7" +
                     ("-filtered" if args.filtered else "") + "/build")
        manifest = load(build / "build-manifest.json")
        assert sha(build / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"] == manifest["provider_sha256"]
        for path, digest in manifest["source_sha256"].items():
            assert sha(Path(path.replace("/workspace/", "/home/sayanshaw/repos/"))) == digest
        identity["provider_sha256"] = native["provider_sha256"]
        invocation[-1] = invocation[-1].replace(
            "/workspace/qwen38-int4-quickgelu-next/build/libonnxruntime_providers_webgpu.so",
            "/workspace/qwen38-int4-exact-wide-gpu7/build/libonnxruntime_providers_webgpu.so",
        ).replace("exec python3", "export ORT_WEBGPU_EXACT_DENSE_WIDE=1; exec python3")
        if args.filtered:
            invocation[-1] = invocation[-1].replace(
                "/workspace/qwen38-int4-exact-wide-gpu7/build/",
                "/workspace/qwen38-int4-exact-wide-gpu7-filtered/build/")
    report = dict(status="running", initial_resources=snapshot(), identity=identity,
                  wide=args.wide, quality_qualified=False, headline_eligible=False)
    output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        subprocess.run(
            ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
             "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1100",
             "--output-prefix", str(prefix), "--", *invocation], check=True,
        )
        report["row"] = validate_gpu7(prefix.with_suffix(".json"), args.context, control, identity)
        assert identities(control) == baseline_identity
        report["status"] = "original_standard_sequences_qualified_gpu7"
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=repr(error))
        raise
    finally:
        report["final_resources"] = snapshot()
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["row"], indent=2))


if __name__ == "__main__":
    main()
