import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-silu", action="store_true")
    parser.add_argument("--gdn-prep", action="store_true")
    parser.add_argument("--shared-post", action="store_true")
    parser.add_argument("--output-name")
    args = parser.parse_args()
    repo = Path("/workspace/onnxruntime-webgpu-h200-perf")
    if sum((args.split_silu, args.gdn_prep, args.shared_post)) > 1:
        raise ValueError("Choose one fusion stage")
    prior_directory = ("build-int4-gdn-prep" if args.shared_post else
                       "build-int4-split-silu-v3" if args.gdn_prep else
                       "build-int4-hc-fusion" if args.split_silu else "build-int4-dynamic-uniforms")
    prior = json.loads((repo / prior_directory / "build-manifest.json").read_text())
    name = args.output_name or ("build-int4-shared-post" if args.shared_post else
                               "build-int4-gdn-prep" if args.gdn_prep else
                               "build-int4-split-silu" if args.split_silu else "build-int4-hc-fusion")
    if Path(name).name != name or not name.startswith("build-int4-"):
        raise ValueError("Output must be a named build-int4- directory within the repository")
    output = repo / name
    output.mkdir(exist_ok=True)
    library = output / "libonnxruntime_providers_webgpu.so"
    if library.exists():
        raise FileExistsError(library)
    baseline_dir = output / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline = baseline_dir / library.name
    link = prior["link_command"].copy()
    link[link.index("-o") + 1] = str(baseline)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    if digest(baseline) != prior["provider_sha256"]:
        raise RuntimeError("Baseline objects do not reproduce the retained provider in this toolchain")
    baseline.unlink()
    baseline_dir.rmdir()
    commands = []
    sources = {}
    source_names = (
        ("bert/hyper_connection_post_mix.cc", "quantization/matmul_nbits.cc",
         "webgpu_contrib_kernels.cc") if args.shared_post else
        ("bert/causal_conv_with_state.cc", "webgpu_contrib_kernels.cc") if args.gdn_prep else
        ("bert/scaled_silu.cc", "webgpu_contrib_kernels.cc") if args.split_silu else
        ("bert/hyper_connection_pre_mix.cc", "bert/hyper_connection_post_mix.cc", "webgpu_contrib_kernels.cc")
    )
    for name in source_names:
        source = repo / "onnxruntime/contrib_ops/webgpu" / name
        suffix = "contrib_ops__webgpu__" + name.replace("/", "__") + ".o"
        old = [p for p in link if p.endswith(suffix) or p.endswith("/" + source.name + ".o")]
        if len(old) != 1:
            raise ValueError(f"Expected one object ending in {suffix}: {old}")
        obj = output / (source.name + ".o")
        command = prior["compile_commands"][0].copy()
        command[command.index("-c") + 1] = str(source)
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=prior["working_directory"], check=True)
        link = [str(obj) if p == old[0] else p for p in link]
        commands.append(command)
        sources[str(source)] = digest(source)
        shutil.copyfile(source, output / source.name)
        if source.with_suffix(".h").exists():
            header = source.with_suffix(".h")
            sources[str(header)] = digest(header)
            shutil.copyfile(header, output / header.name)
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], check=True)
    report = {
        "status": "built_not_GPU_qualified",
        "baseline_byte_identical": True,
        "baseline_provider_sha256": prior["provider_sha256"],
        "provider_sha256": digest(library),
        "source_sha256": sources,
        "compile_commands": commands,
        "link_command": link,
        "working_directory": prior["working_directory"],
        "scope": ("Opt-in shared expert Sigmoid/GatedAdd/HC post-mix fusion plus exact HC/mid dense selectors"
                  if args.shared_post else "Opt-in model-local FP16 GDN projection preparation on the HC+SplitSiLU stack"
                  if args.gdn_prep else "Opt-in model-local FP16 Split/ScaledSiLU fusion on the HC-fused stack"
                  if args.split_silu else "Opt-in model-local FP16 HC pre-sigmoid and post-Div/Sigmoid/Mul fusion"),
        "quality_qualified": False,
    }
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
