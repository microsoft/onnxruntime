"""Assemble two model-package samples from the staged assets:

  portable_package/    - self-contained, portable layout. Base (onnx+data) and
                         compiled (ctx+bin) live in content-addressed shared assets.
  nonportable_package/ - installed layout. Model files live OUTSIDE the package in
                         ../external_assets. ort_info.json files are shipped as
                         *.template.json and filled with absolute paths by resolve.py.

Run tools/generate_assets.py first to produce <sample>/_staging/{base,compiled}.
"""
import hashlib
import json
import os
import shutil

import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.dirname(HERE)                       # samples/model_package
STG = os.path.join(SAMPLE, "_staging")
BASE = os.path.join(STG, "base")
COMPILED = os.path.join(STG, "compiled")

DATA = "weights.data"
BIN = "prefill.ctx_OpenVINOExecutionProvider.bin"    # the single shared bin
COMPONENTS = ["prefill", "iter"]
COMPAT_KEY = "ep_compatibility_info.OpenVINOExecutionProvider"
ASSETS_TOKEN = "__ASSETS_DIR__"


def sha256_hex(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def asset_uri(files):
    """files: {rel_name: abs_src}. Returns (uri, digest) per asset_hasher.cc."""
    text = "".join(sha256_hex(src) + "  " + rel + "\n" for rel, src in sorted(files.items()))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "sha256:" + digest, digest


def read_compat(ctx_path):
    m = onnx.load(ctx_path, load_external_data=False)
    return next((p.value for p in m.metadata_props if p.key == COMPAT_KEY), None)


def variant(vdir, model_ep, device, compat=None):
    v = {"variant_directory": vdir, "ep": model_ep}
    if device:
        v["device"] = device
    if compat:
        v["compatibility_string"] = compat
    v["executor_info"] = {"ort": "ort_info.json"}
    return v


def component(name, compat):
    return {
        "component_name": name,
        "variants": {
            "cpu": variant(f"{name}/cpu", "CPUExecutionProvider", None),
            "ov": variant(f"{name}/ov", "OpenVINOExecutionProvider", "npu", compat),
        },
    }


# ---------------------------------------------------------------- portable
def build_portable():
    pkg = os.path.join(SAMPLE, "portable_package")
    for c in COMPONENTS:
        shutil.rmtree(os.path.join(pkg, c), ignore_errors=True)
    shutil.rmtree(os.path.join(pkg, "shared_assets"), ignore_errors=True)
    os.makedirs(pkg, exist_ok=True)

    base_files = {DATA: os.path.join(BASE, DATA)}
    for c in COMPONENTS:
        base_files[f"{c}.onnx"] = os.path.join(BASE, f"{c}.onnx")
    comp_files = {BIN: os.path.join(COMPILED, BIN)}
    for c in COMPONENTS:
        comp_files[f"{c}.ctx.onnx"] = os.path.join(COMPILED, f"{c}.ctx.onnx")

    data_uri, data_dig = asset_uri(base_files)
    bin_uri, bin_dig = asset_uri(comp_files)
    sa = os.path.join(pkg, "shared_assets")
    for dig, files in ((data_dig, base_files), (bin_dig, comp_files)):
        dst = os.path.join(sa, f"sha256-{dig}")
        os.makedirs(dst, exist_ok=True)
        for rel, src in files.items():
            shutil.copy2(src, os.path.join(dst, rel))

    compat = read_compat(os.path.join(COMPILED, "prefill.ctx.onnx"))
    for c in COMPONENTS:
        cpu_dir = os.path.join(pkg, c, "cpu"); os.makedirs(cpu_dir, exist_ok=True)
        json.dump({
            "model_file": f"{data_uri}/{c}.onnx",
            "session_options": {"session.model_external_initializers_file_folder_path": data_uri},
        }, open(os.path.join(cpu_dir, "ort_info.json"), "w"), indent=2)

        ov_dir = os.path.join(pkg, c, "ov"); os.makedirs(ov_dir, exist_ok=True)
        json.dump({
            "model_file": f"{bin_uri}/{c}.ctx.onnx",
            "session_options": {
                "ep.share_ep_contexts": "1",
                "session.model_external_initializers_file_folder_path": data_uri,
                "ep.context_file_path": f"{bin_uri}/{c}.ctx.onnx",
            },
        }, open(os.path.join(ov_dir, "ort_info.json"), "w"), indent=2)

    manifest = {
        "schema_version": "1.0",
        "package_name": "tiny-mlp-portable",
        "package_version": "1.0.0",
        "description": "Tiny MLP, two shape specializations (prefill/iter), each with cpu and "
                       "ov-npu variants. Portable: base and compiled assets are content-addressed "
                       "shared assets inside the package.",
        "layout": "portable",
        "components": {c: component(c, compat) for c in COMPONENTS},
    }
    json.dump(manifest, open(os.path.join(pkg, "manifest.json"), "w"), indent=2)
    print(f"portable: data asset sha256-{data_dig}, bin asset sha256-{bin_dig}")


# ------------------------------------------------------------- non-portable
def build_nonportable():
    pkg = os.path.join(SAMPLE, "nonportable_package")
    ext = os.path.join(SAMPLE, "external_assets")
    for c in COMPONENTS:
        shutil.rmtree(os.path.join(pkg, c), ignore_errors=True)
    shutil.rmtree(ext, ignore_errors=True)
    os.makedirs(pkg, exist_ok=True)

    # external assets OUTSIDE the package: base (onnx+data) and compiled (ctx+bin)
    os.makedirs(os.path.join(ext, "base"))
    os.makedirs(os.path.join(ext, "compiled"))
    shutil.copy2(os.path.join(BASE, DATA), os.path.join(ext, "base", DATA))
    for c in COMPONENTS:
        shutil.copy2(os.path.join(BASE, f"{c}.onnx"), os.path.join(ext, "base", f"{c}.onnx"))
        shutil.copy2(os.path.join(COMPILED, f"{c}.ctx.onnx"), os.path.join(ext, "compiled", f"{c}.ctx.onnx"))
    shutil.copy2(os.path.join(COMPILED, BIN), os.path.join(ext, "compiled", BIN))

    # templated ort_info: absolute paths via __ASSETS_DIR__ token, filled by resolve.py
    for c in COMPONENTS:
        cpu_dir = os.path.join(pkg, c, "cpu"); os.makedirs(cpu_dir, exist_ok=True)
        json.dump({
            "model_file": f"{ASSETS_TOKEN}/base/{c}.onnx",
            "session_options": {"session.model_external_initializers_file_folder_path": f"{ASSETS_TOKEN}/base"},
        }, open(os.path.join(cpu_dir, "ort_info.template.json"), "w"), indent=2)

        ov_dir = os.path.join(pkg, c, "ov"); os.makedirs(ov_dir, exist_ok=True)
        json.dump({
            "model_file": f"{ASSETS_TOKEN}/compiled/{c}.ctx.onnx",
            "session_options": {
                "ep.share_ep_contexts": "1",
                "session.model_external_initializers_file_folder_path": f"{ASSETS_TOKEN}/base",
                "ep.context_file_path": f"{ASSETS_TOKEN}/compiled/{c}.ctx.onnx",
            },
        }, open(os.path.join(ov_dir, "ort_info.template.json"), "w"), indent=2)

    compat = read_compat(os.path.join(COMPILED, "prefill.ctx.onnx"))
    manifest = {
        "schema_version": "1.0",
        "package_name": "tiny-mlp-nonportable",
        "package_version": "1.0.0",
        "description": "Tiny MLP, two shape specializations. Non-portable (installed) layout: model "
                       "files live outside the package in ../external_assets and are referenced by "
                       "absolute paths written by resolve.py.",
        "layout": "installed",
        "components": {c: component(c, compat) for c in COMPONENTS},
    }
    json.dump(manifest, open(os.path.join(pkg, "manifest.json"), "w"), indent=2)
    print(f"nonportable: external assets at {os.path.relpath(ext, SAMPLE)}")


def main():
    build_portable()
    build_nonportable()
    print("done.")


if __name__ == "__main__":
    main()
