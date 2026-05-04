"""mp_tool — reference implementation for v4 model package authoring.

This module provides primitives for creating, merging, and splitting v4 model
packages. It is designed to be usable both as a standalone CLI script and as
a library that can be wrapped by Olive's `olive` CLI later.

Layout produced (per the v4 design proposal):

    <package>/
    ├── manifest.json
    ├── configs/                       # consumer-shared assets (genai_config base,
    │   ├── genai_config.json          # tokenizer, processor configs, chat template)
    │   └── tokenizer.json, ...
    └── <component>/
        ├── metadata.json              # selection-only (variant names + ep_compatibility)
        └── <variant>/
            ├── variant.json           # files + per-file SO/PO + consumer_metadata overlay
            ├── *.onnx, *.onnx.data    # symlinked from source if --link
            └── shared_weights/<sha>/<blob>   # if shared between variants

Schema versions are pinned to "1.0" everywhere.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger("mp_tool")

SCHEMA_VERSION = "1.0"
LARGE_FILE_SUFFIXES = {".onnx", ".data", ".bin", ".xml", ".so", ".dll", ".dylib"}
CONFIG_FILE_PATTERNS = {
    "genai_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "processor_config.json", "audio_processor_config.json",
    "preprocessor_config.json", "normalizer.json", "generation_config.json",
}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=indent, sort_keys=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def install_file(src: Path, dst: Path, *, link: bool = True) -> None:
    """Symlink src to dst for large binaries, copy otherwise.

    If `link=False`, always copies. Symlinks are absolute paths so the package
    remains valid when accessed from anywhere on the same filesystem.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link and src.suffix in LARGE_FILE_SUFFIXES:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    # For symlinks, hash the target.
    real = path.resolve()
    with real.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# JSON merge patch (RFC 7386) helpers
# ---------------------------------------------------------------------------

def merge_patch(target: Any, patch: Any) -> Any:
    """Apply RFC 7386 JSON Merge Patch.

    - dict + dict: per-key merge; null values delete keys; non-dicts replace.
    - any other case: patch replaces target wholesale.
    """
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    if not isinstance(target, dict):
        target = {}
    out = copy.deepcopy(target)
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = merge_patch(out.get(k), v)
    return out


def diff_patch(base: Any, target: Any) -> Any:
    """Compute a JSON Merge Patch that, when applied to `base`, produces `target`.

    - For dicts: recurse, emit only changed/new/removed keys.
    - For non-dicts (or different shapes): wholesale replace.
    - Removed keys are emitted as `null` (RFC 7386 deletion).
    """
    if isinstance(base, dict) and isinstance(target, dict):
        patch: dict[str, Any] = {}
        for k in target:
            if k not in base:
                patch[k] = copy.deepcopy(target[k])
            elif base[k] != target[k]:
                sub = diff_patch(base[k], target[k])
                # Recurse only if both sides were dicts and the diff itself is non-trivial
                if isinstance(base[k], dict) and isinstance(target[k], dict):
                    if sub:
                        patch[k] = sub
                else:
                    patch[k] = sub
        for k in base:
            if k not in target:
                patch[k] = None
        return patch
    return copy.deepcopy(target)


# ---------------------------------------------------------------------------
# Source genai_config normalization
# ---------------------------------------------------------------------------

def strip_runtime_fields(genai: dict) -> dict:
    """Remove runtime fields from a source genai_config.

    Under v4, these facts live in variant.json (per-file SO/PO and external data),
    not in the genai_config tree:
      - `session_options` (anywhere in the tree)
      - top-level `model.<role>.filename` (a single-file role's filename is named by
        variant.json's `files[0].filename`)

    Pipeline stage `filename` keys are STRUCTURAL (GenAI reads them to identify
    each stage and match it to a file in the variant directory) and are KEPT.
    `config_filename` (vision/speech processor configs) is also kept — it points
    into `configs/`, not into variant directories.
    """
    g = copy.deepcopy(genai)

    def _scrub_session_options(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("session_options", None)
            for v in node.values():
                _scrub_session_options(v)
        elif isinstance(node, list):
            for item in node:
                _scrub_session_options(item)

    _scrub_session_options(g)

    # Top-level role filename only
    model = g.get("model", {})
    for key, val in list(model.items()):
        if isinstance(val, dict):
            val.pop("filename", None)
    return g


def extract_runtime_fields(genai: dict, role: str) -> tuple[dict | None, list[dict]]:
    """Return (session_options, provider_options) for the given role from a source genai_config.

    GenAI's source schema has them at model.<role>.session_options{ "log_id": ..., "provider_options": [...] }.
    Returns flat dicts suitable for variant.json.
    """
    role_block = genai.get("model", {}).get(role, {})
    so_src = role_block.get("session_options") or {}
    po_src = so_src.get("provider_options") or []

    # Flatten session_options: drop the provider_options sub-key, and drop
    # `log_id` (it's a GenAI-specific debug tag synthesized at runtime, not a
    # static package fact).
    so_flat = {k: v for k, v in so_src.items()
               if k not in ("provider_options", "log_id")}

    # provider_options in source is a list of single-key dicts: [{"cuda": {...}}, {"webgpu": {...}}]
    # Convert to v4 shape: flat {key: value} dict scoped to the selected EP.
    # If the source has multiple entries (rare), only the first is used; v4 variants
    # are EP-specific so multi-EP po is not meaningful at the variant level.
    po_flat: dict = {}
    for entry in po_src:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        (_ep_lower, opts), = entry.items()
        if opts:
            po_flat = dict(opts)
        break

    return (so_flat or None, po_flat or None)


# Common GenAI EP-name normalizations (source genai_config uses lowercase aliases).
_EP_NAME_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "ryzenai": "RyzenAIExecutionProvider",
    "nvtensorrtrtx": "NvTensorRtRtxExecutionProvider",
    "trt": "NvTensorRtRtxExecutionProvider",
    "trtrtx": "NvTensorRtRtxExecutionProvider",
}


def _ep_canonical_name(name: str) -> str:
    return _EP_NAME_MAP.get(name.lower(), name)


# ---------------------------------------------------------------------------
# Package authoring primitives
# ---------------------------------------------------------------------------

def annotate_pipeline_run_on_cpu(genai: dict, source_genai: dict) -> dict:
    """Annotate pipeline stages with `run_on_cpu: true` based on source provider_options.

    For each role under `model.<role>.pipeline` in `source_genai`, walk the
    pipeline stages. A stage whose `session_options.provider_options` is
    absent or empty in the source is taken to be CPU-bound (it falls back to
    the role-level CPU defaults at runtime). We mirror that fact onto the
    matching stage in `genai` by setting `run_on_cpu: true`.

    The returned dict is `genai` mutated in place — callers typically pass an
    overlay (which has only the diff'd pipeline structure) and the original
    source genai_config (which still has the SO/PO it was authored with).

    Stages that do have provider_options in source are left untouched: those
    use the variant's selected EP at runtime, which is the default behavior
    when `run_on_cpu` is absent.
    """
    src_model = source_genai.get("model", {})
    dst_model = genai.get("model", {})
    for role, src_role_block in src_model.items():
        if not isinstance(src_role_block, dict):
            continue
        src_pipe = src_role_block.get("pipeline")
        dst_role_block = dst_model.get(role)
        if not isinstance(src_pipe, list) or not isinstance(dst_role_block, dict):
            continue
        dst_pipe = dst_role_block.get("pipeline")
        if not isinstance(dst_pipe, list):
            continue

        # Build a name -> "is_cpu_bound" map from the source pipeline.
        # Pipeline stages may be either {name: body} single-key dicts (one per
        # list element) OR a single list element containing multiple stages
        # as keys; handle both by iterating all (name, body) pairs in each.
        cpu_bound: dict[str, bool] = {}
        for src_stage in src_pipe:
            if not isinstance(src_stage, dict):
                continue
            for name, body in src_stage.items():
                if not isinstance(body, dict):
                    continue
                so = body.get("session_options") or {}
                po = so.get("provider_options")
                cpu_bound[name] = not po  # missing OR empty -> CPU

        # Stamp run_on_cpu onto matching stages in the destination pipeline.
        for dst_stage in dst_pipe:
            if not isinstance(dst_stage, dict):
                continue
            for name, body in dst_stage.items():
                if not isinstance(body, dict):
                    continue
                if cpu_bound.get(name):
                    body["run_on_cpu"] = True

    return genai


class PackageBuilder:
    def __init__(self, pkg_dir: Path, package_name: str, description: str = "", *, link: bool = True):
        self.pkg_dir = pkg_dir
        self.package_name = package_name
        self.description = description
        self.link = link
        # role -> {variant -> {"files": [...], "overlay": {...}, "ep_compat": {EpName: [strings]}}}
        self._components: dict[str, dict[str, dict]] = {}
        self._configs_dir_files: dict[str, Path] = {}  # destination basename -> source path
        self._base_genai: dict | None = None
        self._role_to_component: dict[str, str] = {}

    # ----- component / variant -----

    def set_role_component(self, role: str, component_name: str) -> None:
        """Map a GenAI role (decoder, vision, encoder, ...) to a package component."""
        self._role_to_component[role] = component_name

    def add_variant(
        self,
        component: str,
        variant: str,
        *,
        ep_compatibility: list[dict],
        files: list[dict],
        consumer_metadata: dict | None = None,
    ) -> None:
        """Register a variant.

        `ep_compatibility` is a list of `{ep, device?, compatibility?}` entries.
        Each entry advertises that this variant works with the named EP, optionally
        targeting a specific device id (e.g. "GPU", "NPU"), with an optional list
        of EP-side compatibility strings the EP can use to choose between matching
        variants. `files` items: {filename, source_path,
        session_options?, provider_options?, colocated_files?, shared_files?}.
        """
        comp = self._components.setdefault(component, {})
        if variant in comp:
            raise ValueError(f"Variant {component}/{variant} already added")
        comp[variant] = {
            "ep_compatibility": ep_compatibility,
            "files": files,
            "consumer_metadata": consumer_metadata or {},
        }

    def add_configs_file(self, src: Path, dest_name: str | None = None) -> None:
        """Register a file (or directory) to be installed under <pkg>/configs/."""
        name = dest_name or src.name
        self._configs_dir_files[name] = src

    def set_base_genai_config(self, genai: dict) -> None:
        self._base_genai = genai

    # ----- emit -----

    def write(self) -> None:
        if self._base_genai is None:
            raise ValueError("Base genai_config must be set via set_base_genai_config()")
        self.pkg_dir.mkdir(parents=True, exist_ok=True)

        # 1. configs/
        configs_dir = self.pkg_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        write_json(configs_dir / "genai_config.json", self._base_genai)
        for name, src in self._configs_dir_files.items():
            dst = configs_dir / name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                install_file(src, dst, link=False)  # always copy small config files

        # 2. components & variants
        for component_name, variants in self._components.items():
            comp_dir = self.pkg_dir / component_name
            comp_dir.mkdir(parents=True, exist_ok=True)

            metadata_variants: dict[str, dict] = {}
            for variant_name, vdata in variants.items():
                metadata_variants[variant_name] = {
                    "ep_compatibility": vdata["ep_compatibility"],
                }
                self._write_variant(comp_dir, variant_name, vdata)

            metadata = {
                "schema_version": SCHEMA_VERSION,
                "component_name": component_name,
                "variants": metadata_variants,
            }
            write_json(comp_dir / "metadata.json", metadata)

        # 3. manifest
        components = [
            {"name": name, "metadata": f"{name}/metadata.json"}
            for name in self._components
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "package_name": self.package_name,
            "package_version": "1.0",
            "description": self.description,
            "components": components,
            "configs_dir": "configs",
        }
        write_json(self.pkg_dir / "manifest.json", manifest)

    def _write_variant(self, comp_dir: Path, variant_name: str, vdata: dict) -> None:
        """Install variant files and emit variant.json.

        Note: per the v4 spec, variant.json `files[]` entries do NOT enumerate
        external-data files. ORT discovers them from the ONNX graph's internal
        references. We still install them alongside the .onnx at authoring time
        (driven by the recipe's `colocated_files` list, formerly
        `external_data_files`), but they are not surfaced in variant.json.

        `shared_files` is the exception: a map `{graph-filename: checksum}` that
        tells consumers a graph reference resolves to a blob in the package's
        shared-weights directory rather than next to the .onnx. Listed in
        variant.json so consumers can build the external-initializer mapping.

        `run_on_cpu` is intentionally NOT written to variant.json — it is a
        consumer-side orchestration flag (see `annotate_pipeline_run_on_cpu`)
        carried in the consumer's `genai_config_overlay` pipeline stages.
        """
        variant_dir = comp_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        files_entries = []
        installed_filenames: set[str] = set()
        for file_spec in vdata["files"]:
            filename = file_spec["filename"]
            src = Path(file_spec["source_path"])
            dst = variant_dir / filename
            install_file(src, dst, link=self.link)
            installed_filenames.add(filename)

            entry: dict = {"filename": filename}
            if file_spec.get("session_options"):
                entry["session_options"] = file_spec["session_options"]
            if file_spec.get("provider_options"):
                entry["provider_options"] = file_spec["provider_options"]
            if file_spec.get("shared_files"):
                # {graph_filename: checksum} per v4 spec
                entry["shared_files"] = file_spec["shared_files"]
            files_entries.append(entry)

            # Install colocated files (external data, op libraries, etc.) alongside
            # the .onnx. They are NOT enumerated in variant.json — ORT resolves
            # external data via the .onnx graph's internal pointers; other sidecar
            # files are picked up by EP-specific paths (custom_ops_library, etc.).
            for co_name in file_spec.get("colocated_files", []):
                if co_name in installed_filenames:
                    continue
                co_src = src.parent / co_name
                if co_src.exists():
                    install_file(co_src, variant_dir / co_name, link=self.link)
                    installed_filenames.add(co_name)

        variant_json = {
            "schema_version": SCHEMA_VERSION,
            "files": files_entries,
            "consumer_metadata": vdata.get("consumer_metadata") or {},
        }
        write_json(variant_dir / "variant.json", variant_json)


# ---------------------------------------------------------------------------
# Merge / split helpers (used by both authoring and the future CLI subcommand)
# ---------------------------------------------------------------------------

def split_genai_for_overlay(
    base_genai: dict,
    variant_genai: dict,
    *,
    role: str,
) -> dict:
    """Compute the overlay diff for one variant of a single role.

    Strips runtime fields from both inputs first (session_options, filename, etc),
    then computes the JSON Merge Patch from base → variant. The resulting overlay
    only carries fields where this variant differs from the base.
    """
    base_clean = strip_runtime_fields(base_genai)
    variant_clean = strip_runtime_fields(variant_genai)
    return diff_patch(base_clean, variant_clean)


def merge_overlays_for_test(base_genai: dict, overlays: Iterable[dict]) -> dict:
    """Apply a sequence of overlays to the base config, returning the merged view.

    This mirrors what GenAI does at runtime. Overlays are applied in order.
    Useful for test/verification of split round-trips.
    """
    out = copy.deepcopy(base_genai)
    for o in overlays:
        out = merge_patch(out, o)
    return out


# ---------------------------------------------------------------------------
# CLI entrypoints (placeholder; the real Olive integration is a future task)
# ---------------------------------------------------------------------------

def _cli_inspect(args: argparse.Namespace) -> int:
    pkg = Path(args.package)
    print(json.dumps(read_json(pkg / "manifest.json"), indent=2))
    for comp in (pkg / ".").iterdir():
        if comp.is_dir() and (comp / "metadata.json").exists():
            print(f"--- {comp.name}/metadata.json ---")
            print(json.dumps(read_json(comp / "metadata.json"), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v4 model package authoring tool (reference impl)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_insp = sub.add_parser("inspect", help="Print manifest + per-component metadata")
    p_insp.add_argument("package", help="Path to a v4 model package directory")
    p_insp.set_defaults(func=_cli_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
