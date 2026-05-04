"""Build the four reference v4 model packages.

Run from anywhere:
    python build_packages.py [--copy] [--out <dir>]

Default writes packages to packages/.
By default, large ONNX/data files are symlinked from the source models. Use
--copy to deep-copy instead (slow, GBs).

This is also a worked example of how to drive `mp_tool` from Python; the same
primitives will back the future Olive `olive` CLI subcommand.
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
from pathlib import Path

import mp_tool as mp

logger = logging.getLogger("build")

MODELS_ROOT = Path("models")
QWEN25_TRT = MODELS_ROOT / "qwen2.5-0.5b-instruct" / "trtrtx_gpu"

# Default ep_compatibility lists — keep simple. EP-side preference logic will
# look at these; for v4 demos, we mark NPU variants with soc tags where known
# (qnn) and leave others as the EP default.
EP_COMPAT_BASE: dict[str, list[str]] = {
    "CPUExecutionProvider": [],
    "CUDAExecutionProvider": [],
    "WebGpuExecutionProvider": [],
    "VitisAIExecutionProvider": [],
    "OpenVINOExecutionProvider": [],
    "NvTensorRtRtxExecutionProvider": [],
}


# ---------------------------------------------------------------------------
# Helper to compose a {role: variant_genai_subset} dict from source genai_config
# ---------------------------------------------------------------------------

def _role_subsets(genai: dict, roles: list[str]) -> dict[str, dict]:
    """Given a source genai_config and a list of roles, return per-role architectural slice.

    The per-role slice excludes session_options/filename (those go to variant.json).
    """
    out = {}
    for r in roles:
        block = genai.get("model", {}).get(r, {})
        clean = {k: copy.deepcopy(v) for k, v in block.items()
                 if k not in ("session_options", "filename")}
        out[r] = clean
    return out


def _shared_skeleton(genai: dict, roles: list[str]) -> dict:
    """Make the architecture-only base genai_config: everything except runtime fields."""
    base = mp.strip_runtime_fields(genai)
    # Add a "component" mapping under each role so GenAI can find the right
    # package component for that role. Default: role name == component name.
    for r in roles:
        rb = base.get("model", {}).get(r)
        if isinstance(rb, dict):
            rb.setdefault("component", r)
    return base


# ---------------------------------------------------------------------------
# Phi-4-mini-reasoning (single component "decoder")
# ---------------------------------------------------------------------------

def build_phi4(out_root: Path, *, link: bool) -> None:
    src = MODELS_ROOT / "Phi-4-mini-reasoning"
    pkg_dir = out_root / "phi-4-mini-reasoning.v4.ortpackage"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)

    builder = mp.PackageBuilder(
        pkg_dir,
        package_name="phi-4-mini-reasoning",
        description="Phi-4-mini reasoning, single decoder component, multi-EP variants.",
        link=link,
    )
    builder.set_role_component("decoder", "decoder")

    # Base genai_config: use cpu_and_mobile/genai_config.json minus runtime fields,
    # add the role->component mapping.
    cpu_genai = mp.read_json(src / "cpu_and_mobile" / "genai_config.json")
    base = _shared_skeleton(cpu_genai, ["decoder"])
    builder.set_base_genai_config(base)

    # ----- configs/: tokenizer + chat template -----
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "vocab.json", "merges.txt"]:
        p = src / "cpu_and_mobile" / fname
        if p.exists():
            builder.add_configs_file(p)
    chat_tpl = src / "openvino_npu" / "chat_template.jinja"
    if chat_tpl.exists():
        builder.add_configs_file(chat_tpl)

    # ----- variants -----
    def _add(variant_name: str, src_dir: Path, ep: str, files: list[dict],
             *,
             device: str | None = None,
             compat_strings: list[str] | None = None,
             overlay: dict | None = None) -> None:
        # Compute overlay diff from this variant's genai_config relative to base.
        var_genai_path = src_dir / "genai_config.json"
        if var_genai_path.exists() and overlay is None:
            var_genai = mp.read_json(var_genai_path)
            overlay = mp.diff_patch(
                _shared_skeleton(cpu_genai, ["decoder"]),
                _shared_skeleton(var_genai, ["decoder"]),
            )
            # If the source variant has a pipeline, the overlay carries it but
            # without runtime fields. Annotate stages whose source had no
            # provider_options as run_on_cpu so consumers can still pin those
            # files to CPU when constructing their per-stage sessions.
            mp.annotate_pipeline_run_on_cpu(overlay, var_genai)
        consumer_meta = {}
        if overlay:
            consumer_meta["genai_config_overlay"] = overlay

        ep_entry: dict = {"ep": ep}
        if device:
            ep_entry["device"] = device
        if compat_strings:
            ep_entry["compatibility"] = compat_strings

        builder.add_variant(
            "decoder", variant_name,
            ep_compatibility=[ep_entry],
            files=files,
            consumer_metadata=consumer_meta,
        )

    # cpu
    _add("cpu", src / "cpu_and_mobile", "CPUExecutionProvider", files=[{
        "filename": "model.onnx",
        "source_path": str(src / "cpu_and_mobile" / "model.onnx"),
        "colocated_files": ["model.onnx.data"],
    }])

    # cuda
    _add("cuda", src / "cuda", "CUDAExecutionProvider", files=[{
        "filename": "model.onnx",
        "source_path": str(src / "cuda" / "model.onnx"),
        "colocated_files": ["model.onnx.data"],
        "provider_options": {"enable_cuda_graph": "0"},
    }])

    # webgpu
    _add("webgpu", src / "webgpu", "WebGpuExecutionProvider", files=[{
        "filename": "model.onnx",
        "source_path": str(src / "webgpu" / "model.onnx"),
        "colocated_files": ["model.onnx.data"],
    }])

    # vitis-npu (source dir is "vitia_npu" — typo in source)
    _add("vitis-npu", src / "vitia_npu", "VitisAIExecutionProvider", files=[{
        "filename": "model.onnx",
        "source_path": str(src / "vitia_npu" / "model.onnx"),
        "colocated_files": ["model.onnx.data"],
        "session_options": {
            "custom_ops_library": "onnxruntime_vitis_ai_custom_ops.dll",
        },
    }])

    # openvino-gpu
    _add("openvino-gpu", src / "openvino_gpu", "OpenVINOExecutionProvider", files=[{
        "filename": "openvino_model_dy.onnx",
        "source_path": str(src / "openvino_gpu" / "openvino_model_dy.onnx"),
        "colocated_files": ["openvino_model_dy.bin", "openvino_model_dy.xml"],
        "provider_options": {"device_type": "GPU"},
    }], device="GPU")

    # openvino-npu (context_length=4224, so overlay carries it)
    _add("openvino-npu", src / "openvino_npu", "OpenVINOExecutionProvider", files=[{
        "filename": "openvino_model_dy.onnx",
        "source_path": str(src / "openvino_npu" / "openvino_model_dy.onnx"),
        "colocated_files": ["openvino_model_dy.bin", "openvino_model_dy.xml"],
        "provider_options": {"device_type": "NPU"},
    }], device="NPU")

    # qnn-npu — multi-file pipeline. Variant.json files[] enumerates each file; the
    # overlay carries the model.decoder.pipeline (and type=decoder-pipeline) since
    # that's what differs from base for this variant.
    qnn_dir = src / "qnn_npu"
    qnn_files = [
        {
            "filename": "phi_4_mini_embeddings.all.quant.onnx",
            "source_path": str(qnn_dir / "phi_4_mini_embeddings.all.quant.onnx"),
            "session_options": {"intra_op_num_threads": 3, "inter_op_num_threads": 1},
        },
        {
            "filename": "phi_4_mini_ctx.onnx_ctx.onnx",
            "source_path": str(qnn_dir / "phi_4_mini_ctx.onnx_ctx.onnx"),
            "colocated_files": [b for b in ["phi_4_mini_cb_1.bin", "phi_4_mini_cb_2.bin"] if (qnn_dir / b).exists()],
            "session_options": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            "provider_options": {"htp_performance_mode": "burst",
                                  "htp_graph_finalization_optimization_mode": "3",
                                  "soc_model": "60"},
        },
        {
            "filename": "phi_4_mini_iter.onnx_ctx.onnx",
            "source_path": str(qnn_dir / "phi_4_mini_iter.onnx_ctx.onnx"),
            "colocated_files": [b for b in ["phi_4_mini_cb_3.bin", "phi_4_mini_cb_4.bin"] if (qnn_dir / b).exists()],
            "session_options": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            "provider_options": {"htp_performance_mode": "burst",
                                  "htp_graph_finalization_optimization_mode": "3",
                                  "soc_model": "60"},
        },
        {
            "filename": "phi_4_mini_lm_head.all.quant.onnx",
            "source_path": str(qnn_dir / "phi_4_mini_lm_head.all.quant.onnx"),
            "session_options": {"intra_op_num_threads": 3, "inter_op_num_threads": 1},
        },
    ]
    _add("qnn-npu", qnn_dir, "QNNExecutionProvider", files=qnn_files,
         compat_strings=["soc_model_60", "soc_model_69"])

    # mock trtrtx-gpu — sourced from qwen2.5/trtrtx_gpu (mock for ABI testing only)
    _add("trtrtx-gpu", QWEN25_TRT, "NvTensorRtRtxExecutionProvider", files=[{
        "filename": "model.onnx",
        "source_path": str(QWEN25_TRT / "model.onnx"),
        "colocated_files": ["model.onnx.data"],
        "provider_options": {"enable_cuda_graph": "1"},
    }],
    overlay={"_mock": "Variant ONNX is sourced from qwen2.5-0.5b; structurally compatible decoder for ABI/wiring tests only."})

    builder.write()
    _write_decisions(pkg_dir, """\
# phi-4-mini-reasoning v4 package — decisions

- Single component named `decoder` (matches role).
- 8 variants: cpu, cuda, webgpu, vitis-npu, openvino-gpu, openvino-npu, qnn-npu, trtrtx-gpu (mock).
- `vitis-npu` source dir name in upstream is `vitia_npu` (upstream typo); fixed in package.
- `openvino-npu` overlay carries `model.context_length=4224` (downcast from 131072).
- `qnn-npu` overlay carries the pipeline definition: `model.type` becomes `decoder-pipeline`,
  context length becomes 4096, and the `model.decoder.pipeline` array is set. The four
  pipeline files are listed individually in `variant.json` `files[]` with their own SO/PO.
- `trtrtx-gpu` is a *mock* variant: ONNX content is sourced from `qwen2.5-0.5b-instruct/trtrtx_gpu`.
  The overlay carries a `_mock` flag for tooling visibility. Do not use for inference correctness tests.
- Per-variant SO/PO live ONLY in `variant.json files[]`. The base genai_config and
  per-variant overlays are runtime-field-free per the v4 design proposal.
- ep_compatibility shape: `{"<EpName>": ["<compat-string>", ...]}`. Empty list = "no special compat".
""")


# ---------------------------------------------------------------------------
# Whisper-small (multi-component: encoder, decoder, jump_times)
# ---------------------------------------------------------------------------

def build_whisper(out_root: Path, *, link: bool) -> None:
    src = MODELS_ROOT / "openai-whisper-small"
    pkg_dir = out_root / "openai-whisper-small.v4.ortpackage"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)

    builder = mp.PackageBuilder(
        pkg_dir, package_name="openai-whisper-small",
        description="Whisper-small encoder/decoder (+ jump_times) — multi-component v4 demo.",
        link=link,
    )
    for r in ["encoder", "decoder"]:
        builder.set_role_component(r, r)
    # jump_times is a package component but not a model.<role> in genai_config —
    # it's an auxiliary file referenced by GenAI's word-timestamps path.
    builder.set_role_component("jump_times", "jump_times")

    cpu_genai = mp.read_json(src / "cpu_and_mobile" / "genai_config.json")
    base = _shared_skeleton(cpu_genai, ["encoder", "decoder"])
    # Document the auxiliary jump_times component in the base config.
    base.setdefault("model", {})["jump_times"] = {
        "component": "jump_times",
        "description": "Auxiliary file used by word-level timestamping. Not a session input.",
    }
    builder.set_base_genai_config(base)

    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "vocab.json", "merges.txt", "normalizer.json",
                  "preprocessor_config.json", "audio_processor_config.json"]:
        p = src / "cpu_and_mobile" / fname
        if p.exists():
            builder.add_configs_file(p)

    # ----- variants per component -----
    variants = [
        ("cpu", "cpu_and_mobile", "fp32", "CPUExecutionProvider"),
        ("cuda", "cuda", "fp16", "CUDAExecutionProvider"),
    ]

    for v_name, v_src, dtype, ep in variants:
        v_dir = src / v_src
        v_genai = mp.read_json(v_dir / "genai_config.json")
        # Per-component overlay: each component owns its `model.<role>` slice diff.
        full_overlay = mp.diff_patch(_shared_skeleton(cpu_genai, ["encoder", "decoder"]),
                                     _shared_skeleton(v_genai, ["encoder", "decoder"]))

        # Encoder
        enc_overlay = _scope_overlay_to_role(full_overlay, "encoder")
        builder.add_variant("encoder", v_name,
            ep_compatibility=[{"ep": ep}],
            files=[{
                "filename": f"whisper-small_encoder_{dtype}.onnx",
                "source_path": str(v_dir / f"whisper-small_encoder_{dtype}.onnx"),
                "colocated_files": [f"whisper-small_encoder_{dtype}.onnx.data"]
                                        if (v_dir / f"whisper-small_encoder_{dtype}.onnx.data").exists() else [],
            }],
            consumer_metadata=({"genai_config_overlay": enc_overlay} if enc_overlay else {}))

        # Decoder
        dec_overlay = _scope_overlay_to_role(full_overlay, "decoder")
        builder.add_variant("decoder", v_name,
            ep_compatibility=[{"ep": ep}],
            files=[{
                "filename": f"whisper-small_decoder_{dtype}.onnx",
                "source_path": str(v_dir / f"whisper-small_decoder_{dtype}.onnx"),
                "colocated_files": [f"whisper-small_decoder_{dtype}.onnx.data"]
                                        if (v_dir / f"whisper-small_decoder_{dtype}.onnx.data").exists() else [],
            }],
            consumer_metadata=({"genai_config_overlay": dec_overlay} if dec_overlay else {}))

        # jump_times (aux component)
        jt = v_dir / f"whisper-small_jump_times_{dtype}.onnx"
        if jt.exists():
            builder.add_variant("jump_times", v_name,
                ep_compatibility=[{"ep": ep}],
                files=[{
                    "filename": f"whisper-small_jump_times_{dtype}.onnx",
                    "source_path": str(jt),
                }],
                consumer_metadata={})

    builder.write()
    _write_decisions(pkg_dir, """\
# openai-whisper-small v4 package — decisions

- Three components: `encoder`, `decoder`, `jump_times` (auxiliary, used by word-timestamps path).
- Two variants per component: `cpu` (fp32) and `cuda` (fp16).
- Filenames retain the upstream `whisper-small_<role>_<dtype>.onnx` naming so external data
  references inside the ONNX graphs continue to resolve. (Renaming would require ONNX surgery.)
- Per-variant per-component overlays carry only the `model.<role>` slice diff between the
  base config (cpu) and that variant. For cpu, all overlays are empty by construction.
- Top-level base genai_config has a synthetic `model.jump_times` block that ONLY records
  the package-component mapping (not a runtime input block). GenAI's whisper handler can
  ignore it; the v4 SDK ignores unknown roles by spec.
- Cross-component overlay conflict policy: not exercised here — encoder and decoder do not
  modify each other's keys, and neither touches top-level fields. If a future variant did,
  the *primary* component (decoder) would win for top-level keys.
""")


def _scope_overlay_to_role(full_overlay: dict, role: str) -> dict:
    """Take a full-genai diff and restrict to the slice owned by `role`.

    Returns a dict like {"model": {role: {...}}} containing only the keys that diff
    inside the role's block. Returns {} if no diff in that role.
    """
    role_block = full_overlay.get("model", {}).get(role)
    if role_block in (None, {}):
        return {}
    return {"model": {role: copy.deepcopy(role_block)}}


# ---------------------------------------------------------------------------
# Nemotron speech (multi-component: encoder, decoder, joiner, vad)
# ---------------------------------------------------------------------------

def build_nemotron(out_root: Path, *, link: bool) -> None:
    src = MODELS_ROOT / "nemotron-speech-streaming-en-0.6b" / "cpu_and_mobile"
    pkg_dir = out_root / "nemotron-speech-streaming-en-0.6b.v4.ortpackage"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)

    builder = mp.PackageBuilder(
        pkg_dir, package_name="nemotron-speech-streaming-en-0.6b",
        description="Nemotron streaming speech RNN-T (encoder + decoder + joiner + VAD).",
        link=link,
    )
    roles = ["encoder", "decoder", "joiner", "vad"]
    for r in roles:
        builder.set_role_component(r, r)

    src_genai = mp.read_json(src / "genai_config.json")
    base = _shared_skeleton(src_genai, roles)
    builder.set_base_genai_config(base)

    for fname in ["tokenizer.json", "tokenizer_config.json", "audio_processor_config.json",
                  "vocab.txt"]:
        p = src / fname
        if p.exists():
            builder.add_configs_file(p)

    role_files = {
        "encoder": ("encoder.onnx", ["encoder.onnx.data"]),
        "decoder": ("decoder.onnx", ["decoder.onnx.data"]),
        "joiner":  ("joint.onnx",   ["joint.onnx.data"]),
        "vad":     ("silero_vad.onnx", []),
    }
    for role, (filename, ext) in role_files.items():
        ext_present = [e for e in ext if (src / e).exists()]
        builder.add_variant(role, "cpu",
            ep_compatibility=[{"ep": "CPUExecutionProvider"}],
            files=[{
                "filename": filename,
                "source_path": str(src / filename),
                "colocated_files": ext_present,
            }],
            consumer_metadata={})  # single variant => no overlay needed

    builder.write()
    _write_decisions(pkg_dir, """\
# nemotron-speech-streaming v4 package — decisions

- Four components: `encoder`, `decoder`, `joiner`, `vad`. (RNN-T-style streaming model.)
- Single `cpu` variant per component (only EP shipped upstream).
- Source has an extra `model.onnx` at the variant root with no genai_config reference;
  it's omitted from the package. (Likely an unused all-in-one export.)
- All four component overlays are empty (single variant => base IS the variant).
  The package still demonstrates the overlay-split mechanism: each component
  *would* write to `model.<role>.*` if a second variant were introduced.
- vad is a real component (its own ONNX file) — not an encoder sub-graph.
""")


# ---------------------------------------------------------------------------
# Qwen3-VL (multi-component: vision, embedding, decoder)
# ---------------------------------------------------------------------------

def build_qwen3vl(out_root: Path, *, link: bool) -> None:
    src = MODELS_ROOT / "qwen3-vl-2b-instruct"
    pkg_dir = out_root / "qwen3-vl-2b-instruct.v4.ortpackage"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)

    builder = mp.PackageBuilder(
        pkg_dir, package_name="qwen3-vl-2b-instruct",
        description="Qwen3-VL 2B Instruct multimodal (vision + embedding + decoder).",
        link=link,
    )
    # GenAI roles: vision, embedding, decoder (the source genai uses model.decoder
    # which references text.onnx; we keep the role name "decoder" but the package
    # component is also "decoder" — the upstream filename is text.onnx).
    builder.set_role_component("vision", "vision")
    builder.set_role_component("embedding", "embedding")
    builder.set_role_component("decoder", "decoder")

    cpu_genai = mp.read_json(src / "cpu_and_mobile" / "genai_config.json")
    roles = ["vision", "embedding", "decoder"]
    base = _shared_skeleton(cpu_genai, roles)
    builder.set_base_genai_config(base)

    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
                  "processor_config.json"]:
        p = src / "cpu_and_mobile" / fname
        if p.exists():
            builder.add_configs_file(p)

    variants = [
        ("cpu", "cpu_and_mobile", "CPUExecutionProvider"),
        ("cuda", "cuda", "CUDAExecutionProvider"),
    ]
    role_filenames = {"vision": "vision.onnx", "embedding": "embedding.onnx", "decoder": "text.onnx"}

    for v_name, v_src, ep in variants:
        v_dir = src / v_src
        v_genai = mp.read_json(v_dir / "genai_config.json")
        full_overlay = mp.diff_patch(_shared_skeleton(cpu_genai, roles),
                                     _shared_skeleton(v_genai, roles))

        for role, filename in role_filenames.items():
            scoped = _scope_overlay_to_role(full_overlay, role)
            ext = [f"{filename}.data"] if (v_dir / f"{filename}.data").exists() else []
            builder.add_variant(role, v_name,
                ep_compatibility=[{"ep": ep}],
                files=[{
                    "filename": filename,
                    "source_path": str(v_dir / filename),
                    "colocated_files": ext,
                }],
                consumer_metadata=({"genai_config_overlay": scoped} if scoped else {}))

    builder.write()
    _write_decisions(pkg_dir, """\
# qwen3-vl-2b-instruct v4 package — decisions

- Three components: `vision`, `embedding`, `decoder`.
- The decoder ONNX upstream filename is `text.onnx`; we keep the filename in `variant.json`
  files[] and use the component name `decoder` to match the GenAI role.
- Two variants per component: `cpu` and `cuda`. Both share the same architecture so per-role
  overlays are empty after stripping runtime fields. SO/PO diffs are captured in `variant.json`.
- The vision component's `config_filename: processor_config.json` field points into `configs/`
  per the v4 two-roots layout (configs are package-rooted, not variant-rooted).
- This package exercises the multi-component overlay split mechanism:
  the same source genai_config is fanned out into three role-scoped slices (vision, embedding,
  decoder) when computing each variant's overlay.
- Cross-component conflict policy: decoder is the primary component for `qwen3_vl`. If any
  top-level key (e.g. context_length) is touched by multiple components' overlays in a future
  variant, decoder's wins.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_decisions(pkg_dir: Path, body: str) -> None:
    (pkg_dir / "DECISIONS.md").write_text(body)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="packages")
    parser.add_argument("--copy", action="store_true",
                        help="Deep-copy ONNX/data instead of symlinking")
    parser.add_argument("--only", choices=["phi", "whisper", "nemotron", "qwen3vl"],
                        help="Build only one package (debug)")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    link = not args.copy

    targets = {"phi": build_phi4, "whisper": build_whisper,
               "nemotron": build_nemotron, "qwen3vl": build_qwen3vl}
    if args.only:
        targets = {args.only: targets[args.only]}

    for name, fn in targets.items():
        logger.info("Building %s ...", name)
        fn(out_root, link=link)
        logger.info("  done.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
