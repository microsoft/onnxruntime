# v4 Model Package CLI — reference materials

This directory contains:

- **`model-package-cli-spec.md`** — Spec for `olive model-package <subcommand>`.
  Defines schemas (manifest / metadata / variant), subcommand surface, and
  cross-component overlay conflict policy.
- **`mp_tool.py`** — Reference Python implementation of the authoring
  primitives (`PackageBuilder`, `diff_patch`, `merge_patch`,
  `strip_runtime_fields`, …). Importable as a library and runnable as a CLI:
  `python mp_tool.py inspect <pkg>`.
- **`build_packages.py`** — Worked example that uses `mp_tool` to build
  four reference v4 packages from raw model directories. Also exercises
  multi-file (QNN), multi-component (whisper, nemotron, qwen3-vl), and
  mock variants (phi trtrtx-gpu sourced from qwen2.5).

Run:
```
python build_packages.py
```
to (re-)create the four packages under
`packages/`. Symlinks are used by
default for large ONNX / data files; pass `--copy` to deep-copy.

## Round-trip verification

For all 12 variants across the four packages, applying the per-variant
`consumer_metadata.genai_config_overlay` to `configs/genai_config.json`
reproduces the source genai_config exactly (after stripping runtime
fields). See the "round-trip" check in this session's history; the script
is preserved at the bottom of `build_packages.py` (function unused in
production but referenced here for posterity).

## Reference packages built

| Package | Components | Variants | Notes |
| --- | --- | --- | --- |
| `phi-4-mini-reasoning.v4.ortpackage` | decoder | cpu, cuda, webgpu, vitis-npu, openvino-gpu, openvino-npu, qnn-npu, trtrtx-gpu (mock) | Single-component, multi-EP. QNN is a 4-file pipeline. trtrtx-gpu is a mock for ABI/wiring tests only. |
| `openai-whisper-small.v4.ortpackage` | encoder, decoder, jump_times | cpu (fp32), cuda (fp16) | Demonstrates per-component overlay split. `jump_times` is an auxiliary component (not a session input). |
| `nemotron-speech-streaming-en-0.6b.v4.ortpackage` | encoder, decoder, joiner, vad | cpu only | 4-component RNN-T streaming demo. Single variant ⇒ all overlays empty. |
| `qwen3-vl-2b-instruct.v4.ortpackage` | vision, embedding, decoder | cpu, cuda | Multimodal package; vision points at `processor_config.json` in `configs/`. |

## Open questions flagged for review

1. `manifest.json` uses `schema_version` (was `format_version` in v3). OK?
2. `metadata.json` `ep_compatibility` is now a map keyed by EP name with a
   list of compat strings (was per-file in v3). Empty list = no extra
   refinement. Confirm.
3. Mock TRT variant uses qwen2.5's ONNX as the file content. The overlay
   carries a `_mock` marker. Should we add a more formal `mock: true` flag
   to `metadata.json` so EP scoring can avoid auto-selecting mock variants
   in production? *Currently nothing prevents selection.*
4. Whisper `jump_times` packaging: separate component vs. auxiliary file
   inside the decoder variant? Implemented as a separate component to keep
   variant.json files[] focused on session-input ONNXs. Reconsider if
   GenAI's whisper handler doesn't naturally find jump_times that way.
5. Cross-component overlay conflict policy is documented (decoder wins);
   none of the four reference packages exercise it. Need test cases.
6. Reference impl symlinks for large files. The `olive` integration likely
   needs `--copy` as the default for portability across machines.
