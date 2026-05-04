# Olive `olive` CLI Spec for v4 Model Packages

This document specifies the CLI surface for creating v4 model packages from
flat-directory model dirs, designed to be implemented as `olive` subcommands
in the Olive toolkit.

The reference Python primitives live in `mp_tool.py` next to this doc, and
`build_packages.py` is a worked example that wraps `mp_tool` to produce the
four reference packages.

**Scope.** This spec defines two subcommands only: `create` and `add`. The
intent is that a user can take a stack of flat-directory model dirs (one per
EP build) and assemble a v4 package incrementally without ever editing JSON
by hand. Re-canonicalizing the base config (`merge`), splitting back out to
flat dirs (`unmerge`), validation, inspection, etc. are out of scope for the
first cut and listed under [Open follow-ups](#open-follow-ups).

---

## Glossary

- **Package**: A directory whose root contains `manifest.json`, a `configs/`
  directory of consumer-shared assets (genai_config base, tokenizer, processor
  configs, chat templates), and one subdirectory per **component**.
- **Component**: A logical role in a model architecture — `decoder`,
  `encoder`, `vision`, `embedding`, `joiner`, `vad`, etc. Each component
  contains a `metadata.json` and one subdirectory per **variant**.
- **Variant**: A single shippable build of a component for a given target
  (CPU build, CUDA build, QNN NPU build, …). Each variant is a directory
  with a `variant.json` and the actual model files (ONNX + external data,
  custom op libraries, LoRA adapters, …).
- **Overlay**: A JSON Merge Patch (RFC 7386) authored per-variant in
  `variant.json → consumer_metadata.genai_config_overlay`, expressing how
  this variant's slice of the consumer's config differs from the base in
  `configs/genai_config.json`.

---

## File schemas (v4, schema_version `1.0`)

### `manifest.json`
```jsonc
{
  "schema_version": "1.0",
  "package_name": "<string>",
  "package_version": "1.0",
  "description": "<string>",
  "components": [{"name": "<string>", "metadata": "<rel-path>"}],
  "configs_dir": "configs"
}
```

### `<component>/metadata.json`
Selection-only — never carries per-file detail.
```jsonc
{
  "schema_version": "1.0",
  "component_name": "<string>",
  "variants": {
    "<variant-name>": {
      "ep_compatibility": [
        {
          "ep": "<EpName>",
          "device": "<id>",          // optional, EP-specific (e.g. "GPU", "NPU")
          "compatibility": ["<str>", ...]  // optional, EP-side preference strings
        }
      ]
    }
  }
}
```
`ep_compatibility` is a list — a variant may be compatible with multiple
EPs (e.g. CUDA + CPU fallback). Each entry pins an `ep` and optionally a
`device` and a list of EP-side compatibility strings. Empty / missing
`compatibility` means *EP-default, no extra refinement*. The selection
algorithm filters by `ep` (and `device` if specified), then asks the
chosen EP to score the matching variants by their `compatibility` lists.

### `<component>/<variant>/variant.json`
```jsonc
{
  "schema_version": "1.0",
  "files": [
    {
      "filename": "<basename>",
      "session_options": { "intra_op_num_threads": 1, ... },
      "provider_options": {"<key>": "<value>", ...},
      "shared_files": { "<graph-filename>": "<sha256>" }
    }
  ],
  "consumer_metadata": {
    "genai_config_overlay": { /* RFC 7386 patch */ }
  }
}
```

Per-file CPU pinning (e.g. "embedding and lm_head should run on CPU even
though this is a QNN variant") is **not** a `variant.json` fact. It is a
consumer-side orchestration concern carried in the
`consumer_metadata.genai_config_overlay` pipeline stages — for ORT-GenAI,
the existing `run_on_cpu: true` key on a `pipeline[]` stage. The package
authoring tool annotates pipeline stages whose source had no
`provider_options` (i.e. the source relied on role-level CPU defaults) so
the same intent survives the runtime-field stripping.

External-data files (e.g. `model.onnx.data`) are **not** enumerated in
`variant.json`. ORT discovers them via the .onnx graph's internal
references and resolves them relative to the .onnx file's directory. The
authoring tool still installs them alongside the .onnx in the variant
directory.

`provider_options` is a flat `{key: value}` map scoped to the EP that
selected this variant. There is no list-of-EPs shape because variants are
EP-specific (the EP-side preference logic chose the variant for a single
EP context). If two EPs need different provider options for the same
weights, they should be authored as separate variants.

`shared_files` maps a filename-as-referenced-by-the-onnx-graph to the
sha256 checksum of a shared-weight blob located under the package's
`shared_weights/<sha256>/` directory. Consumers resolve checksums to
absolute paths via `ComponentInstanceGetSharedWeightPath`. Filenames not
listed in `shared_files` are assumed to live next to the .onnx.

---

## Subcommands

### `olive model-package create`

Initialize a new v4 package from a single flat-directory model dir. The flat
dir is the legacy GenAI shape: a `genai_config.json`, one or more `.onnx`
files (with `.data` / `.bin` / `.xml` siblings as needed), and tokenizer /
processor / chat-template files alongside.

```
olive model-package create <flat-dir> \
  --output <pkg-dir> \
  --name <package-name> \
  --variant <variant-name> \
  --ep <EpName> [--device <id>] [--compat <str> ...] \
  [--description <text>] \
  [--symlink|--copy]
```

What `create` does:

1. Parse `<flat-dir>/genai_config.json`. Walk the top-level `model.*` blocks
   that look like roles (presence of `filename` *or* a `pipeline` list)
   to discover the **components**. For each role found, prepare a component
   directory of the same name under `<pkg-dir>/`. (A typical decoder-only
   model produces one component named `decoder`; whisper produces three —
   `encoder`, `decoder`, `jump_times`; multimodal produces one per role.)
2. Build the **package base** at `<pkg-dir>/configs/genai_config.json` by
   running `strip_runtime_fields` on the source config (drops
   `session_options` everywhere and top-level `model.<role>.filename`,
   keeps pipeline-stage `filename` keys and `config_filename` pointers).
   This is the "generic-enough" skeleton meant to remain stable as
   subsequent `add` calls layer in new variants.
3. For each role:
   - If the role is single-file (`model.<role>.filename` is set), the
     variant has one entry in `files[]` taken from that filename. SO/PO
     for the file are extracted from `model.<role>.session_options`
     (`log_id` is dropped; `provider_options` list is flattened to a
     single `{key: value}` map for the variant's EP).
   - If the role is a pipeline (`model.type == "decoder-pipeline"` or
     `model.<role>.pipeline` is a list), each stage becomes one entry in
     `files[]`. SO/PO are extracted from the stage's own `session_options`
     (none if the stage was CPU-bound in the source). The CPU-bound nature
     of a stage is preserved in the **overlay** (see step 4), not in
     `variant.json`.
4. Compute the variant's `consumer_metadata.genai_config_overlay` as the
   JSON Merge Patch (RFC 7386) needed on top of the package base to recover
   the source's per-role view. For pipeline roles, the post-strip overlay
   carries `pipeline[]` stages with only `filename`/`inputs`/`outputs`. The
   tool then walks the source pipeline once more and stamps
   `run_on_cpu: true` onto stages whose source had no `provider_options` so
   the consumer can still pin those files to CPU at session-construction
   time. (Implemented by `mp_tool.annotate_pipeline_run_on_cpu`.) For a
   single source flat dir, the rest of the overlay is typically empty — the
   source *is* the base.
5. Write `<pkg-dir>/<component>/metadata.json` with one entry under
   `variants` containing the structured `ep_compatibility` list:
   `[{ep, device?, compatibility?}]`.
6. Install ONNX files plus their colocated sidecars (`.data`, `.bin`,
   `.xml`, …) into `<pkg-dir>/<component>/<variant>/` (symlink by default,
   copy with `--copy`).
7. Copy tokenizer / processor / chat-template files into
   `<pkg-dir>/configs/`. Heuristic: any `*.json`, `*.txt`, or `*.model`
   file in the source flat-dir that isn't `genai_config.json` and isn't
   referenced by a `model.*.filename` is treated as a config asset.
8. Write `manifest.json` with the discovered component list and
   `configs_dir: "configs"`.

### `olive model-package add`

Append a new EP variant to an existing v4 package, sourcing it from a second
flat-directory model dir.

```
olive model-package add <pkg-dir> <flat-dir> \
  --variant <variant-name> \
  --ep <EpName> [--device <id>] [--compat <str> ...] \
  [--symlink|--copy]
```

What `add` does:

1. Validate `<pkg-dir>` looks like a v4 package (`manifest.json` present,
   `configs/genai_config.json` present).
2. Parse `<flat-dir>/genai_config.json` and discover its roles (same logic
   as `create`).
3. **Cross-check role set.** The new flat-dir's roles must match the package's
   existing component set 1-to-1. Mismatch is an error — adding a new
   component is a separate authoring operation and is not in scope for this
   command. (Out-of-scope component additions are an [open follow-up](#open-follow-ups).)
4. For each role, run the same per-file extraction (`files[]`, SO/PO) used
   by `create`.
5. Compute each component's overlay as `diff_patch(package_base_role_slice,
   stripped_source_role_slice)`, then run `annotate_pipeline_run_on_cpu`
   over it (using the original source genai_config) so pipeline-stage CPU
   pinning survives. Write the result into the variant's
   `consumer_metadata.genai_config_overlay`.
6. Append a new entry under `<component>/metadata.json` `variants` with the
   `ep_compatibility` list. Refuse if the variant name already exists.
7. Install the ONNX files + colocated sidecars under
   `<pkg-dir>/<component>/<variant>/`.
8. Tokenizer / processor / chat-template files in the new flat-dir are
   compared against `<pkg-dir>/configs/`. Identical-content files (by
   sha256) are ignored. Non-identical files raise a warning and are
   ignored unless `--force-config-overwrite` is passed (typically you
   *don't* want different EP builds to ship different tokenizer assets;
   the warning surfaces a producer bug).

### Notes on "generic-enough" base

The base `configs/genai_config.json` written by `create` is, by
construction, exactly the source config with runtime fields stripped. If
the source flat-dir is a CPU build that happens to use single-file decoder
form, the base reflects that. If the first flat-dir is a QNN
`decoder-pipeline` build, the base inherits the pipeline shape — and a
later CPU `add` will need a non-trivial overlay to flip `model.type` and
remove `model.decoder.pipeline`.

Practical guidance: **author `create` from the most "vanilla" flat-dir
first** (typically the CPU build for the decoder family). Subsequent EP
variants then express only their genuine deltas as overlay. Re-canonicalizing
the base after batch-importing many variants (the "merge" operation) is a
follow-up.

---

## Cross-component overlay conflict

Both `create` and `add` produce per-component overlays whose patches are
applied independently to the package base by the consumer. When two
components write the same top-level key (e.g. `model.context_length`)
with different values, the v4 design says the **primary component** wins
at consumer-side merge time. The mapping is encoded in
`mp_tool.PRIMARY_ROLE_FOR_TYPE`:

| `model.type`        | Primary component |
| ------------------- | ----------------- |
| `phi3`, `qwen2`,…   | `decoder`         |
| `decoder-pipeline`  | `decoder`         |
| `whisper`           | `decoder`         |
| `qwen3_vl`          | `decoder`         |
| `nemotron_speech`   | `decoder`         |

The CLI does not detect or warn on conflicts in the first cut — that is a
concern for a future `validate` subcommand. Authors of multi-component
packages are expected to keep variant configs consistent on shared keys.

---

## Implementation notes for Olive

1. The `mp_tool.PackageBuilder` class is the imperative authoring API.
   The `olive` subcommands above are thin wrappers that translate flag-shaped
   user input into builder calls + a final `builder.write()`.
2. `mp_tool.diff_patch` / `mp_tool.merge_patch` are RFC 7386 Merge Patch
   (compatible with what GenAI's `OgaConfigOverlay` does at runtime).
3. `mp_tool.strip_runtime_fields` drops `session_options` everywhere and
   top-level `model.<role>.filename`. Keep pipeline stage `filename` keys
   (structural) and `config_filename` (points into `configs/`).
   Authoring tools also strip `log_id` from extracted session_options —
   it's a GenAI-specific debug tag synthesized at runtime, not a package fact.
4. EP name canonicalization (`mp_tool._EP_NAME_MAP`) handles GenAI's
   lowercase aliases (`cuda`, `webgpu`, `qnn`, …) → ONNX Runtime canonical
   `*ExecutionProvider` strings.
5. Symlink large binaries by default (`.onnx`, `.data`, `.bin`, `.xml`,
   `.so`, `.dll`, `.dylib`); copy small JSON / text files. Add a `--copy`
   flag for users who need fully-relocatable packages.

## Open follow-ups

These are out of scope for the first cut (`create` + `add` only) but
expected to land later:

1. **`merge`**: recompute `configs/genai_config.json` as the intersection
   of all variants' stripped configs and recompute every variant's overlay
   relative to the new base. Useful after batch-importing variants whose
   first arrival biased the base. The reverse direction (`unmerge`/`split`
   back to flat dirs) is *not* planned for Olive — that round-trip lives
   on the consumer / runtime side.
2. **`validate`**: schema-version checks, file-existence checks, overlay
   apply-cleanly checks, cross-component conflict detection on shared
   top-level keys.
3. **Component additions in `add`**: today `add` rejects flat-dirs whose
   role set differs from the package's. A future flag (e.g. `--allow-new-components`)
   could let `add` introduce new components alongside existing variants
   for an expanding multimodal model.
4. **Shared-weights extraction**: `variant.json` `files[].shared_files` is
   reserved. The CLI should learn to detect identical external-data blobs
   across variants, hash them, and dedupe into `<component>/shared_weights/<sha256>/`
   with `shared_files` pointers.
5. **Manifest signing**: the v4 design hints at a `manifest.sig` for
   provenance. Out of scope for the reference impl.
6. **Olive integration tests**: when wiring into `olive`, add fixture
   packages (small test ONNX) covering single- and multi-component cases.
7. **Authoring from Olive workflow output**: extend Olive's existing
   `generate-model-package` (in `olive/cli/model_package.py`) to emit v4
   format directly when `--format v4` is passed.
