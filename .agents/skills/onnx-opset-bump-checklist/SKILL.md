---
name: onnx-opset-bump-checklist
description: Step-by-step checklist for bumping the pinned ONNX dependency / opset in ONNX Runtime (e.g. ONNX 1.21 / opset 26 → 1.22 / opset 27). Use when integrating a new ONNX release or release-candidate, updating the cmake/deps.txt onnx pin or the cmake/external/onnx submodule, regenerating cmake/patches/onnx/onnx.patch, raising kMaxSupportedOpset, or adding a new opset's CPU kernels. Covers the file taxonomy, archive-hash procedures, patch rebase/mirror rules, the RC→formal strategy, and the optimizer/EP gotchas that the automated audit script misses.
---

# ONNX Opset / Version Bump Checklist for ONNX Runtime

Repeatable process for upgrading the ONNX dependency in ONNX Runtime. Recurs on every ONNX
release. Canonical reference PRs: **#27601** (ONNX 1.21, incremental rc1→formal — best
example), #26579 (1.20.1), #25678 (1.19). See also `docs/How_To_Update_ONNX_Dev_Notes.md`.

## 0. Strategy: RC → formal (incremental)

ONNX partner-validates each release candidate against ORT **before** publishing the formal
release — that is the whole point of the integration issue. When the issue points at a
`rel-X.Y.0` branch:

1. **Integrate the RC first.** Pin by the release-branch **HEAD commit** (the `vX.Y.0rcN`
   git *tag* usually does not exist yet). Run the full build + tests; file ONNX bugs upstream
   (tag the ONNX release manager) for any ONNX-side defects found.
2. **Re-pin per RC** (rc2, rc3, …). Usually only the Group-A version-plumbing files change.
   **Drop any `onnx.patch` hunks that ONNX has merged upstream** in the new RC.
3. **Re-pin to the formal tag** `vX.Y.0` once the GitHub Release is published. Note the
   release can sit as a **draft** (no git tag, `git/ref/tags/vX.Y.0` → 404) for a while.

```bash
gh api repos/onnx/onnx/git/ref/heads/rel-X.Y.0 --jq '.object.sha'              # RC branch HEAD
curl -sL https://raw.githubusercontent.com/onnx/onnx/rel-X.Y.0/VERSION_NUMBER  # e.g. X.Y.0rcN
gh api 'repos/onnx/onnx/releases?per_page=5' --jq '.[].tag_name'               # '' => draft
```

## 1. File taxonomy — what to change

Grouped so parallel work is safe. **Group A must land first** (the tree must build before
B/C/D can be validated). Throughout this section, bold letters in parentheses (e.g. gotcha
**a**, **g**) refer to the lettered gotchas **a**–**i** defined in §4.

### Group A — version plumbing (always required, mechanical)
| File | Note |
|---|---|
| `cmake/deps.txt` (`onnx;` line) | archive URL + **SHA1 of the `.zip`** (see §2). ⚠️ Before advancing this pin, verify the target commit still ships `onnx/backend/test/data/node/` — see gotcha **p** (#7959 deletes the on-disk node-test corpus → silent-green CI). |
| `cmake/external/onnx` (submodule) | `git -C cmake/external/onnx fetch && git -C cmake/external/onnx reset --hard <sha> && git add cmake/external/onnx` |
| `cmake/vcpkg-ports/onnx/vcpkg.json` | `version-semver`; reset `port-version` to 0 on a real version bump |
| `cmake/vcpkg-ports/onnx/portfile.cmake` | `REF` + **SHA512 of the `.tar.gz`** (see §2). RC = bare commit as `REF`; formal = `REF "v${VERSION}"` |
| `cmake/vcpkg-ports/onnx/binskim.patch` | keep **byte-identical** to `onnx.patch` (see §3) |
| `cmake/patches/onnx/onnx.patch` | rebase to the new source (see §3) |
| `cmake/vcpkg-ports/onnx/fix-dependency-protobuf.patch`, `fix-cmakelists.patch` | re-`diff` only if they fail to apply |
| **The 7 `requirements.txt` files** with `onnx==X.Y.Z` | `onnxruntime/test/python/requirements.txt`; `tools/ci_build/github/linux/python/requirements.txt`; `tools/ci_build/github/windows/python/requirements.txt`; `tools/ci_build/github/linux/docker/scripts/{requirements.txt,manylinux/requirements.txt,lort/requirements.txt}`; `tools/ci_build/github/linux/docker/inference/aarch64/python/cpu/scripts/requirements.txt`. Confirm with `git grep -n "onnx==<OLD>"` (e.g. `onnx==1.21`) — grep the **old** pin you are replacing, not bare `onnx==1`. ⚠️ **Do NOT bump all matches.** `git grep -n "onnx==1"` also lists 3 transformers-model files — `onnxruntime/python/tools/transformers/models/{llama,phi2,stable_diffusion}/requirements.txt` — that are **intentionally frozen at `onnx==1.18.0`**. Leave those alone; only the 7 CI files above get the new pin. |
| **The 4 QNN/android CI yaml with an INLINE `onnx==X.Y.Z` pip pin** (node-test materialization legs — NOT covered by the `requirements.txt` grep above) | `tools/ci_build/github/azure-pipelines/linux-qnn-ci-pipeline.yml`; `tools/ci_build/github/azure-pipelines/win-qnn-arm64-ci-pipeline.yml`; `tools/ci_build/github/azure-pipelines/android-arm64-v8a-QNN-crosscompile-ci-pipeline.yml`; `.github/workflows/windows_qnn_x64.yml`. Each has a `pip install onnx==X.Y.Z "numpy==...; ..."` step feeding the `onnxruntime_MATERIALIZE_ONNX_NODE_TESTS` gate (see gotcha **p**); the pin must be bumped in lockstep with `cmake/deps.txt`. Confirm with `git grep -n "onnx==<OLD>" -- '*.yml'`. (These are inline, not `-r requirements.txt`, by design — pulling the full requirements would drag heavy unrelated test deps onto the QNN legs.) |

### Group B — opset enablement
| File | Change |
|---|---|
| `onnxruntime/core/optimizer/transpose_optimization/optimizer_api.h` | `kMaxSupportedOpset` → new max opset (e.g. 26 → 27) |
| `onnxruntime/core/providers/cpu/cpu_execution_provider.cc` | add `// Opset N` forward-declares + `BuildKernelCreateInfo<...>` entries for new/updated CPU kernels; mirror the previous opset block exactly |
| `onnxruntime/core/graph/contrib_ops/contrib_defs.h`, `dml_ops/dml_defs.h` | apply ONNX-header-driven `OpSchemaRegisterOnce` macro fixes **only if the build emits those errors** |

> **🆕 Tradition: bump EVERY EP that registers the op, in the SAME PR.** When an op's kernel set
> changes for the new opset (e.g. `Range` gaining fp16/bf16 at opset 27), version-split / bump
> that op's registration in **every** EP that registers it — **CPU and CUDA at minimum** — so no
> EP silently lags behind CPU and the advertised opset boundaries stay consistent. **Even an
> open-ended kernel that already binds the new opset** (e.g. CUDA `Range` at `SinceVersion(11)`,
> which already matches opset-27 nodes) should still be **version-split** for convention/clarity
> and to keep the kernel's advertised boundary matching the schema. Worked example — **PR #28754**
> split `Range` into `[11,26]` + `27` in **both** CPU and CUDA (verified), keeping the same
> numeric type set and deferring fp16/bf16 to ONNX function-expansion.

**EP checklist when an op's kernel set changes for the new opset:**
- [ ] For **each** EP, `grep -rn "<Op>)" onnxruntime/core/providers/<ep>/` (and its `*_execution_provider.cc`) to find every registration of the changed op.
- [ ] EPs that register ONNX kernels via the `ONNX_OPERATOR_[VERSIONED_]KERNEL[_CLASS_NAME]` macros — **cpu**, **cuda**, **js**, and **rocm** if it registers the op (rocm is often hipified from cuda): version-split each into `[prev_start, N-1]` + a new `N` registration (class forward-declare **and** `BuildKernelCreateInfo` entry).
- [ ] EPs with their **own** registration systems assess per their conventions, not the macro split — **dml** (`REG_INFO(ver, Op, …)` in `OperatorRegistration.cpp`), **webgpu**, **coreml/nnapi/qnn/openvino/migraphx**. A partition/capability check (e.g. MIGraphX's `optype == "Range"`) is **not** a kernel registration and needs no split.
- [ ] Bump the EP `GetMaxSupportedOpSet` ceilings (**coreml/nnapi/vsinpu/webnn**) in lockstep — see §4 gotcha **b**.

> **IR version is NOT bumped manually.** ORT reads `ONNX_NAMESPACE::Version::IR_VERSION` from
> the ONNX headers (`onnxruntime/core/graph/model.cc`); it follows the submodule automatically.

### Group C — docs & test data (mostly auto-generated)
| File | Change |
|---|---|
| `docs/OperatorKernels.md` | regenerate (see gotcha **e** — needs a built ORT module): `python tools/python/gen_opkernel_doc.py --output_path docs/OperatorKernels.md` |
| `js/web/docs/webgl-operators.md` | `cd js/web && npm install && npm run build:doc` (the WebAssembly CI "Check out of dated documents" stage fails otherwise) |
| `onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc` | exclude backend tests for ops whose kernels are deferred (gotcha **g**) |
| `onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc` | tolerance overrides for new tests |
| `onnxruntime/test/onnx/TestCase.cc` | `broken_tests` entries for genuinely-unsupported cases |

### Group D — conditional (touch only if build/tests flag it — but see §4 gotchas)
`onnxruntime/core/framework/kernel_type_str_resolver_utils.cc`,
`onnxruntime/core/optimizer/layout_transformation/layout_transformation_potentially_added_ops.h`,
`onnxruntime/core/optimizer/qdq_transformer/qdq_util.cc`, EP `base_op_builder.h` max-opset
guards (gotcha **b**), optimizer fusion path-matchers (gotcha **a**, **c**), and any CPU op
file hard-coding a `SinceVersion` ceiling for a changed op.

## 2. Archive-hash procedures

`cmake/deps.txt` 3rd field = **SHA1 of the downloaded `.zip`** (verified: `sha1sum` of
`v1.21.0.zip` equals the pinned `321d4acc...`):
```bash
URL="https://github.com/onnx/onnx/archive/<commit-or-refs/tags/vX.Y.0>.zip"
curl -sL -o onnx.zip "$URL" && sha1sum onnx.zip      # paste:  onnx;$URL;<sha1>
```
vcpkg `portfile.cmake` uses **SHA512 of the `.tar.gz`**:
```bash
curl -sL -o onnx.tgz "https://github.com/onnx/onnx/archive/<commit-or-tag>.tar.gz"
sha512sum onnx.tgz
```
Shortcut: pin a wrong hash and build — ORT/FetchContent prints the expected one.

> The submodule field (a git commit SHA) and the deps.txt field (a SHA1 of the archive
> *contents*) are **different by design** — never interchange them.

> **vcpkg artifact must be mirrored** to the Microsoft vcpkg store before `--use_vcpkg`
> builds can download it (Terrapin upload — see below). Not self-service; coordinate with
> infra. This is gotcha **f**.

### Mirroring the ONNX archive to the MS vcpkg store (required before `--use_vcpkg` builds pass)

The SHA512 in `portfile.cmake` references a `.tar.gz` that vcpkg downloads from the **Microsoft
vcpkg artifact mirror**, not from GitHub. Until that exact archive is uploaded to the mirror,
`--use_vcpkg` builds fail with a download/hash error. The upload is done with the **Terrapin
Retrieval Tool** and is a **Windows + `az` auth + internal-credential** step — it **cannot** run
from a generic Linux CI host.

**Step 1 — auth (PowerShell):**
```powershell
$authScope = 'https://mspmecloud.onmicrosoft.com/RebuildManager.Web/.default'
$env:TRT_UPLOAD_AUTH_TOKEN = $(az account get-access-token --scope $authScope --query 'accessToken' --output tsv)
```

**Step 2a — commit-version (RC phase, archive keyed by commit SHA):**
```powershell
C:\local\Terrapin\TerrapinRetrievalTool.exe -b https://vcpkg.storage.devpackages.microsoft.io/artifacts/ `
  -a true -u Environment `
  -p https://github.com/onnx/onnx/archive/<commit-sha>.tar.gz `
  -s <sha512-of-tar.gz> `
  -d "<build>\Windows\vcpkg\downloads\onnx-onnx-<commit-sha>.tar.gz.part"
```

**Step 2b — tag-version (formal phase, archive keyed by release tag):**
```powershell
C:\local\Terrapin\TerrapinRetrievalTool.exe -b https://vcpkg.storage.devpackages.microsoft.io/artifacts/ `
  -a true -u Environment `
  -p https://github.com/onnx/onnx/archive/refs/tags/v<version>.tar.gz `
  -s <sha512> `
  -d "<build>\Windows\vcpkg\downloads\onnx-onnx-v<version>.tar.gz"
```

Key notes:
- **(a)** `-s <sha512>` **MUST equal** the `portfile.cmake` SHA512 (the `sha512sum` of the same
  `.tar.gz` — see §2). A mismatch re-uploads the wrong blob and the hash check still fails.
- **(b)** This is *why* `--use_vcpkg` builds fail with a download error until the upload lands —
  the mirror has no copy of the new archive yet.
- **(c)** Microsoft-internal infra step: coordinate with the infra owner (e.g. **@snnn**).
  External / Linux CI **cannot** perform it (needs Windows, the Terrapin tool, and `az` creds).
- **(d)** The **`cmake/deps.txt` path does NOT need this** — that build fetches the `.zip`
  straight from GitHub. Only the vcpkg (`--use_vcpkg`) path depends on the mirror.
- **(e)** The `.part` suffix on the 2a `-d` path is vcpkg's **in-progress-download** temp name
  (vcpkg renames it to the final `.tar.gz` once the hash verifies). Match whatever the failing
  `--use_vcpkg` build actually requests in its `downloads/` dir — the RC run wrote a `.part`;
  the formal-tag run wrote the bare `.tar.gz`. When unsure, copy the exact path from the build's
  download error.
- **(f) Why there is no GitHub fallback — `x-block-origin` (the actual failure mode).**
  `tools/ci_build/build.py` (`add_default_vcpkg_options`, the `--use_vcpkg_ms_internal_asset_cache`
  branch) configures the asset cache as
  `--x-asset-sources=x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/;x-block-origin`
  (or the Terrapin `x-script` form). The trailing **`x-block-origin` forbids vcpkg from falling
  back to the GitHub origin** — so if the blob is absent the leg does **not** silently download
  from GitHub, it **hard-fails**. The asset-cache key is the **bare lowercase SHA512 hex, no
  extension**: the blob lives at `…/artifacts/<sha512>`. Quick probe (no auth needed, read is public):
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" \
    https://vcpkg.storage.devpackages.microsoft.io/artifacts/<portfile-sha512>
  # 200 = already mirrored (legs will pass) ; 404 = NOT mirrored (every vcpkg leg will 404-fail)
  ```
- **(g) This recurs on EVERY archive bump, including each RC.** rc1→rc2→…→formal each produce a
  *new* `.tar.gz` with a *new* SHA512, so each one is a distinct, un-mirrored blob. A green rc1 run
  does **not** mean rc2 is mirrored. After every `portfile.cmake` REF/SHA512 change, run the curl
  probe above; if 404, the upload (steps 1–2) must happen again before any `--use_vcpkg` leg can pass.
  Terrapin-enabled self-hosted Windows legs (`-a true`) self-seed the mirror as a side effect; the
  read-only `x-azurl` legs (Linux/macOS/GitHub-hosted) cannot and will 404 until that seed lands.
- **(h) CI failure signature (how to recognize this in a red build).** Only the **vcpkg-based**
  legs fail; the `cmake/deps.txt` FetchContent legs stay green (they pull the `.zip` from GitHub,
  mirror-independent — see (d)). The failing leg's log shows a vcpkg download error during the
  `onnx` port install, e.g.:
  ```
  error: Failed to download from mirror set
  error: https://vcpkg.storage.devpackages.microsoft.io/artifacts/<sha512>: failed: status code 404
  error: x-block-origin set, prohibiting access to the original source URL
  ```
  That `404` + `x-block-origin set` pair on a `…/artifacts/<sha512>` URL **is** this gotcha — the
  `<sha512>` in the error equals the `portfile.cmake` SHA512. (Confirmed live for ONNX 1.22.0rc2:
  rc1 blob → HTTP 200, rc2 blob → HTTP 404.)
- **(i) Fix options (in order of preference).**
  1. **Self-seed via a Terrapin Windows leg** — trigger/re-run one internal Azure DevOps pipeline
     whose Windows job runs on a self-hosted pool (has `C:\local\Terrapin\TerrapinRetrievalTool.exe`)
     and passes `--use_vcpkg_ms_internal_asset_cache`. Its `-a true` Terrapin fetches the archive
     from origin and **writes it back to the mirror**; afterwards re-run the failing read-only legs.
     (No infra ticket needed.) Manual Terrapin upload commands are in §2 steps 1–2 above.
  2. **`az storage blob upload`** — anyone with write access to the `devpackages` storage account:
     download the exact origin `.tar.gz`, verify `sha512sum` equals the portfile SHA512, then
     `az storage blob upload --container-name artifacts --name <sha512> --file <tar.gz> --auth-mode login`.
     The blob **name must be the bare lowercase SHA512, no extension**.
  3. **EngSys / 1ES ticket** — if neither self-service path is available, ask the team that owns
     `vcpkg.storage.devpackages.microsoft.io` to mirror the asset, giving them the origin URL + SHA512.
  Verify any fix with the §2(f) curl probe returning **HTTP 200** before re-running CI.

  > Detailed worked example (ONNX 1.22.0rc2 coordinates, live-probe evidence, full procedures):
  > see the architect runbook artifact `architect-f1afcb8a/onnx-rc2-vcpkg-mirror-runbook.md`.

## 3. onnx.patch rebase + binskim.patch mirror

For each hunk in `cmake/patches/onnx/onnx.patch`, against the **new** ONNX source:
- Applies cleanly → keep.
- Context shifted → rebase (regenerate line numbers/indices).
- Fixed upstream in the new ONNX → **drop the hunk** (#27601 added a Slice `dim_value==0`
  hunk for rc1/rc2 then removed it at rc3 once ONNX merged it).
- ONNX restructured the region → **rewrite the hunk.** Example (1.21 → 1.22): ONNX replaced
  the `file(GLOB_RECURSE __tmp_srcs ...)` source-gathering block with an
  `add_library(onnx_core OBJECT)` + `add_subdirectory(onnx)` model, so the `ONNX_MINIMAL_BUILD`
  hunk had to switch from editing `set(ONNX_SRCS ...)` to
  `target_sources(onnx_core PRIVATE "${ONNX_ROOT}/onnx/defs/data_type_utils.cc")`. Note
  `data_type_utils.cc` lives in `onnx/defs/` (not `onnx/common/`), and headers do **not**
  belong in `target_sources` (they are not compile units).
  > **Structural note (re-validate every bump):** in the `onnx_core` OBJECT-lib layout,
  > minimal mode skips `add_subdirectory(onnx)`. That is only safe because
  > `add_onnx_compile_options(onnx_core)` (include dirs + protobuf link) and
  > `add_onnx_global_defines(onnx_core)` are **unconditional at the top-level CMakeLists**.
  > If a future ONNX moves those into the `onnx/` subdirectory, minimal builds will fail to
  > configure/link — so never *assume* the minimal hunk still works; re-run the minimal build
  > (§5) on every bump.

Then **mirror the final `onnx.patch` byte-for-byte into
`cmake/vcpkg-ports/onnx/binskim.patch`** — they must stay identical. Verify:
```bash
git apply --check cmake/patches/onnx/onnx.patch
patch --binary --ignore-whitespace -p1 --dry-run < cmake/patches/onnx/onnx.patch
sha1sum cmake/patches/onnx/onnx.patch cmake/vcpkg-ports/onnx/binskim.patch  # must match
```

## 4. Gotchas (the expensive ones — hand-check these)

These are not caught by the automated audit and have bitten real integrations:

**(a) The optimizer audit script misses fusion path-matchers.**
`tools/python/find_optimizer_opset_version_updates_required.py` only inspects **direct kernel
registrations** — it does **not** see opset version lists embedded in optimizer
`graph_utils::EdgeEndToMatch` **path-matchers**. When an op's opset changes, hand-grep the
fusion files and extend the version list. Confirmed sites for **`Range` → 27**:
- `onnxruntime/core/optimizer/embed_layer_norm_fusion.cc` — `EdgeEndToMatch` entries
  `{0, 0, "Range", {1, 11, 27}, kOnnxDomain}` (multiple `parent_path_*`).
- `onnxruntime/core/optimizer/gather_fusion.cc` — `IsSupportedOptypeVersionAndDomain(node,
  "Range", {1, 11, 27})` (Range→Gather→Slice fusion).
```bash
grep -rn '<ChangedOp>' onnxruntime/core/optimizer/*fusion*.cc | grep -iE 'EdgeEndToMatch|IsSupportedOptypeVersionAndDomain|\{[0-9, ]+\}'
```

**(b) EP `base_op_builder.h` `GetMaxSupportedOpSet` must be bumped in lockstep.**
Each NPU/coreml/web EP caps the opset it will partition. Bump every one that returns the old
max:
- `onnxruntime/core/providers/coreml/builders/impl/base_op_builder.h`
- `onnxruntime/core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h`
- `onnxruntime/core/providers/vsinpu/builders/impl/base_op_builder.h`
- `onnxruntime/core/providers/webnn/builders/impl/base_op_builder.h`
```bash
grep -rn 'GetMaxSupportedOpSet' onnxruntime/core/providers/*/builders/impl/base_op_builder.h
# all should return the NEW max opset (e.g. 27)
```

**(c) Fusion `IsSupportedOptypeVersionAndDomain` version lists** (same root cause as **a**) —
grep all of `onnxruntime/core/optimizer/` for the changed op, not just `*fusion*` files.

**(d) The audit script crashes on placeholder macros.**
`find_optimizer_opset_version_updates_required.py` has a pre-existing crash (a placeholder
`'ver'` token gets parsed as an `int`). Expect it to throw on a clean main; don't treat the
crash as a signal from your change. Run it, but rely on the manual greps in **a–c**.

**(e) `OperatorKernels.md` regeneration needs a built ORT Python module.**
`gen_opkernel_doc.py` imports `onnxruntime`, so build + install the wheel first (or download
the regenerated markdown from a CI published-artifact, which the dev-notes recommend).

**(f) vcpkg artifact mirroring** — see §2; `--use_vcpkg` builds fail to download until done.
Because the vcpkg path is mirror-gated (not self-service), **do not rely on it to validate the
`onnx.patch` rebase.** Use the minimal build instead — it is mirror-independent (see §5 and **(i)**).

**(g) Defer-and-filter brand-new, unimplemented ops — and why it is safe.**
Do not block the version bump on full kernels for large new ops (e.g. stateful
attention/conv). Exclude them in `onnx_backend_test_series_filters.jsonc` and track kernels as
follow-up PRs (#25678 deferred `TensorScatter`/`Swish` this way). Keeps the bump PR reviewable.
This is safe because ORT failures here are **node-local, not model-load-blocking**:
- ORT kernel matching is per-`(op, type-set)`. Registering an updated op for only a **subset**
  of the new schema's types (e.g. `Range`-27 with the 5 common numeric types, deferring
  fp16/bf16) does **not** break model load — the ONNX schema validates the model; only the
  specific unsupported-type node fails kernel lookup with a clear "kernel not found".
- Deferring a brand-new op entirely (no kernel) is likewise node-local: ORT advertises the new
  opset because the **submodule** registers the schemas (`operator_sets.h` /
  `DomainToVersionRange::Map()`), independent of the transpose-optimizer `kMaxSupportedOpset`.
  The model **loads**; the unimplemented node fails at kernel-bind. Precedent: `TensorScatter`
  (24), `LpNormalization` (22, test name `test_l2normalization`), `TreeEnsemble` all sit in the filters file as "not implemented".
- Optional new schema attributes whose default "has no effect for other types" (e.g.
  `Range`-27 `stash_type`) can be ignored by the kernel for the registered types — safe to defer.

**(h) Function-op test filters — don't over-filter the `_expanded` variant.**
Many new/updated ops are ONNX **functions** (have `SetContextDependentFunctionBodyBuilder`) —
e.g. `Range`-27, `CausalConvWithState`, `LinearAttention`. ONNX emits two backend-test
variants: `test_<op>` and `test_<op>_expanded` (the decomposed primitive subgraph). ORT can
usually **run** `_expanded` via the primitive decomposition even when the fused kernel is
deferred. An **unanchored** filter like `^test_range_float16_type` also drops `_expanded`,
silently losing real coverage. When deferring an op, in §5/T7 run `onnx_test_runner` and check
whether the `_expanded` variants pass; if they do, **narrow the filter** (anchor it / exclude
only the bare name) to keep coverage. Over-filtering is CI-safe but hides working paths.

**(i) On a shared multi-agent branch, read clean `main`, not the working tree.**
Another agent may already have rebased `onnx.patch`. Inspect with
`git show <main-sha>:cmake/patches/onnx/onnx.patch` rather than reading the file directly.

**(j) A green Linux webgpu CI leg does NOT mean WebGPU ran the node tests.**
New ONNX backend node tests (`OnnxBackendNodeModelTest`) only **execute** on the
**macOS-arm64** webgpu CI leg. The Linux webgpu leg (`py-linux-webgpu-stage.yml`) is
**build-only** — it compiles the WebGPU EP but runs no kernels. So a green Linux webgpu
leg proves the build, not that any WebGPU op actually ran. To exercise WebGPU kernels
off-Mac locally, use a software Vulkan adapter (Mesa lavapipe) — see the
`webgpu-local-testing` skill.

**(k) Filter vs. override — pick the right one for a newly-failing node test.**
Two test-data files handle failures differently; choosing wrong either hides a real bug
or drops coverage:
- `onnx_backend_test_series_filters.jsonc` = **SKIP** a test for a **real EP bug**.
  Always cite the tracking issue **and** a removal condition (when the skip can be
  deleted). When only the reference **decomposition** path is broken, skip just the
  `_expanded` variant — not the bare test (see gotcha **h**).
- `onnx_backend_test_series_overrides.jsonc` = **RELAX ATOL** for benign fp16/ULP
  differences. This **keeps the test running** — prefer it over a skip whenever the
  kernel is actually correct and the failure is only numeric tolerance. **Guardrail:**
  relax atol only **after** root-causing the diff as ~1 ULP at the output magnitude or a
  few elements (e.g. `5e-4` ≈ 1 fp16 ULP at `O(1)` values). Unexplained, large, or
  **growing** diffs are bugs to **investigate**, not override.

**(l) New upstream reference tests can EXPOSE latent EP bugs.**
A bump pulls in new/updated reference tests that may surface pre-existing EP bugs the
old test set never hit. Example: #28969 — a WebGPU broadcast underflow that ONNX
1.22's expanded-Attention reference tests exposed. Treat such failures as bugs to fix
(or filter-with-issue per gotcha **k**), not as bump noise.

**(m) After a FINAL release, re-seed the vcpkg MS-internal asset mirror or every `--use_vcpkg` leg 404s.**
The MS-internal vcpkg asset mirror (`vcpkg.storage.devpackages.microsoft.io`) is **not**
self-service: the new onnx tag tarball must be **Terrapin-seeded** into it. Until that
lands, every `--use_vcpkg` CI leg fails the download with an `x-block-origin` **404**
(the mirror refuses to proxy an un-seeded asset). This bites at the **rc → FINAL** re-pin
too — the released `vX.Y.0` tag tarball is a *different* asset hash than the RC, so the
mirror must be re-seeded for the final tag. Treat "seed the tag tarball to the vcpkg
mirror" as a required step of the Phase-2 final re-pin, not a follow-up. (See §2 for the
mirror procedure; the `--minimal_build extended` gate in §9 is the mirror-independent
stand-in until seeding completes.)

**(n) A FINAL onnx release can still ship a map-max opset > last *release* opset.**
"FINAL" does **not** imply the new opset is released. ONNX 1.22.0 ships
`DomainToVersionRange` **map-max 27** while the last *released* opset is **26** — i.e.
opset 27 stays **under development** for the entire 1.22 cycle. So strict legs (the
default, or `ALLOW_RELEASED_ONNX_OPSET_ONLY=1`) still throw **"Opset N under development"**
at model load on any `*CurrentOpset` test that builds at the map-max opset — even after
the bump is "final". Don't assume the under-development gating ends when the RC does.

**(o) Prefer per-model `ModelOptions{allow_released_opsets_only=false}` over per-leg env flips or `GTEST_SKIP`.**
For `*CurrentOpset` tests caught by gotcha **n**, set the relaxation **per model** via
`ModelOptions{/*allow_released_opsets_only*/ false, …}` on the `Model::Load` /
`TestGraphTransformer` / `TransformerTester` call. This is **leg-agnostic** (the test then
exercises the new opset on *every* CI leg, not just the ones that happen to set the env
var) and **preserves opset coverage** (unlike `GTEST_SKIP`, which silently drops it).
Avoid flipping a per-leg CI env var (`ALLOW_RELEASED_ONNX_OPSET_ONLY=0`) — it only fixes
the legs you remember to touch and leaves the default-strict legs red. Precedent on this
branch: `38f17243b` (GatherToSlice), generalized to all `*CurrentOpset` fusion tests.
Annotate each call site with a one-line WHY + the tracking issue so it can be removed once
the opset is released (#28966).

**(p) A future onnx commit may DELETE the on-disk node-test corpus — a SILENT-GREEN C++ bump landmine.**
ORT's **C++** node-test coverage depends on ONNX **shipping pre-generated `.pb` artifacts
inside the pinned archive**: `onnx_test_runner` reads `model.onnx` + `test_data_set_*/*.pb`
from `_deps/onnx-src/onnx/backend/test/data/node/<test>/` (the FetchContent archive pinned by
the **SHA1 on the `onnx;` line of `cmake/deps.txt` ~L40** — *not* the `cmake/external/onnx`
submodule, which only the QNN CI `.../node` path uses). **ONNX PR [onnx/onnx#7959] "Remove
node test artifacts" deletes that entire on-disk corpus** (~2992+ files) — it targets onnx
master, so the first affected release is expected to be **onnx 1.23** (one past the current
1.22 pin), consistent with the JS `NOTE(#7959)` "onnx >= 1.23" caveat — **and removes the
`cmd_tools.py generate-data` node path**, replacing both with a **Python-only, in-memory**
flow (`loader.load_node_model_tests()` → `runner.run_node()`, `model_dir=None`, nothing
written to disk). Naming / directory layout / `.pb` format are **unchanged — the files are
simply gone.**
- **⚠️ The C++ failure is SILENT, not loud.** Because names/layout/format don't change, the
  C++ skip contracts (`GetBrokenTests` / `immutable_broken_tests` in `TestCase.cc` +
  `main.cc`, names *without* the `test_` prefix) **don't throw or mismatch** — their target
  paths just cease to exist. An empty `data/node` yields **ZERO** collected cases; the
  runner's directory BFS finds nothing, `ctest`/`add_test` **exit 0**, and
  `onnx_test_runner -e cuda .../node/<test>` prints nothing and **returns success.** C++
  node-test **kernel guarding evaporates behind a GREEN CI with no failure signal** —
  strictly worse than a hard break.
- **The Python leg SURVIVES automatically — do not conflate it.** `onnx_backend_test_series.py`
  does **zero** `.pb`/`model.onnx` disk I/O: it subclasses `onnx.backend.test.runner.Runner`
  and delegates discovery+execution to the **installed pip `onnx`** package, whose base
  `_add_model_test` already branches on `model_dir is None` → in-memory. ORT's only override
  is a rtol/atol injector (`onnx_backend_test_series.py:60-71`), field-agnostic, no disk. So
  the name-keyed **Python** contracts (`onnx_backend_test_series_filters.jsonc` `^test_`
  regexes, `onnx_backend_test_series_overrides.jsonc` atol) keep resolving against the pip
  package's in-memory tests and stay valid. **The mitigation surface is C++-only.** (Residual
  Python risk is only if #7959 changes the public subclass contract — `Runner.__init__` /
  `_add_model_test` / `assert_similar_outputs` sig / `TestCase` field names.)
- **Guardrail — verify the corpus survives BEFORE advancing the `cmake/deps.txt` pin.** As of
  this writing #7959 is **OPEN + CONFLICTING** (no onnx pin includes it yet), so this is
  latent, not active. When re-pinning, confirm the target commit still ships the on-disk tree:
  ```bash
  # Does the target onnx commit/tag still contain the on-disk node-test corpus?
  gh api "repos/onnx/onnx/contents/onnx/backend/test/data/node?ref=<target-commit-or-tag>" \
    --jq 'length'      # >0 = corpus present (safe) ; 404 = DELETED (see #7959) -> STOP
  # note: the contents API caps directory listings at 1000 entries (~1799 dirs exist today),
  # so 'length' returns 1000, not the true count — fine here since we only test >0 vs 404.
  # For an exact count use the git trees API instead:
  #   gh api "repos/onnx/onnx/git/trees/<target-sha>?recursive=1" --jq '[.tree[].path | select(startswith("onnx/backend/test/data/node/"))] | length'
  # Or on a materialized archive/worktree:
  ls _deps/onnx-src/onnx/backend/test/data/node | head   # empty => the landmine has landed
  ```
- **Mitigation if a bump is forced onto a #7959 commit (C++ leg only):** ORT must **own
  node-test materialization** — a CMake `add_custom_command` (gated on
  `onnxruntime_BUILD_UNIT_TESTS`) that imports the ONNX Python case generators (**confirmed
  byte-identical / untouched by #7959** — `onnx/backend/test/case/node/*.py`'s `export.*` +
  `expect(...)` remain; #7959 removes only the serialized `.pb` output, not the source) via
  `load_node_model_tests()`, and re-serializes each `_NodeTestCase` back to the on-disk
  `model.onnx` + `test_data_set_*/{input,output}_N.pb` layout the C++ loader expects (exact
  serialization detail below — **not** `from_array`-only). Then repoint the C++ consumers
  (runner arg + QNN CI `.../node` path). This also collapses the 3-independent-onnx-pins /
  dual-skip-list tangle into one materialized copy. **Reference implementation already exists — vendor it, do not
  re-derive.** Vendor the ~85-line node branch of `onnx/backend/test/cmd_tools.py:generate_data`
  (`cmd_tools.py:64-110` — the very disk-writer #7959 removes) into a build-time script:
  - **Entry point: `collect_testcases(op_type=None)`** — pass `op_type=None` **explicitly**
    to collect ALL ops (it is a required positional; a real `op_type` silently returns a
    1-op subset — another silent-undercount path). It **self-calls `import_recursive`
    internally** (`onnx/backend/test/case/node/__init__.py:417`), so one call both populates
    and returns `_NodeTestCases` — no separate import step, no empty-list footgun.
  - **Per `TestCase`**: write `case.model.SerializeToString()` → `model.onnx`, and serialize
    each input/output into `test_data_set_0/{input,output}_j.pb` using the reference's
    **4-branch dispatch on `graph.input[j].type`** — `numpy_helper.from_dict` (map) /
    `from_list` (sequence) / `from_optional` (optional) / `from_array`-or-`SerializeToString`
    (tensor). **Do NOT hand-simplify to `from_array`-only** — that silently mis-serializes
    every Sequence/Map/Optional node test (`SequenceInsert`, `Optional*`, some Loop/Scan
    fixtures). Dir name = `case.name` (already carries the `test_` prefix). Positional binding
    is by the same `j` as the model's `graph.input`, so any `len(inputs) != len(graph.input)`
    mismatch IndexErrors at BUILD time (fail-loud, never a bad corpus).
  - **MANDATORY version parity (correctness, not just hygiene).** `expect()` stamps each
    `model.onnx`'s `opset_import`/IR from the **compiled onnx RUNTIME's** C++ schema registry
    (`get_schema(op_type, domain).since_version`, `case/node/__init__.py:311-319`), *not* the
    `.py` case source. So the shim MUST run under an installed onnx wheel whose
    `onnx.__version__` **==** the `cmake/deps.txt` pin, **hard-asserted as its first line**
    (`assert onnx.__version__ == <deps-derived>`) — a mismatched wheel bakes the WRONG opset
    into `model.onnx` = silent corpus drift. Make this structural by **deriving the 6 CI
    `requirements.txt` `onnx==` pins FROM `cmake/deps.txt`** so wheel==archive by construction
    (this is also what keeps the *Python* leg testing the pinned version). Under that parity
    the installed wheel's `case/node/*.py` are byte-identical to the archive's, so the shim
    uses the installed wheel directly — no `_deps/onnx-src` source plumbing needed.
  - Also **assert `len(_NodeTestCases) > 0`** at shim start — fail the build on an empty
    materialization (the build-time twin of the runtime tripwire below).
- **Min-count TRIPWIRE (mitigation 0 — highest ROI, cause-AGNOSTIC, do this regardless of
  #7959).** The deepest problem is that corpus absence is **silent-green**, so assert a floor
  on discovered node-test count in **both** harnesses: C++ at `onnx/main.cc:937` right after
  `LoadTests` populates the vector — `ORT_ENFORCE(tests.size() >= floor, ...)` (add a
  `--min_cases N` flag); Python in `onnx_backend_test_series.py` after `create_backend_test()`
  asserting a floor on `len(backend_test.test_cases)`. This converts *any* future
  corpus-absence regression (#7959, a bad archive, broken FetchContent, an over-matching
  filter) from an invisible pass into a **loud red** — the single highest-leverage, lowest-cost
  change here.
- **Other on-disk node-test consumers to repoint when #7959 lands (track, don't lose).** Beyond the
  C++ `onnx_test_runner` + QNN `.../node` path, three more consumers read the ONNX on-disk
  node-test data and will break/skip silently once #7959 deletes it — repoint/update each in the
  same bump: **`csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh`** (C# backend test
  runner pointed at the node dir), **`js/scripts/prepare-onnx-node-tests.ts`** (the JS/web test
  harness that stages the node corpus), and **`docs/python/conf.py`** (Sphinx doc build that
  references the node-test data). None are
  covered by the C++ materialization mitigation above; each needs its own repoint to the
  materialized tree (or removal) when the corpus source disappears.

## 5. Build & validate locally

**The canonical build + test command set lives in §9 (verification gauntlet)** — run those to
define "done". This section keeps only the
*rationale* for the highest-risk gate (don't duplicate the command list here — it drifts):

> **MERGE GATE for the patch rebase (gotcha a/i + §3):** the `ONNX_MINIMAL_BUILD` hunk in
> `onnx.patch`/`binskim.patch` is the single highest-risk change and is **only** exercised when
> `ONNX_MINIMAL_BUILD=ON` — set by `build.py --minimal_build` **and always by the vcpkg port**
> (`tools/python/util/vcpkg_helpers.py`). Since the vcpkg path is mirror-gated (gotcha f), the
> `--minimal_build extended` command (§9 step 2) is the **cheap, mirror-independent gate**: it
> pulls the ONNX archive from `cmake/deps.txt` (no vcpkg mirror needed) and proves the rebased
> minimal hunk **configures + compiles + links**. Make a green minimal build a required gate
> before merging any `onnx.patch` rebase.

Then update `TestCase.cc` / `onnx_backend_test_series_*.jsonc` for newly-introduced failing
node tests. After the PR is up, manually queue every packaging pipeline on the branch, and ask
infra to deploy any new ONNX test data to CI machines (dev-notes).

## 6. Quick checklist
- [ ] Group A: deps.txt (zip SHA1), submodule, vcpkg.json, portfile.cmake (tar.gz SHA512), onnx.patch rebased, binskim.patch mirrored byte-identical, all 7 requirements.txt (NOT the 3 transformers-model files frozen at onnx==1.18.0)
- [ ] Group B: `kMaxSupportedOpset`, cpu_execution_provider.cc opset block, **version-split the changed op in EVERY EP that registers it (cpu+cuda+js, rocm if present) — §1 Group B all-EP tradition**, (contrib/dml macros if build demands)
- [ ] **Safety invariant (§11): every new no-kernel op MUST carry an ONNX function body — else its native kernel is a BLOCKER this PR, not a follow-up**
- [ ] Gotchas: fusion path-matchers (embed_layer_norm_fusion.cc, gather_fusion.cc), all 4 EP `GetMaxSupportedOpSet`, run audit script (expect crash), defer-and-filter new ops (node-local & safe), narrow function-op `_expanded` filters
- [ ] Group C: OperatorKernels.md (built module), webgl-operators.md, backend test filters/overrides
- [ ] Validate: run the **§9 verification gauntlet** (full build, `--minimal_build extended` gate, onnxruntime_test_all, onnx_test_runner -e cpu; `--use_vcpkg` after artifact mirrored)

## 7. Worked example — ONNX 1.21 → 1.22.0rc1 (a template to copy)

The real values from this session's bump. **Replace every value in `<...>` for the next bump**;
the structure stays the same.

| Field | This bump (1.21 → 1.22.0rc1) | Where it goes |
|---|---|---|
| Source pin | commit `bc3be77bec2f628788796dff60819186bacf49df` (`rel-1.22.0` HEAD; RC has **no git tag** yet) | submodule + deps.txt URL + portfile `REF` |
| deps.txt SHA1 | `421e5a9afb6c41a54696e424e5b9a3796aab6821` (**SHA1 of the `.zip`**) | `cmake/deps.txt` 3rd field |
| portfile SHA512 | `e0c526f50767f376b8ad2ac3dc6b109c65f5b3ed20418fd3c4260a954b796d828a30cb5141a23db9b4db8e4db391bfe5042ef99d141f60bdbdbb991b1f3ce467` (**SHA512 of the `.tar.gz`**) | `cmake/vcpkg-ports/onnx/portfile.cmake` |
| Opset | 26 → **27** | `kMaxSupportedOpset`, cpu_execution_provider.cc |
| IR version | **13 (unchanged)** — auto from headers, do not touch | — |
| New/updated ops | `Range`-27 (updated, function), `CausalConvWithState` + `LinearAttention` (new, functions) | see §11 safety invariant |

**Exact files touched this bump** (26 files — use as a coverage checklist):
```
cmake/deps.txt                          # zip SHA1 + URL
cmake/external/onnx                     # submodule -> bc3be77b
cmake/patches/onnx/onnx.patch           # rebase: 2 files / 3 @@ hunks — CMakeLists.txt (option decl + ONNX_MINIMAL_BUILD src restructure) + onnx/defs/nn/old.cc (GroupNormalization-18 .Deprecate() removal). Utils.cmake NOT touched (upstream-merged protobuf hunk dropped this bump)
cmake/vcpkg-ports/onnx/binskim.patch    # byte-identical mirror of onnx.patch
cmake/vcpkg-ports/onnx/portfile.cmake   # REF bc3be77b + tar.gz SHA512
cmake/vcpkg-ports/onnx/vcpkg.json       # version-semver + port-version 0
docs/OperatorKernels.md                 # CPU-only surgical update (full regen = follow-up)
js/web/docs/webgl-operators.md          # npm run build:doc
onnxruntime/core/optimizer/embed_layer_norm_fusion.cc   # Range path-matcher {1,11,27} (gotcha a)
onnxruntime/core/optimizer/gather_fusion.cc             # Range {1,11,27} (gotcha c)
onnxruntime/core/optimizer/transpose_optimization/optimizer_api.h   # kMaxSupportedOpset 26->27
onnxruntime/core/providers/cpu/cpu_execution_provider.cc            # opset-27 kernel block
onnxruntime/core/providers/cpu/generator/range.cc                   # Range-27 kernel
onnxruntime/core/providers/{coreml,nnapi,vsinpu,webnn}/.../base_op_builder.h  # GetMaxSupportedOpSet ->27 (gotcha b)
onnxruntime/test/optimizer/graph_transform_test.cc                  # fusion test updates
onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc    # defer-and-filter new ops
onnxruntime/test/python/requirements.txt + 6 CI requirements.txt    # onnx==1.22.0rc1 (NOT the 3 frozen at 1.18.0)
```

## 8. Hash-recompute one-liners

Both hashes are of the **GitHub source archive at the pinned ref**, but of **different archive
formats** — keep them straight:

```bash
REF=bc3be77bec2f628788796dff60819186bacf49df      # commit (RC) — or  v1.22.0  (formal tag)

# deps.txt 3rd field = SHA1 of the .ZIP:
curl -sL "https://github.com/onnx/onnx/archive/${REF}.zip"    | sha1sum

# portfile.cmake SHA512 = SHA512 of the .TAR.GZ:
curl -sL "https://github.com/onnx/onnx/archive/${REF}.tar.gz" | sha512sum
```
> `.zip` → **SHA1** → deps.txt. `.tar.gz` → **SHA512** → portfile. Never cross them.
> For a formal release use `refs/tags/v<version>` in place of the bare commit in both URLs.

> ⚠️ **RC commit and formal tag produce DIFFERENT hashes — recompute is MANDATORY when you
> switch `REF`.** GitHub names the archive's top-level directory after the ref
> (`onnx-<commit-sha>/` for a commit vs `onnx-<version>/` for `refs/tags/v<version>`), so the
> archive **bytes differ** even though the tree content is identical. **Never reuse the RC
> commit's SHA1/SHA512 for the formal tag** — re-run both one-liners above against the tag.

## 9. Verification gauntlet — what "done" means

Run all of these green before calling a bump complete:

```bash
# 1. Full build (deps.txt path):
./build.sh --config RelWithDebInfo --build_wheel --parallel
# 2. Minimal-build gate (proves the rebased ONNX_MINIMAL_BUILD onnx.patch hunk; mirror-independent):
./build.sh --config RelWithDebInfo --minimal_build extended --parallel --skip_tests
# 3. ORT unit tests:
./build/Linux/RelWithDebInfo/onnxruntime_test_all
# 4. ONNX backend node tests for the new opset on CPU EP:
./build/Linux/RelWithDebInfo/onnx_test_runner -e cpu \
  build/Linux/RelWithDebInfo/_deps/onnx-src/onnx/backend/test/data/node
```
- **(a)** `ALLOW_RELEASED_ONNX_OPSET_ONLY` is **ORT-side load-time validation**
  (`onnxruntime/core/graph/model_load_utils.h`), **not** schema registration — the ONNX schemas
  for the new opset are **always** compiled in from the submodule regardless. With the default
  `=1`, ORT **rejects at model-load** any model whose opset is still "under development" (e.g.
  opset 27 during the RC), so the new-opset node tests fail / report "not implemented". Build and
  run the gauntlet with **`ALLOW_RELEASED_ONNX_OPSET_ONLY=0`** so ORT loads and runs those
  under-development-opset models. (Consistent with §4 gotcha **g**: schemas come from the
  submodule; this flag only governs whether ORT *accepts* the model at load.)
- **(b)** Expect the new-op node tests you intentionally deferred to be **filtered** in
  `onnx_backend_test_series_filters.jsonc` — but confirm the `_expanded` variants still run
  (gotcha **h**): a function op with no native kernel must still PASS via decomposition (§11).
- **(c)** `--use_vcpkg` build is **not** a merge gate until the artifact is mirrored (§2); the
  minimal build (step 2) is the mirror-independent stand-in that proves the patch rebase.

## 10. PR-body template (fill in the blanks)

```markdown
### Bump ONNX to <version> (opset <N>)

Pin: onnx/onnx@<commit-or-tag>. Opset <OLD> → <N>. IR version <unchanged/new>.

**Standing caveats**
- **EP `GetMaxSupportedOpSet` bumped to <N>** (coreml/nnapi/vsinpu/webnn). These EPs gain no
  new op kernels in this PR; the bump only raises the support ceiling so existing ops on
  opset-<N> models partition. New-op kernels for these EPs are follow-ups.
- **`OperatorKernels.md` updated for CPU EP only** (surgical edit). A full multi-EP regen needs
  a built ORT module per EP and is tracked as a follow-up (gotcha **e**).
- **Opset <N> is under development** (RC): ORT load-time validation rejects opset-<N> models
  unless `ALLOW_RELEASED_ONNX_OPSET_ONLY=0` (schemas are always compiled in from the submodule
  — see §9a/§4g). Brand-new ops without kernels are **deferred + filtered**
  in `onnx_backend_test_series_filters.jsonc` (node-local, safe — see skill §4 gotcha g).
- **GPU EPs unverified**: only CPU EP was built/tested here. CUDA/DML/etc. opset-<N> coverage
  is a follow-up.

**Validation:** full build ✅, `--minimal_build extended` ✅, onnxruntime_test_all ✅,
onnx_test_runner -e cpu (opset-<N> node tests) ✅.
```

## 11. SAFETY INVARIANT — no-kernel ops MUST carry a function body

> **🚨 For every new/updated opset op that you ship WITHOUT a native ORT kernel, confirm the
> op carries an ONNX FUNCTION BODY. If it has NO function body, a native kernel is MANDATORY in
> THIS PR — it cannot be a follow-up. Otherwise the model loads but the node fails at session
> initialization.**

**"Native kernel" means** any registered kernel that **binds the opset-N node** — including an
**open-ended existing kernel** (e.g. one registered `SinceVersion(13)` with no upper bound) that
already covers opset N. Such an op is **already kernel-covered** and does **not** trigger the
blocker below; the blocker applies **only** when *no* registered kernel binds the new-opset node
**and** the op has no function body.

> **Convention (cross-EP consistency) — distinct from the binding-coverage rule above.** Even
> when an open-ended kernel already binds opset N (so it is *not* a blocker), still
> **version-split** it — and do so in **every** EP that registers the op (see the Group B all-EP
> tradition). PR #28754 split `Range` `[11,26]`+`27` in **both** CPU and CUDA even though CUDA's
> `SinceVersion(11)` kernel already bound opset 27, so the advertised boundary matches the schema
> and no EP lags behind CPU. Splitting is about **clarity/consistency**; binding-coverage is
> about **correctness** — keep the two concerns distinct.

**Why (the function-expansion mechanism):** an ONNX *function* op ships a reference
decomposition into primitive ops (`SetContextDependentFunctionBodyBuilder` /
`FunctionBodyHelper`). At graph-partition time ORT **inlines** (expands) any function-op node
that has no registered kernel into its primitive subgraph, which IS kernel-covered — so the op
runs with zero native code. A **non-function** op has no such fallback: with no binding kernel,
kernel binding fails at **session initialization** (fail-fast `kernel not found`), even though
the model loaded (schema came from the submodule — see §4 gotcha **g**).

> ⚠️ **Scope: runtime function-inlining is a FULL-build behavior.** Minimal / mobile /
> ORT-format builds do **not** runtime-inline function ops. The §9 gauntlet only proves the
> minimal build **compiles** (`--skip_tests`); it never **runs** a function-op model in a
> minimal build. So do **not** assume the function-inlining fallback gives minimal-build runtime
> coverage — a function-only op with no native kernel will not execute in a minimal build.

This is exactly why the 1.22 bump could defer native kernels for **`CausalConvWithState`**,
**`LinearAttention`**, and **`Range`-27 fp16/bf16** and still pass (in a full build): all three
are ONNX functions, so ORT inlined them. Decision rule per new op:

| A binding kernel covers the opset-N node? | Op has ONNX function body? | Action |
|---|---|---|
| Yes (incl. open-ended existing kernel) | — | Already covered; done |
| No | **Yes** | Safe to defer kernel — ORT inlines (full build); verify the `_expanded` test passes |
| No | **No** | **BLOCKER — write the kernel in this PR** (op would load then fail at session init) |

Quick check: grep the ONNX op's `defs.cc` for `SetContextDependentFunctionBodyBuilder` or
`FunctionBody`; if absent, it is not a function and needs a kernel.
