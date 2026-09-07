---
name: ort-release-notes
description: Draft ONNX Runtime release notes using preset configurations for full ORT or scoped component releases. Use when generating highlights with PR links, compiling human contributor acknowledgments from compile_contributors.py output, and applying preset path filtering.
argument-hint: "preset base_ref target_ref [version] [output_dir]"
---

# ONNX Runtime Release Notes

Use this skill to produce a consistent release-note draft from commit history and contributor metadata.

## When To Use

Use this skill when you need to:

- Draft release notes for a full ONNX Runtime release
- Draft release notes for a scoped component (e.g., in-tree plugin EP) release
- Select a release profile by preset name instead of manually supplying path/version files
- Add PR links to highlight bullets
- Build a human-only contributor acknowledgment list from contributor metadata

## Required Inputs

Collect these inputs from the user or infer from context:

1. `preset`: release profile name (for example, `ort`, `webgpu-plugin-ep`, `cuda-plugin-ep`)
2. `base_ref`: previous release tag
3. `target_ref`: release commit/tag/branch tip

Optional inputs:

- `version` override
- `output_dir` override

## Presets

Read preset definitions from [presets.json](./presets.json).

The config defines shared output defaults:

1. `outputDirPattern`
2. `draftFileName`

Each preset defines:

1. `displayName`: reader-facing product or component name
2. `versionFile`
3. `pathsFile` (nullable): file of git pathspecs to filter to, one per line. Use `:(top)` to anchor an entry at repo root.

Example presets:

1. `ort` (full ONNX Runtime)
2. `webgpu-plugin-ep` (scoped WebGPU Plugin EP)
3. `cuda-plugin-ep` (scoped CUDA Plugin EP)

### CUDA Plugin EP Scope

The `cuda-plugin-ep` preset uses the pathspecs in `plugin-ep-cuda/paths.txt` to scope release-note changes.

## Workflow

1. Determine release mode.
   - Select preset and load configuration from [presets.json](./presets.json).
   - Use the preset's `displayName` whenever the release-note content names the product or component. The preset
     key is internal and must not appear in published content.
   - If preset has `pathsFile`, run in scoped mode. Otherwise run full mode.
2. Resolve version, in this order:
   1. explicit `version` input
   2. value from preset `versionFile`
3. Resolve output directory, referred to as `resolved_output_dir` after this step.
   The output directory contains contributor artifacts and the release notes draft.
   Resolve it in this order:
   1. explicit `output_dir` input
   2. shared `outputDirPattern` rendered with preset name and resolved version
4. Gather metadata.
   - If the output directory is missing or lacks contributor artifacts, generate them with
     `tools/python/compile_contributors.py`.
   - Generation can take a while because it scans commit history and fetches PR metadata.
   - Use `--paths-file` only when preset has a `pathsFile`.
   - If existing contributor artifacts are reused, verify `resolved_output_dir/logs.txt` matches base/target before trusting them.
5. Read `resolved_output_dir/detail.csv` as the primary source for PR numbers, titles, authors, target commits, and cherry-pick mapping.
   - Use `resolved_output_dir/logs.txt` for contributor summary context and base/target verification.
   - Use `git log` only as a fallback sanity check when artifacts are present but incomplete or suspect.
    - Check PRs with unexpectedly large author lists for rebased history that imported unrelated commits. Replace those
       authors with the actual PR author or authors before building contributor acknowledgments; do not credit authors
       solely because they authored an unrelated imported commit. For example, PR #28299, the rebased history contains unrelated commits and co-author metadata.
6. Build highlight categories.
   - Full ORT example categories: performance, model/operator support, execution providers, API/languages,
     reliability/security, build/packaging/tooling, docs/dev workflow.
   - Scoped mode: narrow categories to the component domain.
7. Draft markdown.
   - Write the release-note draft to `resolved_output_dir/<draftFileName>`.
   - Contents:
     - Intro sentence
     - `## Highlights`
     - Inline PR links on every highlight bullet
     - `## Contributors`
     - Optional scope note for scoped-component releases
       - Use the preset's reader-facing `displayName`, not the internal preset key.
       - Describe the scope in reader-facing terms, such as "commits affecting WebGPU Plugin EP code and
         packaging."
     - AI disclaimer if AI drafted
   - Do not mention presets, `pathsFile`, configuration files, or other release-note-generation implementation
     details in the release-note content.
   - Do not refer to the release notes as a "draft" in their content. "Draft" is only an internal workflow
     and file-naming concept.
8. Build contributors section.
   - Start from `detail.csv` output
   - Include humans only
   - Exclude bots/agents (for example: `github-actions[bot]`, `app/copilot-swe-agent`, `claude`)
   - Sort alphabetically
9. Validate draft quality.
   - Every highlight bullet has at least one PR link
   - PRs are traceable to metadata or git history
   - Contributor list is human-only and alphabetical
   - Scope is correct for full vs component release

## PowerShell Command Patterns

### compile_contributors.py

Full ORT metadata:

```powershell
python .\tools\python\compile_contributors.py \
  --base <previous_tag> \
  --target <target_ref> \
  --dir <resolved_output_dir>
```

Scoped metadata:

```powershell
python .\tools\python\compile_contributors.py \
   --base <previous_tag> \
   --target <target_ref> \
   --dir <resolved_output_dir> \
   --paths-file <paths_file>
```

## Style and Policy

Default policy unless release owners override:

1. Treat the range as changes since the previous release.
2. Keep PR links inline with highlight claims.
3. Keep contributor acknowledgments human-only and best effort.
4. Include an AI disclaimer when highlights are AI drafted.
5. Use GitHub Releases pages as preferred style references.
   E.g., [ORT 1.28 release page](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0).
6. Prefer preset-driven configuration over ad-hoc path/version arguments.
7. Use a single `output_dir` for contributor artifacts, logs, and the release-note draft.
