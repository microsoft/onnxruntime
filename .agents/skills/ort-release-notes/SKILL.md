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

1. `preset`: release profile name (for example, `ort`, `webgpu-plugin-ep`)
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

1. `versionFile`
2. `pathsFile` (nullable)

Example presets:

1. `ort` (full ONNX Runtime)
2. `webgpu-plugin-ep` (scoped WebGPU Plugin EP)

## Workflow

1. Determine release mode.
   - Select preset and load configuration from [presets.json](./presets.json).
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
   - Use `--paths` only when preset has a `pathsFile`.
   - If existing contributor artifacts are reused, verify `resolved_output_dir/logs.txt` matches base/target before trusting them.
5. Read `resolved_output_dir/detail.csv` as the primary source for PR numbers, titles, authors, target commits, and cherry-pick mapping.
   - Use `resolved_output_dir/logs.txt` for contributor summary context and base/target verification.
   - Use `git log` only as a fallback sanity check when artifacts are present but incomplete or suspect.
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
     - Optional AI disclaimer if AI drafted highlights
     - `## Contributors`
     - Optional scope note for scoped-component releases
       - Include the component/preset name.
       - Include a brief scope statement that commits were filtered by the preset `pathsFile`.
8. Build contributors section.
   - Start from `detail.csv` output
   - Include humans only
   - Exclude bots/agents (for example: `github-actions[bot]`, `app/copilot-swe-agent`, `claude`)
   - Sort alphabetically
9. Validate draft quality.
   - Every bullet has at least one PR link
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
$pathList = Get-Content <paths_file> | Where-Object { $_ -and $_.Trim() }

python .\tools\python\compile_contributors.py \
   --base <previous_tag> \
   --target <target_ref> \
   --dir <resolved_output_dir> \
   --paths $pathList
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
