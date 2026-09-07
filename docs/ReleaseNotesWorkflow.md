# ONNX Runtime Release Notes Workflow

ONNX Runtime release notes are drafted using the `ort-release-notes` agent skill:

- [`.agents/skills/ort-release-notes/SKILL.md`](../.agents/skills/ort-release-notes/SKILL.md)

The skill is the source of truth for the release-note workflow, including inputs, artifact discovery, contributor
handling, draft structure, validation, and PowerShell command patterns. Keep procedural details there so this document
does not drift out of sync.

## Supported Presets

Preset definitions live in:

- [`.agents/skills/ort-release-notes/presets.json`](../.agents/skills/ort-release-notes/presets.json)

For example, there is a preset for the core ONNX Runtime release and one for the WebGPU plugin EP release.

Refer to that file for the available preset names and their version/path configuration.

## Maintainer Usage

Ask your favorite AI agent to draft release notes using a preset name, base ref, and target ref. For example:

```text
Draft release notes using the ort-release-notes skill for preset webgpu-plugin-ep from
plugin-ep-webgpu/v0.1.0 to <target-ref>.
```

## Adding A Preset

To add release-note support for another component, add a preset to
[`presets.json`](../.agents/skills/ort-release-notes/presets.json). Component presets should define a version file and,
when the release should be scoped, a path filter file similar to
[`plugin-ep-webgpu/paths.txt`](../plugin-ep-webgpu/paths.txt).
