# CGManifest Files
This directory contains CGManifest (cgmanifest.json) files.
See [here](https://docs.opensource.microsoft.com/tools/cg/cgmanifest.html) for details.

The WebGPU-specific manifest is in `webgpu/cgmanifest.webgpu.json`. It is intentionally not named `cgmanifest.json`
so default whole-repository Component Governance scans do not pick it up automatically. WebGPU packaging or
NOTICE-generation pipelines should stage it as `cgmanifest.json` in their scan input.
