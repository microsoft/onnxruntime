# Investigating WebGPU shader cache collisions

Set `ORT_WEBGPU_EP_VALIDATE_SHADER_CACHE=1` before creating the first WebGPU
session in a process. This enables a diagnostic that regenerates the expected
shader on cache hits and rejects reuse when the same key describes different
WGSL, pipeline override constants, binding layouts, or host uniform layouts.
The existing shader APIs and cache keys are unchanged.

```powershell
$env:ORT_WEBGPU_EP_VALIDATE_SHADER_CACHE = '1'
python reproduce.py
Remove-Item Env:ORT_WEBGPU_EP_VALIDATE_SHADER_CACHE
```

The error includes the key, both program names, the first differing description
line, and the complete previous and current descriptions. Compare those sections
to identify the specialization missing from the key. Runtime tensor contents,
uniform values, and dynamic shape/offset values are not compared; their layouts
are. A textual difference is treated conservatively as a conflict, even if two
shaders might be mathematically equivalent.

Descriptions are recorded before asynchronous compilation and checked before
both pending-build and completed-cache reuse. Concurrent sessions sharing a
program manager share synchronized observations. The first observation is kept
until that manager is destroyed, including after a failed compilation. Separate
devices/managers do not share observations. Graph capture is checked while its
programs are recorded; replay does not regenerate shaders.

The diagnostic is off by default, independently of `validationMode`. When off,
cache hits do not regenerate shaders and no description table is allocated.
When on, every program invocation generates WGSL on the CPU and compares a
description; cache hits do not compile another GPU pipeline. The table retains
one complete description per observed key. Use this for investigations or
selected correctness runs, not performance measurement.

Detection requires exercising both conflicting variants in the same manager's
lifetime. This is not proof that all keys are complete, and it does not determine
which of two conflicting shaders is correct. The existing cache regression tests
can be run with the diagnostic enabled; no additional operator variant matrix is
required.
