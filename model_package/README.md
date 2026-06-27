# Model Package Library

A standalone C library for parsing and inspecting ONNX Runtime Model Packages.

**No dependency on ONNX Runtime.** This library can be consumed independently by any component (ORT, GenAI, FL, or external tools).

## What it does

- Parses model package directory structures (`manifest.json`, `metadata.json`, `variant.json`)
- Provides read-only access to:
  - Components and their variants
  - EP compatibility declarations (opaque strings)
  - Model file paths within variants
  - Session/provider options per file
  - Consumer metadata (opaque JSON)

## What it does NOT do

- Variant selection (requires runtime EP factory validation → stays in ORT)
- Session creation (requires ORT `InferenceSession`)
- Any interpretation of `compatibility_string` tokens

## Building

```bash
cmake -B build -S .
cmake --build build
```

Options:
- `-DMODEL_PACKAGE_BUILD_SHARED=ON|OFF` — Build as shared (default) or static library
- `-DMODEL_PACKAGE_BUILD_TESTS=ON` — Build tests (default OFF)

## C API Usage

```c
#include "model_package_api.h"

ModelPackageContext* ctx = NULL;
ModelPackageStatus* status = ModelPackage_CreateContext("/path/to/package", &ctx);
if (status != NULL) {
    printf("Error: %s\n", ModelPackage_GetErrorMessage(status));
    ModelPackage_ReleaseStatus(status);
    return;
}

size_t count = 0;
ModelPackage_GetComponentCount(ctx, &count);

for (size_t i = 0; i < count; i++) {
    const char* name = NULL;
    ModelPackage_GetComponentName(ctx, i, &name);
    printf("Component: %s\n", name);
}

ModelPackage_ReleaseContext(ctx);
```

## Integration with ORT

ORT compiles this library as part of its build and wraps the C API through `OrtModelPackageApi`, adding:
- Variant selection via EP factory compatibility validation
- Session creation with merged options

## Package Format

```
package_root/
├── manifest.json              # schema_version, components list
└── models/
    └── <component_name>/
        ├── metadata.json      # variants + EP compatibility declarations
        └── <variant_name>/
            ├── variant.json   # files list, consumer_metadata
            └── model.onnx     # (or other model files)
```

Single-component shorthand (metadata.json at root, no manifest.json) is also supported.
