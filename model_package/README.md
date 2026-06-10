# Model Package Library

A standalone C library for **reading, authoring, and committing** ONNX
Runtime Model Packages. No dependency on ONNX Runtime, so any consumer
(ORT, ONNX Runtime GenAI, Foundry Local, external publisher tools) can
link against it.

## What it does

- **Open** a package directory and walk its components / variants /
  shared assets through a POD tree (`ModelPackage_Info()`).
- **Author** from scratch or in place via mutation calls
  (`SetComponentInline`, `SetVariant`, `SetVariantExecutorInfoExternal`,
  `AddSharedAsset`, etc.) and serialize with `ModelPackage_Commit()`.
- **Resolve** path / shared-asset references via
  `ModelPackage_ResolveStringRef()`. Accepts relative paths, absolute
  paths (installed layout only), `..` segments (installed only), bare
  `sha256:<hex>` asset URIs, and `sha256:<hex>/sub/path` forms.
- **Prune** stale orphan directories and **Validate** structural,
  reachability, path, and rehash invariants.

## What it deliberately does NOT do

- **Variant selection** — picking which variant best matches the
  available execution providers requires EP factory introspection and
  lives in the executor (ORT in particular).
- **Session creation** — building an `OrtSession` is ORT's job.
- **Interpreting `executor_info` payloads** — each consumer namespace
  (`ort`, `genai`, …) is opaque to this library.
- **Interpreting `compatibility_string`** — the format is owned by EPs.

## Building

```bash
cmake -B build -S . [-DMODEL_PACKAGE_BUILD_TESTS=ON]
cmake --build build -j
ctest --test-dir build --output-on-failure   # requires BUILD_TESTS=ON
```

CMake options:
- `MODEL_PACKAGE_BUILD_SHARED` (default `ON`) — shared vs static.
- `MODEL_PACKAGE_BUILD_TESTS`  (default `OFF`) — build the four
  unit-test executables (`test_asset_hashing`, `test_inspection`,
  `test_authoring`, `test_commit`).

## C API quick tour

All public entry points are declared in `include/model_package.h`. Open
a package and walk its info tree:

```c
#include "model_package.h"

ModelPackage* pkg = NULL;
ModelPackageStatus* st = ModelPackage_Open("/path/to/pkg", NULL, &pkg);
if (st) {
    fprintf(stderr, "open failed: %s\n", ModelPackageStatus_Message(st));
    ModelPackageStatus_Release(st);
    return 1;
}

const ModelPackageInfo* info = ModelPackage_Info(pkg);
for (size_t i = 0; i < info->num_components; ++i) {
    const ModelComponentInfo* c = &info->components[i];
    printf("component %s (%zu variants)\n", c->name, c->num_variants);
}

ModelPackage_Close(pkg);
```

Author a package from scratch:

```c
ModelPackage* pkg = NULL;
ModelPackage_New(&pkg);
ModelPackage_SetComponentInline(pkg, "encoder", "{\"variants\": {}}");
ModelPackage_SetVariant(pkg, "encoder", "v1",
                        "{\"ep\":\"CPU\",\"variant_directory\":\"encoder/v1\"}");
ModelPackage_SetVariantExecutorInfoInline(
    pkg, "encoder", "v1", "ort", "{\"model_file\":\"model.onnx\"}");
ModelPackage_Commit(pkg, "/path/to/new_pkg", MODEL_PACKAGE_WRITE_PRESERVE);
ModelPackage_Close(pkg);
```

### Lifetime contract

Every `const char*` and every `const ModelPackageInfo*` (plus
sub-arrays) returned by the read API is owned by the `ModelPackage`
handle and remains valid **until the next mutation of that scope** or
until `ModelPackage_Close()`. Any `Set*`/`Remove*`/`Add*`/`Commit` call
invalidates cached pointers in the mutated scope; re-read
`ModelPackage_Info()` after mutating.

`ModelPackage_AddSharedAsset` returns its `out_uri` under the same
"valid until next mutation" contract.

## Package format

A package is a directory rooted at `package_root/` containing
`manifest.json`. Components may be declared inline in the manifest or
externally as a sibling `component.json`/folder. Variants live under a
`variant_directory` (defaults to `<component_dir>/<variant_name>`),
which holds the model files plus any executor-specific configuration
referenced by `executor_info`. Shared, content-addressed asset
directories live under `shared_assets/sha256-<hex>/`.

```
package_root/
├── manifest.json
├── decoder/                       # external component
│   ├── component.json
│   └── cpu/                       # variant_directory
│       └── model.onnx
└── shared_assets/
    └── sha256-<64hex>/            # content-addressed asset
        └── ...
```

See `/datadisks/jambaykinley/archive/m/model_package_redesign.md` for
the full design rationale.

## ORT integration

ORT's `OrtModelPackageApi` (see `onnxruntime_c_api.h`) wraps this
library and adds variant selection plus `OrtSession` creation:
`CreateModelPackageOptionsFromSessionOptions` →
`OrtModelPackageApi::SelectComponent` →
`OrtModelPackageApi::CreateSession`.

The library itself never links against ORT.
