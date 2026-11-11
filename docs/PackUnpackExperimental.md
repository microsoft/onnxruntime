# Experimental Pack and Unpack

This branch implements CPU kernels for the standard-domain `Pack` and `Unpack`
operators proposed in [onnx/onnx#8576](https://github.com/onnx/onnx/pull/8576).
It is an early-validation implementation, **not official support for opset 29**.
The operator definitions, encodings, and opset assignment may change before release.
No backward-compatibility guarantee is made for experimental models.

## ONNX dependency

Schemas, type/shape inference, function decompositions, and Python reference
implementations come directly from
[`gramalingam/onnx`, branch `rama/unpack`](https://github.com/gramalingam/onnx/tree/rama/unpack),
pinned at `dab257d2bc07e415d6afeff7726a6d139a593a15`.
There are no duplicated Pack/Unpack schemas in ORT.

The FetchContent archive, ONNX submodule, vcpkg port, and Python test requirements
all use that revision. Existing ORT integration patches are rebased onto it;
they do not modify Pack/Unpack. The proposal's ONNX Python package requires
Python 3.11 or newer and is built from source, not downloaded as a release wheel.
For node-test materialization, CMake checks both the version and pip's recorded
VCS commit to prevent accidentally using a release wheel or a different proposal
revision. Install ONNX using the pinned Git requirement, rather than an editable
local checkout, for that check.

## Building and trying models

Use the normal ORT build instructions. For example, on Windows with the C++
build tools installed, from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_ARGS = "-DONNX_BUILD_CUSTOM_PROTOBUF=ON"
python -m pip install -r onnxruntime\test\python\requirements.txt
Remove-Item Env:\CMAKE_ARGS
python tools\ci_build\build.py --config Release --build_wheel --parallel `
  --skip_submodule_sync --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
```

Install the generated ORT wheel into the virtual environment before running
models. For C++ operator-only validation, a Python ONNX installation is not
required:

```powershell
python tools\ci_build\build.py --config Release --cmake_generator Ninja `
  --update --build --target onnxruntime_provider_test --parallel `
  --skip_submodule_sync --skip_onnx_tests `
  --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
Push-Location build\Windows\Release
.\onnxruntime_provider_test.exe --gtest_filter="PackUnpackTest.*"
Pop-Location
```

The Ninja example requires a shell configured with the Visual Studio C++ tools.
Visual Studio generators put executables in an additional `Release` subdirectory.
Ninja also avoids Visual Studio path-length failures while pip builds ONNX.
Building the pinned protobuf sources prevents unrelated installed protobuf
packages from being selected. If supplying an external `protoc`, its version
must match the protobuf library used for that build; the ONNX Python build and
ORT build can use different protobuf versions in their separate binaries.

Create nodes in the default ONNX domain (`""`) with an opset-29 import, using
the proposal's ONNX package. To run them, explicitly allow unreleased opsets
for that session:

```python
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

graph = helper.make_graph(
    [helper.make_node("Pack", ["X"], ["Y"], radix=3, group_size=5)],
    "ternary_pack",
    [helper.make_tensor_value_info("X", TensorProto.UINT8, [5])],
    [helper.make_tensor_value_info("Y", TensorProto.UINT8, [1])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)])
options = ort.SessionOptions()
options.add_session_config_entry("session.allow_released_opsets_only", "0")
session = ort.InferenceSession(
    model.SerializeToString(), options, providers=["CPUExecutionProvider"]
)
result = session.run(None, {"X": np.array([0, 1, 2, 0, 1], dtype=np.uint8)})
assert result[0].tolist() == [102]
```

The default released-opset restriction is unchanged. This branch does not raise
other EPs' supported-opset ceilings or claim general opset-29 coverage.

## Kernel coverage

Both kernels support `bits=1..8` and the mutually exclusive
`radix`/`group_size` spelling, including group indices up to 63 bits,
UINT8 and UINT32 storage, arbitrary positive/negative axes, partial groups,
cross-word encodings, zero dimensions, and dynamic scalar INT64 `Unpack.count`.
The kernels check malformed attributes, out-of-range input codes, invalid group
indices, and mismatched packed lengths. They ignore unused final-word bits and
unused final-group digits, as specified by the proposal.

Provider tests compare both operators independently against a bit-at-a-time
encoding oracle and fixed proposal examples; they also check invalid inputs
and assert that a single native node remains rather than a function expansion.
GPU kernels and quantization-pattern fusions are not implemented here.

## Synchronizing with the proposal

1. Get the new head of `onnx/onnx#8576` and update `cmake/external/onnx`
   to that exact commit. The submodule's configured tracking branch is
   `rama/unpack`; builds still use an immutable commit.
2. Update the commit URL and archive SHA1 in `cmake/deps.txt`, its
   `ONNX_VERSION` comment if needed, and the vcpkg port's commit/SHA512
   (computed from the GitHub `.tar.gz`, not the `.zip`). Update the vcpkg
   version if ONNX's `VERSION_NUMBER` changes.
3. Update the pinned ONNX Git requirement in the seven CI/test
   `requirements.txt` files and `requirements-materialize-onnx-node-tests.txt`.
   Leave intentionally frozen transformer-model requirements unchanged.
4. Verify/rebase existing `cmake/patches/onnx/onnx.patch` integration hunks
   and vcpkg patches. Keep `binskim.patch` byte-identical to `onnx.patch`.
5. Rebuild, run `PackUnpackTest.*`, and compare against the updated ONNX
   reference cases. Update the experimental kernel registration version if
   the proposal changes it. Revalidate node-test materialization against
   the matching source-built Python ONNX package.

Internal vcpkg builds with origin downloads disabled additionally require the
new archive to be uploaded to their artifact mirror; the FetchContent build
does not depend on that mirror.
