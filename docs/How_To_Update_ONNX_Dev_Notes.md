# How to update ONNX

This note is only for ONNX Runtime developers.

If you need to update the ONNX submodule to a different version, follow the steps below.

1. Update the ONNX submodule
```sh
cd cmake/external/onnx
git remote update
git reset --hard <commit_id>
cd ..
git add onnx
```
(Change the <commit_id> to yours. If you are not sure, use 'origin/master'. Like 'git reset --hard origin/master')

1. Update [cgmanifests/generated/cgmanifest.json](/cgmanifests/generated/cgmanifest.json).
This file should be generated. See [cgmanifests/README](/cgmanifests/README.md) for instructions.

1. Update Python requirements files with the updated ONNX version (e.g., `onnx==1.16.0`) or commit hash if building from source (e.g., `git+http://github.com/onnx/onnx.git@targetonnxcommithash#egg=onnx`).
- [onnxruntime/test/python/requirements.txt](/onnxruntime/test/python/requirements.txt)
- [tools/ci_build/github/linux/docker/scripts/requirements.txt](/tools/ci_build/github/linux/docker/scripts/requirements.txt)
- [tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt](/tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt)
- [tools/ci_build/github/linux/python/requirements.txt](/tools/ci_build/github/linux/python/requirements.txt)
- Run `git grep -rn "onnx==1" .` to find other locations and update this document if necessary.

1. If there is any change to `cmake/external/onnx/onnx/*.in.proto`, you need to regenerate OnnxMl.cs.
   [Building onnxruntime with Nuget](https://onnxruntime.ai/docs/build/inferencing.html#build-nuget-packages) will do
   this.

1. If you are updating ONNX from a released tag to a new commit, please ask Changming (@snnn) to deploy the new test
   data along with other test models to our CI build machines. This is to ensure that our tests cover every ONNX opset.

1. Send your PR, and **manually** queue a build for every packaging pipeline for your branch.

1. If there is a build failure in stage "Check out of dated documents" in WebAssembly CI pipeline, update ONNX Runtime
   Web WebGL operator support document:
   - Make sure Node.js is installed (see [Prerequisites](../js/README.md#Prerequisites) for instructions).
   - Follow step 1 in [js/Build](../js/README.md#Build-2) to install dependencies).
   - Follow instructions in [Generate document](../js/README.md#Generating-Document) to update document. Commit changes applied to file `docs/operators.md`.

1. Usually some newly introduced tests will fail. Then you may need to update
- [onnxruntime/test/onnx/main.cc](/onnxruntime/test/onnx/main.cc)
- [onnxruntime/test/providers/cpu/model_tests.cc](/onnxruntime/test/providers/cpu/model_tests.cc)
- [csharp/test/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp/InferenceTest.netcore.cs](/csharp/test/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp/InferenceTest.netcore.cs)
- [onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc](/onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc)
- [onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc](/onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc)

1. If an operator has changed we may need to update optimizers involving that operator.
- Run [find_optimizer_opset_version_updates_required.py](/tools/python/find_optimizer_opset_version_updates_required.py), compare with the output from the current main branch, and check for any new warnings.
- If there are new warnings contact the optimizer owner (which can usually be determined by looking at who edited the file most recently) or failing that ask the 'ONNX Runtime Shared Core' mailing list.
