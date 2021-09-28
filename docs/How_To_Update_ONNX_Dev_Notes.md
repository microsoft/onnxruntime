This is a note only for ONNX Runtime developers.

It's very often, you need to update the ONNX submodule to a newer version in the upstream. Please follow the steps below, don't miss any!

1. Update the ONNX subfolder
```
cd cmake/external/onnx
git remote update
git reset --hard <commit_id>
cd ..
git add onnx
```
(Change the <commit_id> to yours. If you are not sure, use 'origin/master'. Like 'git reset --hard origin/master')

2. Update [cgmanifests/submodules/cgmanifest.json](/cgmanifests/submodules/cgmanifest.json).
This file should be generated. See [cgmanifests/README](/cgmanifests/README.md) for instructions.

3. Update [tools/ci_build/github/linux/docker/scripts/install_onnx.sh](/tools/ci_build/github/linux/docker/scripts/install_onnx.sh).
Search 'for version2tag', update the commit hashes. The list should contain every release version from ONNX 1.2, and the latest one in our cmake/external/onnx folder.

4. If there is any change to `cmake/external/onnx/onnx/*.in.proto`, you need to re-regenerate OnnxMl.cs. Please build onnxruntime on Windows with csharp enabled, then the file will be auto-updated.

6. If you are updating ONNX from a released tag to a new commit, please tell Changming deploying the new test data along with other test models to our CI build machines. This is to ensure that our tests cover every ONNX opset. 

5. Send you PR, and **manually** queue a build for every packaging pipeline for your branch.

6. If there is a build failure in stage "Check out of dated documents" in WebAssembly CI pipeline, update ONNX Runtime Web WebGL operator support document:
   - Make sure Node.js is installed (see [Prerequisites](../js/README.md#Prerequisites) for instructions).
   - Follow step 1 in [js/Build](../js/README.md#Build-2) to install dependencies).
   - Follow instructions in [Generate document](../js/README.md#Generating-Document) to update document. Commit changes applied to file `docs/operators.md`.

7. Usually there would be some unitest failures, because you introduced new test cases. Then you may need to update
- [onnxruntime/test/onnx/main.cc](/onnxruntime/test/onnx/main.cc)
- [onnxruntime/test/providers/cpu/model_tests.cc](/onnxruntime/test/providers/cpu/model_tests.cc)
- [csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs](/csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs)
- [onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc](/onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc)
