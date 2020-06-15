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

4. If there is any change to `cmake/external/onnx/onnx/*.in.proto`, update onnxruntime/core/protobuf as follows : 
```
- Apply these changes to onnxruntime/core/protobuf/*.in.proto
- Copy cmake/external/onnx/onnx/gen_proto.py to onnxruntime/core/protobuf and use this script to generate the new \*.proto and \*.proto3 files
- Regenerate csharp/test/Microsoft.ML.OnnxRuntime.Tests/OnnxMl.cs
```

5. Send you PR, and run the CI builds.

6. If there is any unitest failure, caught by onnx_test_runner. Please also update
- [onnxruntime/test/onnx/main.cc](/onnxruntime/test/onnx/main.cc)
- [onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc](/onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc)
