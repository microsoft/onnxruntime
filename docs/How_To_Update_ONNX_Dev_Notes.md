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

2. Update [cgmanifest.json](/cgmanifest.json)        
Search 'https://github.com/onnx/onnx.git', update the commitHash with it.

3. Update [tools/ci_build/github/linux/docker/scripts/install_deps.sh](/tools/ci_build/github/linux/docker/scripts/install_deps.sh) 
and [tools/ci_build/github/linux/docker/scripts/install_deps_x86.sh](/tools/ci_build/github/linux/docker/scripts/install_deps_x86.sh) 
Search 'for onnx_version in', update the commit hashes. The list should contain every release version from ONNX 1.2, and the latest one in our cmake/external/onnx folder.

4. Update onnxruntime/core/protobuf        
If there is any change on `cmake/external/onnx/onnx/*.in.proto` since the last sync, please apply these changes to `onnxruntime/core/protobuf/*.in.proto`

Then copy cmake/external/onnx/onnx/gen_proto.py to onnxruntime/core/protobuf, use that script generating the new \*.proto and \*.proto3 files.

Regenerate csharp/test/Microsoft.ML.OnnxRuntime.Tests/OnnxMl.cs

5. Send you PR, and run the CI builds.        

6. If there is any unitest failure, caught by onnx_test_runner. Please also update        
- [onnxruntime/test/onnx/main.cc](/onnxruntime/test/onnx/main.cc)
- [onnxruntime/test/python/onnx_backend_test_series.py](/onnxruntime/test/python/onnx_backend_test_series.py)









