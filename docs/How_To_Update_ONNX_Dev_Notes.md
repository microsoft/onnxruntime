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

3. Update the commit id of onnx for building the linux image for CI pipelines
- [manylinux image requirements](onnxruntime/tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt)
- [all other linux image requirements](onnxruntime/tools/ci_build/github/linux/docker/scripts/requirements.txt)

4. Send PR and run CI builds

5. If there are any test failures because of the new ops that are coming in as part of the onnx commit bump, update test filters to filter tests for these ops. These tests need to be added back after ort implements these ops.
- c++ test filters [onnxruntime/test/onnx/test_filters.h](onnxruntime/test/onnx/test_filters.h)
- python andf nodejs test filters [onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc](/onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc)
