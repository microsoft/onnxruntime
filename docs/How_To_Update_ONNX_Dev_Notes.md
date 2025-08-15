# How to update ONNX

This note is only for ONNX Runtime developers.

If you need to update the ONNX submodule to a different version, follow the steps below.

## Update ONNX installation

Currently, ONNXRUNTIME supports two ways to install ONNX cpp dependencies, one is through cmake/deps.txt, and the other one is by vcpkg. And both of them are guarded by CI. It is recommended to test vcpkg within Windows machines.

### Update the ONNX submodule (commit would be more precise than branch)

```sh
cd cmake/external/onnx
git remote update
git reset --hard <commit_id>
cd ..
git add onnx
```

(Change the <commit_id> to yours. If you are not sure, use 'origin/main'. Like 'git reset --hard origin/main')

### Update cmake/deps.txt

1. Update [cmake/deps.txt](/cmake/deps.txt) with the correct zip download link and SHA (alternatively, build it with the wrong SHA and ORT should tell you the expected one.).
2. Check [cmake/patches/onnx/onnx.patch](/cmake/patches/onnx/onnx.patch) to see whether the diffs are resolved in the latest ONNX version.
3. Try to build ONNXRUNTIME from source. If the build fails, please make the changes accordingly, or use onnx.patch if it's ONNX bugs. An example build:

```bash
./build.sh --config RelWithDebInfo --use_cuda --cuda_home /usr/local/cuda-12.6/ --cudnn_home /usr/local/cuda-12.6/ --build_wheel --parallel --skip_tests
```

### Update cmake/vcpkg-ports

1. Modify [cmake/vcpkg-ports/onnx/binskim.patch](/cmake/vcpkg-ports/onnx/binskim.patch) to be the same as [cmake/patches/onnx/onnx.patch](/cmake/patches/onnx/onnx.patch).
2. The other patches are required/created by vcpkg repository to build ONNX. We just need to re-run diff to makes sure the patches can be applied in the updated ONNX version.
3. Update [cmake/vcpkg-ports/onnx/portfile.cmake](/cmake/vcpkg-ports/onnx/portfile.cmake) with the correct commit id and SHA512. (alternatively, build it with the wrong SHA and ORT should tell you the expected one.)
4. Upload your package: [Follow the instructions](https://microsoft.sharepoint.com/teams/ONNX2/_layouts/15/Doc.aspx?sourcedoc={170774be-e1c6-4f8b-a3ae-984f211fe410}&action=edit&wd=target%28Development.)one%7C63d3ab47-51d1-4a62-9965-66882234bd44%2FAdd%20or%20Update%20a%20C%2B%2B%20dependency%7Cb6ae6a97-94fc-4436-8fc6-08c21ae895da%2F%29&wdorigin=NavigationUrl

Alternatively, directly run Terrapin to upload ONNX package (need SHA512):

```
C:\local\Terrapin\TerrapinRetrievalTool.exe -b https://vcpkg.storage.devpackages.microsoft.io/artifacts/ -a true -u Environment -p https://github.com/onnx/onnx/archive/866d145a8168e36f34741534df4c94d8e7588379.tar.gz -s 80e3a5db347bf3046a6fb8721f308c054c923aa5649ab704f0265e83bba6edac9f7e402dde2be3d9c01d3b291f5eb284eaba2df6986744161ece887e0c2fc845 -d "C:\Users\titaiwang\onnxruntime\build\Windows\vcpkg\downloads\onnx-onnx-866d145a8168e36f34741534df4c94d8e7588379.tar.gz.25136.part"
```

5. Try to build ONNXRUNTIME from source. If the build fails, please make the changes accordingly, or use binskim.patch if it's ONNX bugs. An example build:

```bash
./build.sh --config RelWithDebInfo --use_cuda --cuda_home /usr/local/cuda-12.6/ --cudnn_home /usr/local/cuda-12.6/ --build_wheel --parallel --skip_tests --use_vcpkg
```

## Update ONNX related documentations

We need to update the auto-generated ONNX kernels markdowns and requirements.txt.

### AUTO-generated ONNX kernels

We can either use the following command lines to generate them, or go to CI (AzureDevOps published Artifacts) to download and upload the generated markdowns from CI to update them (suggested).

If you want to do the command lines:

1. Update [docs/OperatorKernels.md](/docs/OperatorKernels.md)

```bash
# under onnxruntime root
python tools/python/gen_opkernel_doc.py --output_path docs/OperatorKernels.md
```

1. Update [js/web/docs/webgl-operators.md](/js/web/docs/webgl-operators.md) with the script: [generate-webgl-operator-md.ts](/js/web/script/generate-webgl-operator-md.ts)

```bash
# Install dependencies at the parent JS level
cd /path/to/onnxruntime/js
npm install
# Install dependencies at the web level
cd /path/to/onnxruntime/js/web
npm install
# Build the doc
npm run build:doc
```

### Update requirements.txt

Update Python requirements files with the updated ONNX version (e.g., `onnx==1.16.0`) or commit hash if building from source (e.g., `git+http://github.com/onnx/onnx.git@targetonnxcommithash#egg=onnx`).

- [onnxruntime/test/python/requirements.txt](/onnxruntime/test/python/requirements.txt)
- [tools/ci_build/github/linux/docker/scripts/requirements.txt](/tools/ci_build/github/linux/docker/scripts/requirements.txt)
- [tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt](/tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt)
- [tools/ci_build/github/linux/python/requirements.txt](/tools/ci_build/github/linux/python/requirements.txt)
- Run `git grep -rn "onnx==1" .` to find other locations and update this document if necessary.

## Additional Notes

1. If there is any change to `cmake/external/onnx/onnx/*.in.proto`, you need to regenerate OnnxMl.cs.
   [Building onnxruntime with Nuget](https://onnxruntime.ai/docs/build/inferencing.html#build-nuget-packages) will do
   this.
2. If you are updating ONNX from a released tag to a new commit, please ask Changming (@snnn) to deploy the new test
   data along with other test models to our CI build machines. This is to ensure that our tests cover every ONNX opset.
3. Send your PR, and **manually** queue a build for every packaging pipeline for your branch.
4. If there is a build failure in stage "Check out of dated documents" in WebAssembly CI pipeline, update ONNX Runtime
   Web WebGL operator support document:

   - Make sure Node.js is installed (see [Prerequisites](../js/README.md#Prerequisites) for instructions).
   - Follow [js/Build](../js/README.md#Build-2) to install dependencies.
   - Follow instructions in [Generate document](../js/README.md#Generating-Document) to update document. Commit changes applied to file `docs/operators.md`.
5. Usually some newly introduced tests will fail. Then you may need to update

- [onnxruntime/test/onnx/main.cc](/onnxruntime/test/onnx/main.cc)
- [onnxruntime/test/providers/cpu/model_tests.cc](/onnxruntime/test/providers/cpu/model_tests.cc)
- [csharp/test/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp/InferenceTest.netcore.cs](/csharp/test/Microsoft.ML.OnnxRuntime.Tests.NetCoreApp/InferenceTest.netcore.cs)
- [onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc](/onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc)
- [onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc](/onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc)

1. If an operator has changed we may need to update optimizers involving that operator.

- Run [find_optimizer_opset_version_updates_required.py](/tools/python/find_optimizer_opset_version_updates_required.py), compare with the output from the current main branch, and check for any new warnings.
- If there are new warnings contact the optimizer owner (which can usually be determined by looking at who edited the file most recently) or failing that ask the 'ONNX Runtime Shared Core' mailing list.
