## c-api-artifacts-package-and-publish-steps-windows.yml
- Line 28: [`- name: trtEnabled`](https://github.com/microsoft/onnxruntime/blob/master/c-api-artifacts-package-and-publish-steps-windows.yml#L28)
- Line 29: [`displayName: Include TRT EP libraries?`](https://github.com/microsoft/onnxruntime/blob/master/c-api-artifacts-package-and-publish-steps-windows.yml#L29)
- Line 51: [`# copy trt ep libraries only when trt ep is enabled`](https://github.com/microsoft/onnxruntime/blob/master/c-api-artifacts-package-and-publish-steps-windows.yml#L51)
- Line 52: [`copy $(Build.BinariesDirectory)\${{parameters.buildConfig}}\${{parameters.buildConfig}}\onnxruntime_providers_tensorrt.dll $(Build.BinariesDirectory)\${{parameters.artifactName}}\lib`](https://github.com/microsoft/onnxruntime/blob/master/c-api-artifacts-package-and-publish-steps-windows.yml#L52)
- Line 53: [`copy $(Build.BinariesDirectory)\${{parameters.buildConfig}}\${{parameters.buildConfig}}\onnxruntime_providers_tensorrt.pdb $(Build.BinariesDirectory)\${{parameters.artifactName}}\lib`](https://github.com/microsoft/onnxruntime/blob/master/c-api-artifacts-package-and-publish-steps-windows.yml#L53)
- Line 54: [`copy $(Build.BinariesDirectory)\${{parameters.buildConfig}}\${{parameters.buildConfig}}\onnxruntime_providers_tensorrt.lib $(Build.BinariesDirectory)\${{parameters.artifactName}}\lib`](https://github.com/microsoft/onnxruntime/blob/master/c-api-artifacts-package-and-publish-steps-windows.yml#L54)
- Line 61: [`copy $(Build.SourcesDirectory)\include\onnxruntime\core\providers\tensorrt\tensorrt_provider_factory.h  $(Build.BinariesDirectory)\${{parameters.artifactName}}\include`](https://github.com/microsoft/onnxruntime/blob/master/c-api-artifacts-package-and-publish-steps-windows.yml#L61)

## linux-gpu-tensorrt-packaging-pipeline.yml
- Line 4: [`default: 'onnxruntime-linux-x64-gpu-tensorrt-$(OnnxRuntimeVersion)'`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L4)
- Line 8: [`default: 'onnxruntime-linux-x64-gpu-tensorrt'`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L8)
- Line 19: [`- stage: Linux_C_API_Packaging_GPU_TensorRT_x64`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L19)
- Line 36: [`Dockerfile: tools/ci_build/github/linux/docker/Dockerfile.manylinux2014_cuda11_8_tensorrt8_6`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L36)
- Line 39: [`Repository: onnxruntimecuda118xtrt86build`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L39)
- Line 47: [`--volume /data/models:/build/models:ro --volume $HOME/.onnx:/home/onnxruntimedev/.onnx -e NIGHTLY_BUILD onnxruntimecuda118xtrt86build \`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L47)
- Line 49: [`--skip_submodule_sync --parallel --build_shared_lib ${{ parameters.buildJavaOption }} --use_tensorrt --cuda_version=$(CUDA_VERSION) --cuda_home=/usr/local/cuda-$(CUDA_VERSION) --cudnn_home=/usr --tensorrt_home=/usr --cmake_extra_defines CMAKE_CUDA_HOST_COMPILER=/opt/rh/devtoolset-11/root/usr/bin/cc 'CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80'`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L49)
- Line 57: [`artifactName: 'onnxruntime-java-linux-x64-tensorrt'`](https://github.com/microsoft/onnxruntime/blob/master/linux-gpu-tensorrt-packaging-pipeline.yml#L57)

## py-linux-gpu.yml
- Line 26: [`Dockerfile: tools/ci_build/github/linux/docker/Dockerfile.manylinux2014_cuda11_8_tensorrt8_6`](https://github.com/microsoft/onnxruntime/blob/master/py-linux-gpu.yml#L26)
- Line 29: [`Repository: onnxruntimecuda118xtrt86build${{ parameters.arch }}`](https://github.com/microsoft/onnxruntime/blob/master/py-linux-gpu.yml#L29)
- Line 38: [`arguments: -i onnxruntimecuda118xtrt86build${{ parameters.arch }} -x "-d GPU"`](https://github.com/microsoft/onnxruntime/blob/master/py-linux-gpu.yml#L38)

## py-packaging-selectable-stage.yml
- Line 310: [`Dockerfile: tools/ci_build/github/linux/docker/Dockerfile.manylinux2014_cuda11_8_tensorrt8_6`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-selectable-stage.yml#L310)
- Line 313: [`Repository: onnxruntimecuda118xtrt86build`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-selectable-stage.yml#L313)
- Line 327: [`onnxruntimecuda118xtrt86build \`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-selectable-stage.yml#L327)
- Line 334: [`--enable_onnx_tests --use_tensorrt --use_tensorrt_builtin_parser --cuda_version=11.8 --tensorrt_home=/usr --cuda_home=/usr/local/cuda-11.8 --cudnn_home=/usr/local/cuda-11.8 \`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-selectable-stage.yml#L334)
- Line 364: [`--enable_onnx_tests --use_tensorrt --use_tensorrt_builtin_parser --cuda_version=11.4 --tensorrt_home=/usr --cuda_home=/usr/local/cuda-11.4 --cudnn_home=/usr/local/cuda-11.4  \`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-selectable-stage.yml#L364)
- Line 395: [`EpBuildFlags: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.0.12.Windows10.x86_64.cuda-11.8" --cuda_version=$(CUDA_VERSION) --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$(CUDA_VERSION)" --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=37;50;52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-selectable-stage.yml#L395)

## py-packaging-stage.yml
- Line 275: [`EP_BUILD_FLAGS: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.0.12.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-stage.yml#L275)
- Line 283: [`EP_BUILD_FLAGS: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.0.12.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-stage.yml#L283)
- Line 291: [`EP_BUILD_FLAGS: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.0.12.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-stage.yml#L291)
- Line 299: [`EP_BUILD_FLAGS: --use_tensorrt --tensorrt_home="C:\local\TensorRT-8.5.1.7.Windows10.x86_64.cuda-11.8.cudnn8.6" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/blob/master/py-packaging-stage.yml#L299)

