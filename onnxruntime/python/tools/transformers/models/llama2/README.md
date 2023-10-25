# Llama2 inference

## Prerequisites

Execute `pip install -r requirements.txt` to install neccessary packages.

For ORT installation, to include 'AllReduce' op, please add `--use_mpi --enable_nccl --nccl_home ${NCCL_HOME}` into onnxruntime build flags.

Sample installation commands:

```bash
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
# tmp branch
git checkout linmin/llama2_bk
bash build.sh --use_cuda --cuda_version=11.7 --config=Release \
    --build_wheel --update --build --cmake_generator Ninja --skip_tests --mpi_home=/usr/local/mpi \
    --cuda_home=/usr/local/cuda --cudnn_home=/usr/lib/x86_64-linux-gnu --enable_nccl \
    --use_mpi --enable_cuda_profiling

pip install build/Release/dist/*.whl
```

In ROCm environment the build command is like

```bash
bash ./build.sh \
    --config Release --enable_training --build_wheel --update --build --cmake_generator Ninja --skip_tests \
    --use_rocm --rocm_version=5.6.0 --rocm_home /opt/rocm --enable_nccl \
    --nccl_home /opt/rocm --allow_running_as_root --use_mpi --mpi_home /usr/lib/x86_64-linux-gnu/openmpi \
    --enable_rocm_profiling \
    --cmake_extra_defines onnxruntime_USE_COMPOSABLE_KERNEL=ON \
                          onnxruntime_USE_ROCBLAS_EXTENSION_API=ON \
                          onnxruntime_USE_HIPBLASLT=ON \
                          onnxruntime_USE_TRITON_KERNEL=OFF \
                          onnxruntime_BUILD_KERNEL_EXPLORER=ON
```

## Running

Please use `bash sample_run.sh` to run the test. The command is like

```bash
bash sample_run.sh <NUM_GPUS> [extra_args]
```

Among the `extra_args`, the following are supported:

- `--export`: export the torch model into onnx model, when running with distributed mode, multiple onnx model files suffixed with rank id will be exported.
- `--optimize`: optimize the onnx model with onnxruntime optimizer. Also needed when running the optimized model.
- `--merge`: merge decoder and decoder_with_past into a single model. Also needed when running a merged model.
- `--generate`: run a single prompt generation test, add `--torch` or `--ort` to run with torch or ort.
- `--benchmark`: run benchmark for the model with `--torch` or `--ort`.
- `--provider [ep]`: run benchmark/test with specified execution provider, e.g. `--provider rocm` (default) or `--provider cuda`.

For example, the following command will export the model (distributed for 8 GPUs), merge decoders, then run sample generation test, and finally run benchmark, both with ort.

```bash
bash sample_run.sh 8 --export --generate --ort --benchmark --merge
```

Some notes for benchmarks in MI250X:

- Options `--tunable` and `--tuning` are used to enable ORT tuning (including GEMM, LayerNorm, etc.), which will significantly improve the performance. Note it will take some time in the tuning process.
- Option `--optimize` applies graph optimizations for ORT (only fuses LayerNorm for now)

So the recommended commands for MI250X are

1. export the model

    ```bash
    bash sample_run.sh 8 --export --optimize --merge
    ```

2. run the benchmark (note we still need `--optimize` here)

    ```bash
    bash sample_run.sh 8 --optimize --merge --custom-gen --benchmark --ort --torch --tunable --tuning
    ```

Or simply combine them

```bash
bash sample_run.sh 8 --export --optimize --merge --custom-gen --benchmark --ort --torch --tunable --tuning
```

See scripts [sample_run.sh](sample_run.sh) and [llama-v2.py](llama-v2.py) for more details.

## Llama2 model

This module is used to test distributed inference of Llama2 model.

The model file is based on huggingface modeling_llama.py and patched for distributed inference.

You can check the [modeling/patchinging_llama.py](modeling/patching_llama.py) script for more details.
