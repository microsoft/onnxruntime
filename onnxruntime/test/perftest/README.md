# ONNX Runtime Performance Test

`onnxruntime_perf_test` measures inference latency and throughput of a model with ONNX Runtime using a chosen execution provider. It builds an inference session, runs a warm-up iteration, and then repeatedly runs the model (either for a fixed number of times or a fixed duration), reporting timing and resource-usage statistics.

## Building the tool

`onnxruntime_perf_test` is built together with the ONNX Runtime tests. Build from source with `--build` and tests enabled (the default), for example:

```bash
./build.sh --config Release --build_dir build/Release --parallel        # Linux/macOS
.\build.bat --config Release --build_dir build\Release --parallel        # Windows
```

The binary is produced under the build directory, for example `build/Release/Release/onnxruntime_perf_test`. See the [build instructions](https://onnxruntime.ai/docs/build/) for prerequisites and execution-provider specific flags.

## Usage

```
onnxruntime_perf_test [options...] model_path [result_file]
```

- `model_path`: path to the `.onnx` (or `.ort`) model file.
- `result_file`: optional path to append the run results to. If omitted, statistics (`-s`) are printed to stdout by default.

Options may be given with a single dash (`-e cpu`) or a double dash (`--e cpu`); both forms are equivalent.

For the complete, always-current list of options (including the many execution-provider specific runtime options passed via `-i`), run:

```bash
onnxruntime_perf_test --help
```

## Providing input data

The tool needs one set of inputs per model. There are two ways to supply them.

### 1. Auto-generate random input (simplest)

Pass `-I` to have the tool generate input tensors automatically. No data files are required. Free (symbolic) dimensions are treated as `1` unless overridden with `-f`, and `-S` sets a fixed random seed for reproducible data.

```bash
onnxruntime_perf_test -I -e cpu model.onnx
# override a symbolic dimension named "batch" to 4
onnxruntime_perf_test -I -f "batch:4" -e cpu model.onnx
```

### 2. Provide test data files

The tool reuses the `onnx_test_runner` / ONNX backend-test data layout. Place one or more input-set subdirectories next to the model file:

```
<model_dir>/
├── model.onnx                  # pass this path as model_path (any .onnx name works)
├── test_data_set_0/            # one input set
│   ├── input_0.pb              # first model input  (serialized onnx.TensorProto)
│   └── input_1.pb              # second model input, ...
└── test_data_set_1/            # optional additional input set(s)
    └── input_0.pb
```

Notes:

- Every subdirectory of the model's directory is treated as one input set, and the tool cycles through the sets across iterations. The conventional name is `test_data_set_<N>`, but any subdirectory name works.
- Within a set, files named `input_<N>.pb` are loaded in sorted order and bound to the model inputs by position. Each `.pb` file is a serialized `onnx.TensorProto`.

#### Creating `input_<N>.pb` files

Use the helper script [`tools/python/onnx_test_data_utils.py`](../../../tools/python/onnx_test_data_utils.py) to generate a serialized `TensorProto`. For example, to create a random `float32` tensor of shape `10240x512` for a model input named `x`:

```bash
python tools/python/onnx_test_data_utils.py \
    --action random_to_pb \
    --name x \
    --shape 10240,512 \
    --datatype f4 \
    --output my_model/test_data_set_0/input_0.pb
```

`--name` must match the model's input name. `--datatype` is a numpy dtype string (for example `f4` = float32, `f2` = float16, `i8` = int64), and `--seed` can be used for deterministic values. Run `python tools/python/onnx_test_data_utils.py --help` for the full set of actions (for example converting existing `.npy` data to `.pb`).

## Examples

```bash
# Auto-generated input, CPU EP, show statistics
onnxruntime_perf_test -I -e cpu -s model.onnx

# Run for a fixed number of iterations (times mode) on CUDA
onnxruntime_perf_test -e cuda -m times -r 2000 model.onnx result.txt

# Run for a fixed duration (duration mode) for 30 seconds
onnxruntime_perf_test -e cpu -m duration -t 30 model.onnx

# Use test data directories located next to the model
onnxruntime_perf_test -e cpu my_model/model.onnx

# Pass an execution-provider specific runtime option (TensorRT FP16)
onnxruntime_perf_test -e tensorrt -i "trt_fp16_enable|true" model.onnx
```

## Common options

| Option | Description |
| --- | --- |
| `-e [provider]` | Execution provider: `cpu` (default), `cuda`, `dnnl`, `tensorrt`, `nvtensorrtrtx`, `openvino`, `dml`, `acl`, `nnapi`, `coreml`, `qnn`, `snpe`, `migraphx`, `xnnpack`, `vitisai`, `webgpu`. |
| `-m [mode]` | Test mode: `duration` (default) or `times`. |
| `-r [count]` | Number of iterations to run in `times` mode. Default: 1000. |
| `-t [seconds]` | Seconds to run in `duration` mode. Default: 600. |
| `-c [count]` | Max number of runs to invoke simultaneously. Default: 1. |
| `-I` | Auto-generate model input; no test data files required. |
| `-S [seed]` | Random seed for generated input data (for reproducibility). Default: -1 (uninitialized). |
| `-f "name:value"` | Override a free (symbolic) dimension by name. May be repeated. |
| `-x [count]` | Intra-op thread count (0 lets ORT choose). |
| `-y [count]` | Inter-op thread count (0 lets ORT choose). |
| `-o [level]` | Graph optimization level: 0 (disable), 1 (basic), 2 (extended), 3 (layout), 99 (all). Default: 99. |
| `-p [file]` | Enable profiling and write the profile data to `file`. |
| `-i "k1\|v1 k2\|v2"` | Execution-provider specific runtime options (see `--help` for per-provider keys). |
| `-C "k1\|v1 k2\|v2"` | Session configuration entries. See `onnxruntime_session_options_config_keys.h` for valid keys. |
| `-s` | Show latency statistics (P50, P90, ...). Defaults to on when no `result_file` is given. |
| `-v` | Verbose output. |
| `-h` | Print the full usage, including all options. |

This is a curated subset of the most commonly used options. Run `onnxruntime_perf_test --help` for the authoritative and complete list.

## Sample output

A typical summary printed to stdout looks like:

```
Session creation time cost: 0.512 s
First inference time cost: 12 ms
Total inference time cost: 5.88053 s
Total inference requests: 1000
Average inference time cost total: 5.88053 ms
Total inference run time: 5.88102 s
Number of inferences per second: 170.04
Avg CPU usage: 98 %
Peak working set size: 123456789 bytes
```

When `-s` is enabled, the latency percentiles are also reported:

```
Min Latency: 0.0559777 s
Max Latency: 0.0623472 s
P50 Latency: 0.0587108 s
P90 Latency: 0.0599845 s
P95 Latency: 0.0605676 s
P99 Latency: 0.0619517 s
P999 Latency: 0.0623472 s
```
