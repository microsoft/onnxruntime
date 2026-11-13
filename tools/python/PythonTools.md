# Python Helpers

## generate_matmul_nbits_cases.py

Generate deterministic, single-operator `com.microsoft::MatMulNBits` models for workload-estimation experiments.
Requires NumPy and ONNX; generation does not require an ORT wheel or GPU.

```bash
python tools/python/generate_matmul_nbits_cases.py --output /tmp/matmul-nbits-pilot --seed 123
```

The default 14-case INT4 pilot includes decode/prefill, bias and zero-point variants, vector/batched inputs,
unaligned K/N held-out cases, FP16, and a requested INT8 accuracy level. It is not exhaustive route coverage.
For other configurations, use `--config experiments.json` with an explicit case list:

```json
{
  "cases": [
    {
      "a_shape": [2, 3, 65],
      "n": 32,
      "bits": 4,
      "block_size": 32,
      "dtype": "float32",
      "bias": true,
      "zero_points": true,
      "accuracy_level": 0,
      "cohort": "held_out"
    }
  ]
}
```

Required fields are `a_shape` and `n`. Other defaults are bits=4, block_size=32, dtype=float32, bias=false,
zero_points=false, accuracy_level=0 and cohort=calibration. Supported generation settings are 2/4/8 bits,
FP32/FP16, positive dimensions, and power-of-two block sizes >=16. Kernel eligibility must be checked during
execution; an accuracy level is a request, not proof of the effective compute mode. BF16, empty dimensions,
dynamic shapes and EP-specific prepacked layouts are outside this initial generator.

Each case directory contains `model.onnx` with constant weights/scales/optional bias/zero points and
`inputs.npz` containing only activation `A` (load with `allow_pickle=False`). The final `manifest.json` lists
relative paths, settings, per-case seeds, output shapes, expected invocation multiplicity, and exact logical
core/bias reference counts as decimal strings. Case IDs and generated tensors are independent of list ordering
and cohort assignment; duplicate settings across cohorts are rejected to prevent validation leakage.
The manifest records NumPy/ONNX versions; reproduce with the same dependencies and base seed.

Weights are synthetic unsigned quantized codes, not quantized real/model weights. Packing uses the standard
column-wise low-bits-first layout, including packed zero points and zero-valued dequantized K padding.
Positive power-of-two scales keep the data controlled. An omitted zero point means the schema's midpoint.
No executed-FLOP reference, whole-op count, numerical reference output, timing or route validation is claimed.

Existing output directories are rejected, never overwritten. Invalid configuration is rejected before creating
the output directory. Generation failures are explicit and may leave partial case files, but no completed
manifest; retry in a new directory. Large explicit cases can consume substantial memory/disk.

Targeted validation:

```bash
python -m pytest tools/python/test_generate_matmul_nbits_cases.py
```

## run_matmul_nbits_experiments.py

Run CPU capture/report validation against cases from `generate_matmul_nbits_cases.py`. Use a Python environment
with NumPy, ONNX and an ORT wheel built from the branch containing CPU route-aware `expected_flops` reporting.
An older logical-only estimator or a published wheel without the feature must not pass these checks.

```bash
python tools/python/run_matmul_nbits_experiments.py \
  --manifest /tmp/matmul-nbits-pilot/manifest.json \
  --output /tmp/matmul-nbits-results \
  --runtime-label "local branch CPU Release wheel"
```

The runner requests CPU only, disables graph optimizations, uses sequential execution and defaults to one
intra-op thread (`--threads` changes this). Each case runs once in a reporting-disabled session and once in a
separate reporting-enabled session. It checks finite output shapes/dtypes and exact reporting-on/off equality.
Explicit `--rtol` and `--atol` allow nonnegative finite tolerances, recorded in the results. This is an
instrumentation-equivalence check, not comparison against independent numerical reference outputs.

Reports must appear after Run, not initialization, and must contain exactly one successfully captured invocation
without drops/reuse rejection. Shape, dtype, bias, bit width, block size, provider, route/mode consistency and
backend geometry are checked. Logical core and bias counts are recomputed from settings, checked against the
manifest, and compared exactly against both per-invocation counts and report subtotals.

The separate expected-FLOP components/model/availability are retained. The runner checks count encoding,
subtotal/availability consistency and independently known bias, but deliberately does not duplicate route-aware
core/dequantization formulas or claim measured executed-FLOP accuracy. `pass` means capture/logical-count/output
checks passed, not that the floating-point estimate matches actual execution. Requested accuracy level is not
assumed to equal actual effective compute mode.

Outputs:

- `results.json`: runtime/import/build information, user-supplied runtime label, execution settings, per-case statuses,
  captured estimates, and status/route totals. `executed_fp_accuracy` remains `not_validated`. The file is updated
  after each case; `complete: false` identifies an interrupted/incomplete batch.
- `summary.csv`: compact case/status/route/mode/expected-FLOP table with diagnostic messages and report paths.
- Per-case directories: raw uniquely named estimator report and `outputs.npz` containing reporting-disabled/enabled
  outputs when execution succeeds.

The runtime label is user-supplied provenance, not proof of the installed wheel's source revision. Verify the
recorded import path/build information and retain the actual wheel/build provenance. Run from outside the source
tree with an absolute script path if source-package shadowing prevents importing the installed wheel.

Every non-pass case is printed with a diagnostic and retained in the results; execution proceeds to other cases.
Statuses distinguish `input_failure`, `execution_failure`, `output_mismatch`, `report_missing`, `report_invalid`,
`incomplete_capture`, `metadata_mismatch`, `count_mismatch`, `estimator_unavailable` and `estimate_unavailable`.
Unsupported route estimates are not assigned zero or counted as passes. Exit codes: 0 if all cases pass,
1 if any case does not pass, 2 for setup/output/import errors. Existing results directories are never overwritten.
Manifest validation occurs before output creation. Fatal output errors may leave partial results.

This initial runner has no CUDA, timing, warm-up, hardware profiling, route-forcing or whole-op FP-count validation.
It only exercises the routes naturally selected by the installed CPU runtime. Independent route audits/profiling
are the next step for assessing estimated versus actual FP work.

```bash
python -m pytest tools/python/test_generate_matmul_nbits_cases.py tools/python/test_run_matmul_nbits_experiments.py
```

## Repeatable MatMulNBits matrix and hardware profiling

For the CPU/CUDA qualification roadmap, evidence boundaries and CUDA commands, see the
[testing strategy](../../docs/annotated_partitioning/matmul_nbits_experiments/testing_strategy.md).
`profile_matmul_nbits_cuda.py` is a separate single-case capture/NVTX driver; it does not use CPU perf counters
or claim automatic instruction-derived GPU FLOP counts.

`generate_matmul_nbits_matrix.py` preserves the 84-case matrix used in the November 2026 CPU experiment:
FP32 inputs at accuracy levels 0-4 and FP16 at levels 0/4, crossed with M=1/8/32/128 and
(K,N)=(64,32)/(128,64)/(1024,1024). All cases use INT4, block size 32, no bias and implicit midpoint zero points.
It reuses the case generator and also saves the explicit configuration. Requested accuracy is not observed mode.

`profile_matmul_nbits.py` replaces the inline perf collection/report commands. It requires Linux `perf`, usable
`fp_ret_sse_avx_ops.all:u` hardware counters, `lscpu`, NumPy, ONNX and the same ORT build used by the estimator runner.
Counter names/semantics are architecture-specific; this event was exercised on AMD EPYC 7V12. Validate PMU access
and arithmetic-event semantics before using another host. No privileges or system security settings are changed.

From an activated environment containing the branch wheel, use absolute paths to avoid source-package shadowing:

```bash
REPO=/absolute/path/to/onnxruntime
EXP=$(mktemp -d /tmp/matmul-matrix-XXXXXX)
cd /tmp

python "$REPO/tools/python/generate_matmul_nbits_matrix.py" --output "$EXP/cases"
python "$REPO/tools/python/run_matmul_nbits_experiments.py" \
  --manifest "$EXP/cases/manifest.json" --output "$EXP/estimates"
python "$REPO/tools/python/profile_matmul_nbits.py" collect \
  --manifest "$EXP/cases/manifest.json" --results "$EXP/estimates/results.json" \
  --output "$EXP/three-trials" --trials 3 --repeats 1000 --warmup 100
```

Only continue when estimator checks pass. Collection requires all cases to pass, matching manifest/result settings,
and matching installed ORT version/build info. Build-info equality does not prove binary identity; retain wheel
hashes separately when freezing a baseline. Actual route/mode labels come from the preceding estimator run,
not independent instrumentation of each perf subprocess.

Collection automatically launches the workload under perf; no second terminal or PID entry is needed. It preserves
the runner's thread count, sequential CPU execution and disabled graph optimizations. Each process initializes a
session, warms up, then executes R or 2R times with reporting disabled. With 84 cases, three pairs produce 504
process measurements. More repetitions may be needed for noisy cases; this is FP counting, not inference timing.

The new output directory contains:

- `metadata.json`, `manifest.json`, `estimates.json`: method, build/platform information and estimator snapshots.
- `index.json` and `hardware/`: exact commands, raw semicolon-delimited counters and subprocess logs.
- `trial-results.json`, `trials.csv`: `(count_2R - count_R) / R` per trial.
- `summary.csv`, `report.md`: per-case mean, sample standard deviation, min/max and signed differences;
  the report includes the accuracy-level key and measurement-scope limitations.

The script rejects failed commands, missing/duplicate counters, nonpositive differential counts and counters
reporting less than 100% running time. Errors stop explicitly with `complete: false` in collection metadata.
Only a fully validated matrix produces a completed report. Output directories are never overwritten.
Zero sample standard deviation means identical derived counts; for one trial it is unavailable, not zero.
Percentage difference is unavailable when the estimate is zero.

Regenerate documentation without rerunning hardware collection:

```bash
python "$REPO/tools/python/profile_matmul_nbits.py" report \
  --collection "$EXP/three-trials" --output "$EXP/report-regenerated"
```

For the earlier inline three-trial collection, supply its successful estimator results explicitly:

```bash
python "$REPO/tools/python/profile_matmul_nbits.py" report \
  --collection /path/to/matmul-three-trials-directory \
  --results /path/to/baseline/estimates-v2/results.json \
  --output /path/to/new-report-directory
```

Historical indexes with absolute counter paths are accepted only while those paths remain inside the collection.
If moving a collection, update a copy of the index to use relative paths. Reports are recomputed from raw counters,
not copied from previously calculated summaries. Existing November 9 results are historical inline-command results;
these scripts preserve the method but are not claimed to be the exact original executed driver.

### Bias and tail matrix

The same matrix generator supports `--preset bias-tails`, leaving the default 84-case compute preset unchanged.
The default boundary preset has 128 cases for B=32: FP32 input, INT4 weights, accuracy requests 0/4, implicit
midpoint zero points, and matched bias-absent/present cases. It is an exploratory/calibration cohort, not held-out
qualification. The one-factor probes are:

- K=B-1/B/B+1, 2B-1/2B/2B+1, 3B and 4B at M=1/8, N=32: partial blocks and odd/even block counts.
- N=3/4/5 and 31/32/33 at M=1/8, K=4B: small column tiles and strip boundaries.
- M=2/3/7/9/32 at K=4B, N=32, supplementing M=1/8 controls: row-tail and multi-row branches.
- One combined-tail probe: M=3, K=2B+1, N=33.

Every geometry/request has bias off/on. Duplicate controls are removed. Explicit zero-point, dtype, bit-width,
threading and zero-activation-block experiments remain separate to avoid confounding this first matrix.
Optional `--block-sizes 32 64 128` expands to 384 cases; start with B=32 before expanding.

```bash
python "$REPO/tools/python/generate_matmul_nbits_matrix.py" \
  --preset bias-tails --output "$EXP/cases"
python "$REPO/tools/python/run_matmul_nbits_experiments.py" \
  --manifest "$EXP/cases/manifest.json" --output "$EXP/estimates"
```

Inspect every non-pass status and observed compute mode before collecting counters. Valid generated shapes are
not proof that every host kernel accepts them. Preserve failures and investigate them; do not silently filter
them out to obtain a successful hardware batch. The perf collector requires every supplied case to have passed.
With all 128 cases passing, three R/2R trial pairs produce 768 process measurements:

```bash
python "$REPO/tools/python/profile_matmul_nbits.py" collect \
  --manifest "$EXP/cases/manifest.json" --results "$EXP/estimates/results.json" \
  --output "$EXP/hardware" --trials 3 --repeats 1000 --warmup 100
```

Do not automatically apply aligned arithmetic candidates to tails. Logical bias remains MN, but executed bias
work can depend on epilogue lanes and placement. Paired process-count differences are exploratory evidence:
check route changes and compiled arithmetic before attributing the difference entirely to bias.

Measured recurring process FP work and modeled operator components have different scopes. Process subtraction
does not guarantee startup cancellation, and recurring runtime work remains included. INT8 activation quantization,
reductions and tile repetition may not be modeled. Differences are not isolated operator accuracy percentages.

```bash
python -m pytest tools/python/test_profile_matmul_nbits.py \
  tools/python/test_generate_matmul_nbits_cases.py tools/python/test_run_matmul_nbits_experiments.py
```

## Authentication method for Azure DevOps

Generally speaking Azure DevOps services has three ways of doing authentication:

1. Microsoft Entra ID(with an Entra token)
2. Personal Access Tokens(PATs)
3. Azure DevOps OAuth

In this project all tools should only use the first one. An Entra token is very long compared to PATs. There are two ways of getting such a token:

1. Use a service connection with Workload Identity Federation, which is for scripts that run in pipelines in Azure cloud.
2. Get it from AzCli, which is for scripts that runs on Microsoft developers machines.  

Some tools need to support the both scenarios, for example, nuget.exe. We are still exploring ways to make it work well with Entra ID auth(need to find a way to support cross-org nuget publishing).

## delete_ado_pipeline.py
Prerequisites: Install AzCli and get logged in.

This script is designed to completely delete an Azure DevOps pipeline definition and all of its associated data.

The process is as follows:
1. Asynchronously finds and deletes ALL retention leases associated with the pipeline.
2. Asynchronously deletes ALL build runs (history) for the pipeline.
3. Deletes the pipeline definition itself.


## upload_symbol.py

This script downloads ONNX Runtime Windows release artifacts for a specified version from GitHub. Extracts PDB symbol files and uploads them to your Azure DevOps symbol server, simplifying debugging for teams hosting symbols there.

This script requires symbols.exe, which can be downloaded by using the following Azure REST API: https://learn.microsoft.com/en-us/rest/api/azure/devops/symbol/client/get?view=azure-devops-rest-7.1

symbols.exe handles authentication.

## run_packaging_pipelines.py

The script is for triggering Azure DevOps pipelines for a specific Github pull request or git branch. ONNX Runtime's release managers should use this script to trigger packaging pipelines.
You may also use this script to trigger pull request pipelines for external PRs. 

### Prerequisites

1. Install AzCli and get logged in.
2. pip install pyyaml

### Usages

It supports two modes:

1. CI Build Mode (Default):
   - Triggers pipelines in the 'Lotus' project.
   - Filters pipelines based on the following criteria:
     - Repository Association: Must be 'https://github.com/microsoft/onnxruntime'.
     - Recent Activity: Must have run in the last 30 days.
     - Pipeline Type: Must be YAML-based.
     - Trigger Type: Must NOT be triggered by another pipeline resource.
     - Template Requirement: Must extend from 'v1/1ES.Official.PipelineTemplate.yml@1esPipelines'.
2. Pull Request (PR) Mode:
   - Activated by using the '--pr <ID>' argument.
   - Triggers pipelines in the 'PublicPackages' project.
   - Filters pipelines based on a simplified criteria:
     - Repository Association: Must be 'https://github.com/microsoft/onnxruntime'.
     - Recent Activity: Must have run in the last 30 days.
     - Pipeline Type: Must be YAML-based.

The script also includes a feature to cancel any currently running builds for a matching
pipeline on the target branch/PR before queuing a new one.

## ort_test_dir_utils.py

Provides helpers for creating ONNX test directories that can be run using onnx_test_runner and onnxruntime_perf_test.

In order to import ort_test_dir_utils you need to either
  - run python from the `<onnxruntime root dir`>/tools/python directory
  - add the directory to your PYTHONPATH
  - add the directory to sys.path prior to importing

e.g. to add to sys.path
```python
import sys
sys.path.append('<onnxruntime root dir>/tools/python')

import ort_test_dir_utils
```

### Creating a test directory for a model.

The create_test_dir helper can create the input and output pb files in various ways.

Often a support request will only provide a problematic model and no input data. create_test_dir can be used to create input to allow the model to be debugged more easily. Random input can be generated if not provided. If expected output is not provided, the model will be run with the input, and the output from that will be saved as the expected output.

```python
def create_test_dir(model_path, root_path, test_name,
                    name_input_map=None, symbolic_dim_values_map=None,
                    name_output_map=None):
    """
    Create a test directory that can be used with onnx_test_runner, onnxruntime_perf_test.
    Generates random input data for any missing inputs.
    Saves output from running the model if name_output_map is not provided.

    :param model_path: Path to the onnx model file to use.
    :param root_path: Root path to create the test directory in.
    :param test_name: Name for test. Will be added to the root_path to create the test directory name.
    :param name_input_map: Map of input names to numpy ndarray data for each input.
    :param symbolic_dim_values_map: Map of symbolic dimension names to values to use for the input data if creating
                                    using random data.
    :param name_output_map: Optional map of output names to numpy ndarray expected output data.
                            If not provided, the model will be run with the input to generate output data to save.
    :return: None
    """
```

Example usage:

```python
import sys
import ort_test_dir_utils
import onnx_test_data_utils

# example model with two float32 inputs called 'input1' (dims: {2, 1}) and 'input2' (dims: {'dynamic', 4})
model_path = '<onnxruntime root dir>/onnxruntime/test/testdata/transform/expand_elimination.onnx'

# when using the default data generation any symbolic dimension values must be provided
symbolic_vals = {'dynamic':2} # provide value for symbolic dim named 'dynamic' in 'input2'

# let create_test_dir create random input in the (arbitrary) default range of -10 to 10.
# it will create data of the correct type based on the model.
ort_test_dir_utils.create_test_dir(model_path, 'temp/examples', 'test1', symbolic_dim_values_map=symbolic_vals)

# alternatively some or all input can be provided directly. any missing inputs will have random data generated.
# symbolic dimension values are only required for input data that is randomly generated,
# so we don't need to provide that in this case as we're explicitly providing all inputs.
inputs = {'input1': np.array([100, 200]).reshape((2,1)).astype(np.float32),
          'input2': np.random.randn(2,4).astype(np.float32)} # use 2 for the 'dynamic' dimension so shape is {2, 4}

ort_test_dir_utils.create_test_dir(model_path, 'temp/examples', 'test2', name_input_map=inputs)

# can easily dump the input and output to visually check it's as expected
onnx_test_data_utils.dump_pb('temp/examples/test2/test_data_set_0')
```

### Running the test using python

To execute the test once the directory is created you can use the onnx_test_runner or onnxruntime_perf_test executables if you have built onnxruntime from source, or the run_test_dir helper. Input can be either the test directory, or the model in case there are multiple in the test directory.

```python
def run_test_dir(model_or_dir):
    """
    Run the test/s from a directory in ONNX test format.
    All subdirectories with a prefix of 'test' are considered test input for one test run.

    :param model_or_dir: Path to onnx model in test directory,
                         or the test directory name if the directory only contains one .onnx model.
    :return: None
    """
```

Example usage:

```python
import sys
import ort_test_dir_utils

try:
    ort_test_dir_utils.run_test_dir('temp/examples/test1')
    ort_test_dir_utils.run_test_dir('temp/examples/test2/expand_elimination.onnx')
except Exception:
    print("Exception:", sys.exc_info()[1])
```

## onnx_test_data_utils.py

Provides helpers for generating/reading protobuf files containing ONNX TensorProto data.

```
usage: onnx_test_data_utils.py [-h] --action {dump_pb,numpy_to_pb,image_to_pb,random_to_pb,update_name_in_pb}
                               [--input INPUT] [--name NAME] [--output OUTPUT] [--resize RESIZE] [--channels_last] [--add_batch_dim]
                               [--shape SHAPE] [--datatype DATATYPE] [--min_value MIN_VALUE] [--max_value MAX_VALUE] [--seed SEED]

        Utilities for working with the input/output protobuf files used by the ONNX test cases and onnx_test_runner.
        These are expected to only contain a serialized TensorProto.

        dump_pb: Dumps the TensorProto data from an individual pb file, or all pb files in a directory.
        numpy_to_pb: Convert numpy array saved to a file with numpy.save() to a TensorProto, and serialize to a pb file.
        image_to_pb: Convert data from an image file into a TensorProto, and serialize to a pb file.
        random_to_pb: Create a TensorProto with random data, and serialize to a pb file.
        update_name_in_pb: Update the TensorProto.name value in a pb file.
                           Updates the input file unless --output <filename> is specified.


optional arguments:
  -h, --help            show this help message and exit
  --action {dump_pb,numpy_to_pb,image_to_pb,random_to_pb,update_name_in_pb}
                        Action to perform
  --input INPUT         The input filename or directory name
  --name NAME           The value to set TensorProto.name to if creating/updating one.
  --output OUTPUT       Filename to serialize the TensorProto to.

image_to_pb:
  image_to_pb specific options

  --resize RESIZE       Provide the height and width to resize to as comma separated values. e.g. --shape 200,300 will resize to height 200 and width 300.
  --channels_last       Transpose image from channels first to channels last.
  --add_batch_dim       Prepend a batch dimension with value of 1 to the shape. i.e. convert from CHW to NCHW

random_to_pb:
  random_to_pb specific options

  --shape SHAPE         Provide the shape as comma separated values e.g. --shape 200,200
  --datatype DATATYPE   numpy dtype value for the data type. e.g. f4=float32, i8=int64. See: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
  --min_value MIN_VALUE
                        Limit the generated values to this minimum.
  --max_value MAX_VALUE
                        Limit the generated values to this maximum.
  --seed SEED           seed to use for the random values so they're deterministic.
```

## dump_subgraphs.py

If you're investigating a model with control flow nodes (Scan/Loop/If) the subgraphs won't be displayed in Netron. Run dump_subgraphs to dump the subgraphs as .onnx files that can be viewed individually.

```
usage: dump_subgraphs.py [-h] -m MODEL [-o OUT]

Dump all subgraphs from an ONNX model into separate onnx files.

optional arguments:
  -h, --help                show this help message and exit
  -m MODEL, --model MODEL   model file
  -o OUT, --out OUT         output directory (default: <current dire)
```

## onnx2tfevents.py

Convert ONNX model to TensorBoard events file so that we can visualize the model in TensorBoard. This is especially useful for debugging large models that cannot be visualized in Netron.

```
usage: onnx2tfevents.py [-h] [--logdir LOGDIR] [--model MODEL]

optional arguments:
  -h, --help       show this help message and exit
  --logdir LOGDIR  Tensorboard log directory
  --model MODEL    ONNX model path
```

Once the events file is generated, start the tensorboard and visualize the model graph.

Note: This tool requires torch and tensorboard to be installed.
