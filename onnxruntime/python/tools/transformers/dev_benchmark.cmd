@echo off

REM Run benchmark in Windows for developing purpose. For official benchmark, please use run_benchmark.sh.
REM Settings are different from run_benchmark.sh: no cli, batch and sequence, input counts, average over 100, no fp16, less models etc.

REM Please install PyTorch (see https://pytorch.org/) before running this benchmark. Like the following:
REM   GPU:   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
REM   CPU:   conda install pytorch torchvision cpuonly -c pytorch

REM When run_cli=true, this script is self-contained and you need not copy other files to run benchmarks
REM                    it will use onnxruntime-tools package.
REM If run_cli=false, it depends on other python script (*.py) files in this directory.
set run_cli=false

REM only need once
set run_install=false

REM Engines to test
set run_ort=true
set run_torch=false
set run_torchscript=false

REM Devices to test.
REM Attention: You cannot run both CPU and GPU at the same time: gpu need onnxruntime-gpu, and CPU need onnxruntime.
set run_gpu_fp32=false
set run_gpu_fp16=false
set run_cpu_fp32=true
set run_cpu_int8=true

set average_over=100

REM Enable optimizer (use script instead of OnnxRuntime for graph optimization)
set use_optimizer=true

set batch_sizes=1 16
set sequence_length=8 128

REM Number of inputs (input_ids, token_type_ids, attention_mask) for ONNX model.
REM Note that different input count might lead to different performance
set input_counts=1

REM Pretrained transformers models can be a subset of: bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
set models_to_test=bert-base-cased gpt2

REM If you have mutliple GPUs, you can choose one GPU for test. Here is an example to use the second GPU:
REM set CUDA_VISIBLE_DEVICES=1

REM This script will generate a logs file with a list of commands used in tests.
>benchmark.log echo echo ort=%run_ort% torch=%run_torch% torchscript=%run_torchscript% gpu_fp32=%run_gpu_fp32% gpu_fp16=%run_gpu_fp16% cpu=%run_cpu% optimizer=%use_optimizer% batch="%batch_sizes%" sequence="%sequence_length%" models="%models_to_test%" input_counts="%input_counts%"

REM Set it to false to skip testing. You can use it to dry run this script with the benchmark.log file.
set run_tests=true

REM -------------------------------------------
if %run_cpu_fp32% == true if %run_gpu_fp32% == true echo cannot test cpu and gpu at same time & goto :EOF
if %run_cpu_fp32% == true if %run_gpu_fp16% == true echo cannot test cpu and gpu at same time & goto :EOF
if %run_cpu_int8% == true if %run_gpu_fp32% == true echo cannot test cpu and gpu at same time & goto :EOF
if %run_cpu_int8% == true if %run_gpu_fp16% == true echo cannot test cpu and gpu at same time & goto :EOF

if %run_install% == true (
  pip uninstall --yes ort_nightly
  pip uninstall --yes onnxruntime
  pip uninstall --yes onnxruntime-gpu
  if %run_cpu% == true (
    pip install onnxruntime
  ) else (
    pip install --upgrade onnxruntime-gpu
  )

  pip install --upgrade onnxruntime-tools
  pip install --upgrade git+https://github.com/huggingface/transformers
)

if %run_cli% == true (
  echo Use onnxruntime_tools.transformers.benchmark
  set optimizer_script=-m onnxruntime_tools.transformers.benchmark
) else (
  set optimizer_script=benchmark.py
)

REM remove --overwrite can save some time if you did not update any of these: transformers, PyTorch or fusion logic in optimizer.
set onnx_export_options=-i %input_counts% -v -b 0 -f fusion.csv --overwrite
set benchmark_options=-b %batch_sizes% -s %sequence_length% -t %average_over% -f fusion.csv -r result.csv -d detail.csv

if %use_optimizer% == true (
  set onnx_export_options=%onnx_export_options% -o
  set benchmark_options=%benchmark_options% -o
)

if %run_gpu_fp32% == true (
  for %%m in (%models_to_test%) DO (
    echo Run GPU FP32 Benchmark on model %%m
    call :RunOneTest %%m -g
  )
)

if %run_gpu_fp16% == true (
  for %%m in (%models_to_test%) DO (
    echo Run GPU FP16 Benchmark on model %%m
    call :RunOneTest %%m -g -p fp16
  )
)

if %run_cpu_fp32% == true (
  for %%m in (%models_to_test%) DO (
    echo Run CPU FP32 Benchmark on model %%m
    call :RunOneTest %%m
  )
)

if %run_cpu_int8% == true (
  for %%m in (%models_to_test%) DO (
    echo Run CPU Int8 Benchmark on model %%m
    call :RunOneTest %%m -p int8
  )
)

if %run_tests%==false more benchmark.log

call :RemoveDuplicateLines result.csv
call :RemoveDuplicateLines fusion.csv
call :RemoveDuplicateLines detail.csv

echo Done!

goto :EOF

REM -----------------------------
:RunOneTest

if %run_ort% == true (
  >>benchmark.log echo python %optimizer_script% -m %1 %onnx_export_options% %2 %3 %4
  >>benchmark.log echo python %optimizer_script% -m %1 %benchmark_options% %2 %3 %4 -i %input_counts%
  if %run_tests%==true (
    python %optimizer_script% -m %1 %onnx_export_options% %2 %3 %4
    python %optimizer_script% -m %1 %benchmark_options% %2 %3 %4 -i %input_counts%
  )
)

if %run_torch% == true (
  >>benchmark.log echo python %optimizer_script% -e torch -m %1 %benchmark_options% %2 %3 %4
  if %run_tests%==true python %optimizer_script% -e torch -m %1 %benchmark_options% %2 %3 %4
)
  
if %run_torchscript% == true (
  >>benchmark.log echo python %optimizer_script% -e torchscript -m %1 %benchmark_options% %2 %3 %4
  if %run_tests%==true python %optimizer_script% -e torchscript -m %1 %benchmark_options% %2 %3 %4
)

goto :EOF


REM -----------------------------
:RemoveDuplicateLines
SET FileSize=%~z1
IF %FileSize% LSS 10 goto :EOF
python -c "import sys; lines=sys.stdin.readlines(); h=lines[0]; print(''.join([h]+list(sorted(set(lines)-set([h])))))"   < %1  > sort_%1
FindStr "[^,]" sort_%1 > summary_%1
DEL sort_%1
goto :EOF