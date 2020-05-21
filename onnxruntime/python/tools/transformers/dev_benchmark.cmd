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
set run_gpu_fp32=true
set run_gpu_fp16=false
set run_cpu=false

set average_over=100

REM Enable optimizer (use script instead of OnnxRuntime for graph optimization)
set use_optimizer=true

set batch_sizes=1 16
set sequence_length=8 128

REM Number of inputs (input_ids, token_type_ids, attention_mask) for ONNX model.
REM Note that different input count might lead to different performance
set input_counts=1 2 3

REM Pretrained transformers models can be a subset of: bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
set models_to_test=bert-base-cased gpt2

REM If you have mutliple GPUs, you can choose one GPU for test. Here is an example to use the second GPU:
REM set CUDA_VISIBLE_DEVICES=1

REM This script will generate a logs file with a list of commands used in tests.
echo echo ort=%run_ort% torch=%run_torch% torchscript=%run_torchscript% gpu_fp32=%run_gpu_fp32% gpu_fp16=%run_gpu_fp16% cpu=%run_cpu% optimizer=%use_optimizer% batch="%batch_sizes%" sequence="%sequence_length%" models="%models_to_test%" input_counts="%input_counts%" > benchmark.log

REM Set it to false to skip testing. You can use it to dry run this script with the benchmark.log file.
set run_tests=true

REM -------------------------------------------

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

set onnx_export_options=-i %input_counts% -v -b 0 --overwrite -f fusion.csv
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
    call :RunOneTest %%m -g --fp16
  )
)

if %run_cpu% == true (
  for %%m in (%models_to_test%) DO (
    echo Run CPU Benchmark on model %%m
    call :RunOneTest %%m
  )
)

if %run_tests%==true more %log_file%

call :RemoveDuplicateLines result.csv
call :RemoveDuplicateLines fusion.csv
call :RemoveDuplicateLines detail.csv
goto :EOF

REM -----------------------------
:RunOneTest

if %run_ort% == true (
  echo python %optimizer_script% -m %1 %onnx_export_options% %2 %3 >> benchmark.log
  echo python %optimizer_script% -m %1 %benchmark_options% %2 %3 -i %input_counts% >> benchmark.log
  if %run_tests%==true (
    python %optimizer_script% -m %1 %onnx_export_options% %2 %3
    python %optimizer_script% -m %1 %benchmark_options% %2 %3 -i %input_counts%
  )
)

if %run_torch% == true (
  echo python %optimizer_script% -e torch -m %1 %benchmark_options% %2 %3 >> benchmark.log
  if %run_tests%==true python %optimizer_script% -e torch -m %1 %benchmark_options% %2 %3
)
  
if %run_torchscript% == true (
  echo python %optimizer_script% -e torchscript -m %1 %benchmark_options% %2 %3 >> benchmark.log
  if %run_tests%==true python %optimizer_script% -e torchscript -m %1 %benchmark_options% %2 %3
)

goto :EOF


REM -----------------------------
REM this might have one duplicated header line, that is not a big deal.
:RemoveDuplicateLines
@echo off
setlocal enableDelayedExpansion
set "file=%1"
set /p "ln=" < "%file%"
>"%file%.new" (
  echo(!ln!
  more +1 "%file%" | sort /UNIQUE
)

REM remove lines without data
findstr /v /r /c:"^[, ]*$" "%file%.new" > "%file%.new"

move /y "%file%.new" "%file%" >nul
goto :EOF
