REM Run benchmark in Windows for developing purpose: average over 100 times inferences per test.
REM Please install PyTorch 1.5.0 (see https://pytorch.org/) before running this benchmark. For example,
REM   GPU:   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
REM   CPU:   conda install pytorch torchvision cpuonly -c pytorch
@echo off

REM When run_cli=true, this script is self-contained and you need not copy other files to run benchmarks - it will use onnxruntime-tools package
REM If run_cli=false, it depends on other python script (*.py) files in this directory.
set run_cli=false

REM only need once
set run_install=false

REM Engines to test.
set run_ort=true
set run_torch=false
set run_torchscript=true

REM Devices to test (You can run either CPU or GPU, but not both: gpu need onnxruntime-gpu, and CPU need onnxruntime).
set run_gpu_fp32=true
set run_gpu_fp16=false
set run_cpu=false

REM enable optimizer (use script instead of OnnxRuntime for graph optimization)
set use_optimizer=true

set batch_sizes=1 16
set sequence_length=8 128

REM Pretrained transformers models can be a subset of: bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
set test_models=bert-base-cased gpt2

REM If you have mutliple GPUs, you can choose one GPU for test. Here is an example to use the second GPU:
REM set CUDA_VISIBLE_DEVICES=1

REM -------------------------------------------

if %run_install% == true (
  if %run_cpu% == true (
    pip install --upgrade onnxruntime
  ) else (
    pip install --upgrade onnxruntime-gpu
  )
  pip install --upgrade onnxruntime-tools
  pip install --upgrade git+https://github.com/huggingface/transformers
)

if %run_cli% == true (
  echo "Use onnxruntime_tools.transformers.benchmark" 
  set OPTIMIZER_SCRIPT="-m onnxruntime_tools.transformers.benchmark"
) else (
  set OPTIMIZER_SCRIPT="benchmark.py"
)

set ONNX_EXPORT_OPTIONS=-v -b 0 --overwrite -f fusion.csv
set BENCHMARK_OPTIONS=-b %batch_sizes% -s %sequence_length% -t 100 -f fusion.csv -r result.csv -d detail.csv

if %enable_optimization% == true (
  set ONNX_EXPORT_OPTIONS=%ONNX_EXPORT_OPTIONS% -o
  set BENCHMARK_OPTIONS=%BENCHMARK_OPTIONS% -o
)


if %run_gpu_fp32% == true (
  for %%m in (%test_models%) DO (
    echo "Run GPU FP32 Benchmark on model %%m
    call :RunOneTest %%m -g
  )
)

if %run_gpu_fp16% == true (
  for %%m in (%test_models%) DO (
    echo "Run GPU FP16 Benchmark on model %%m
    call :RunOneTest %%m -g --fp16
  )
)

if %run_cpu% == true (
  for %%m in (%test_models%) DO (
    echo "Run CPU Benchmark on model %%m
    call :RunOneTest %%m
  )
)

call :RemoveDuplicateLines result.csv
call :RemoveDuplicateLines fusion.csv
call :RemoveDuplicateLines detail.csv
goto :EOF


REM -----------------------------
:RunOneTest
SET BENCHMARK_MODEL=%1

if %run_ort% == true (
  python %OPTIMIZER_SCRIPT% -m %1 %ONNX_EXPORT_OPTIONS% %2 %3
  python %OPTIMIZER_SCRIPT% -m %1 %BENCHMARK_OPTIONS% %2 %3
)
    
if %run_torch% == true (
  python %OPTIMIZER_SCRIPT% -e torch -m %1 %BENCHMARK_OPTIONS% %2 %3
)
  
if %run_torchscript% == true (
  python %OPTIMIZER_SCRIPT% -e torchscript -m %1 %BENCHMARK_OPTIONS% %2 %3
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
move /y "%file%.new" "%file%" >nul
goto :EOF