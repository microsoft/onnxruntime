# Run benchmark in Linux for measurement: average over 1000 inferences per test.
# Please install PyTorch 1.5.0 (see https://pytorch.org/) before running this benchmark:
# GPU:   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# CPU:   conda install pytorch torchvision cpuonly -c pytorch

# When run_cli=true, this script is self-contained and you need not copy other files to run benchmarks - it will use onnxruntime-tools package.
# If run_cli=false, it depends on other python script (*.py) files in this directory.
run_cli=true

# only need once
run_install=true

# Engines to test.
run_ort=true
run_torch=false
run_torchscript=true

# Devices to test (You can run either CPU or GPU, but not both: gpu need onnxruntime-gpu, and CPU need onnxruntime).
run_gpu_fp32=true
run_gpu_fp16=true
run_cpu=false

# enable optimizer (use script instead of OnnxRuntime for graph optimization)
use_optimizer=true

# Batch Sizes and Sequence Lengths
batch_sizes="1 4"
sequence_lengths="8 16 32 64 128 256 512 1024"

# Pretrained transformers models can be a subset of: bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
test_models="bert-base-cased roberta-base gpt2"

# If you have mutliple GPUs, you can choose one GPU for test. Here is an example to use the second GPU:
# export CUDA_VISIBLE_DEVICES=1

# -------------------------------------------
if [ "$run_install" = true ] ; then
  if [ "$run_cpu" = true ] ; then
    pip install --upgrade onnxruntime
  else
    pip install --upgrade onnxruntime-gpu
  fi
  pip install --upgrade onnxruntime-tools
  pip install --upgrade git+https://github.com/huggingface/transformers
fi

if [ "$run_cli" = true ] ; then
  echo "Use onnxruntime_tools.transformers.benchmark" 
  export OPTIMIZER_SCRIPT="-m onnxruntime_tools.transformers.benchmark"
else
  export OPTIMIZER_SCRIPT="benchmark.py"
fi

export ONNX_EXPORT_OPTIONS="-v -b 0 --overwrite -f fusion.csv"
export BENCHMARK_OPTIONS="-b $BATCH_SIZES -s $SEQUENCE_LENGTHS -t 1000 -f fusion.csv -r result.csv -d detail.csv"

if [ "$use_optimizer" = true ] ; then
  export ONNX_EXPORT_OPTIONS="$ONNX_EXPORT_OPTIONS -o"
  export BENCHMARK_OPTIONS="$BENCHMARK_OPTIONS -o"
fi

# -------------------------------------------
run_on_test() {
    if [ "$run_ort" = true ] ; then
      python $OPTIMIZER_SCRIPT -m $1 $ONNX_EXPORT_OPTIONS $2 $3
      python $OPTIMIZER_SCRIPT -m $1 $BENCHMARK_OPTIONS $2 $3
    fi
    
    if [ "$run_torch" = true ] ; then
      python $OPTIMIZER_SCRIPT -e torch -m $1 $BENCHMARK_OPTIONS $2 $3
    fi
  
    if [ "$run_torchscript" = true ] ; then
      python $OPTIMIZER_SCRIPT -e torchscript -m $1 $BENCHMARK_OPTIONS $2 $3
    fi  
}

# -------------------------------------------
if [ "$run_gpu_fp32" = true ] ; then
  for m in $PRETRAINED_MODELS
  do
    echo "Run GPU FP32 Benchmark on model ${m}"
    run_on_test "${m}" -g
  done
fi

if [ "$run_gpu_fp16" = true ] ; then
  for m in $PRETRAINED_MODELS
  do
    echo "Run GPU FP16 Benchmark on model ${m}"
    run_on_test "${m}" -g --fp16
  done
fi

if [ "$run_cpu" = true ] ; then
  for m in $PRETRAINED_MODELS
  do
    echo "Run CPU Benchmark on model ${m}"
    run_on_test "${m}" 
  done
fi 


# Remove duplicated lines
awk '!x[$0]++' ./result.csv > summary_result.csv
awk '!x[$0]++' ./fusion.csv > summary_fusion.csv
awk '!x[$0]++' ./detail.csv > summary_detail.csv