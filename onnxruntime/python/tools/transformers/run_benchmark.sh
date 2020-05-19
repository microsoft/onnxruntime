# Please install PyTorch 1.5.0 (see https://pytorch.org/) before running this benchmark.

# When run_cli=true, this script is self-contained and you need not copy other files to run benchmarks - it will use onnxruntime-tools package.
# If run_cli=false, it depends on other python script (*.py) files in this directory.
run_cli=true


run_install=true


# Engines to test.
run_ort=true
run_torch=false
run_torchscript=true

# Devices to test (You can run either CPU or GPU, but not both: gpu need onnxruntime-gpu, and CPU need onnxruntime).
run_gpu_fp32=true
run_gpu_fp16=true
run_cpu=false

# Pretrained transformers models to test
export PRETRAINED_MODELS="bert-base-cased roberta-base gpt2"
#export PRETRAINED_MODELS="bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased"


# If you have mutliple GPUs, you can choose one GPU for test. Here is an example to use the second GPU:
# export CUDA_VISIBLE_DEVICES=1

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

export ONNX_EXPORT_OPTIONS="-o -v -b 0 --overwrite -f fusion.csv"

# Please choose or update one combination of batch sizes and sequence lengths below
export BENCHMARK_OPTIONS="-b 1 4 -s 8 16 32 64 128 256 512 1024 -t 1000 -f fusion.csv -r result.csv -d detail.csv"
#export BENCHMARK_OPTIONS="-b 1 2 4 8 16 32 64 128 -s 8 128 -t 100 -f fusion.csv -r result.csv -d detail.csv"


if [ "$run_gpu_fp32" = true ] ; then
  for m in $PRETRAINED_MODELS
  do
    echo "Run GPU FP32 Benchmark on model ${m}"
     
    export BENCHMARK_MODEL="${m}"
  
    if [ "$run_ort" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -m $BENCHMARK_MODEL $ONNX_EXPORT_OPTIONS
      python $OPTIMIZER_SCRIPT -g -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS  
      python $OPTIMIZER_SCRIPT -g -o -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS 
    
      python $OPTIMIZER_SCRIPT -g -m $BENCHMARK_MODEL $ONNX_EXPORT_OPTIONS --fp16 
      python $OPTIMIZER_SCRIPT -g -o -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS --fp16       
    fi
    
    if [ "$run_torch" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -e torch -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS
      python $OPTIMIZER_SCRIPT -g -e torch -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS --fp16
    fi
  
    if [ "$run_torchscript" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -e torchscript -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS
      python $OPTIMIZER_SCRIPT -g -e torchscript -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS --fp16
    fi  
  done
fi

if [ "$run_gpu_fp16" = true ] ; then
  for m in $PRETRAINED_MODELS
  do
    echo "Run GPU FP16 Benchmark on model ${m}"
     
    export BENCHMARK_MODEL="${m}"
  
    if [ "$run_ort" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -m $BENCHMARK_MODEL $ONNX_EXPORT_OPTIONS --fp16
      python $OPTIMIZER_SCRIPT -g -o -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS --fp16       
    fi
    
    if [ "$run_torch" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -e torch -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS --fp16
    fi
  
    if [ "$run_torchscript" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -e torchscript -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS --fp16
    fi  
  done
fi

if [ "$run_cpu" = true ] ; then
  for m in $PRETRAINED_MODELS
  do
    echo "Run CPU Benchmark on model ${m}"
     
    export BENCHMARK_MODEL="${m}"
  
    if [ "$run_ort" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -m $BENCHMARK_MODEL $ONNX_EXPORT_OPTIONS
      python $OPTIMIZER_SCRIPT -g -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS
      python $OPTIMIZER_SCRIPT -g -o -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS    
    fi
    
    if [ "$run_torch" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -e torch -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS
    fi
  
    if [ "$run_torchscript" = true ] ; then
      python $OPTIMIZER_SCRIPT -g -e torchscript -m $BENCHMARK_MODEL $BENCHMARK_OPTIONS
    fi  
  done
fi 


# Remove duplicated lines
awk '!x[$0]++' ./result.csv > summary_result.csv
awk '!x[$0]++' ./fusion.csv > summary_fusion.csv
awk '!x[$0]++' ./detail.csv > summary_detail.csv