# This script is self-contained and you need not copy other files in this directory when you set run_cli=true.
# Please install PyTorch 1.5.0 (see https://pytorch.org/) before running this benchmark.

run_install=true

# Change it false if you want to run the script from master branch instead of published package.
run_cli=true

# Engines
run_ort=true
run_torch=false
run_torchscript=true

# Devices (You can run either CPU or GPU, but not both at the same time: gpu need onnxruntime-gpu, and CPU need onnxruntime.
run_gpu_fp32=true
run_gpu_fp16=true
run_cpu=false

# If you have mutliple GPUs, you can choose one GPU for test. Here is an example of using the second GPU:
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
#export BENCHMARK_OPTIONS="-b 1 4 -s 8 16 32 64 128 256 512 1024 -t 1000 -f fusion.csv -r result.csv -d detail.csv"
export BENCHMARK_OPTIONS="-b 1 2 4 8 16 32 64 128 -s 8 64 128 -t 100 -f fusion.csv -r result.csv -d detail.csv"

if [ "$run_gpu_fp32" = true ] ; then
  for m in bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
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
  for m in bert-base-cased
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
  for m in bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
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