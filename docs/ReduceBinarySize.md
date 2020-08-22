# Reduce binary size
To reduce compiled binary size, two options are available:

- --include_ops_by_model=<path to directory of models\>
- --include_ops_by_file=<path to a file\>

The options empower building to comment out operators listed in execution provider(s), thereby downsizing the output.

## include_ops_by_model
The argument enables the compile binary of including only operators consumed by models in the specified directory.

## include_ops_by_file
The argument enables the compiled binary of including only operators referred. The file has format like:
```
#domain;opset;op1,op2...
ai.onnx;1;MemcpyToHost,MemcpyFromHost
ai.onnx;11;Gemm
```

## More usage tips
- By default, the trimming happens only on cpu execution provider, with --use_cuda it will also be applied to cuda;
- If both are specified, operators referred from either argument will be kept active;
- The script is located under toos/ci_build/, and could go solo to apply to cpu and cuda providers as:
```
python exclude_unused_ops.py --model_path d:\ReduceSize\models --file_path d:\ReduceSize\ops.txt --ort_root d:\onnxruntime
```