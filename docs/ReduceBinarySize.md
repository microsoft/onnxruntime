# Reduce binary size
To reduce compiled binary size, two arguments are available:

- --include_ops_by_model=<path to directory of models\>
- --include_ops_by_file=<path to a json file\>

The arguments empower building script of commenting out operators listed in execution provider(s), thereby downsizing build output.

## include_ops_by_model
The argument enables the compile binary of including only operators consumed by models in the specified directory.

## include_ops_by_file
The argument enables the compiled binary of including only operators referred. Json file has format like:
```
{
    "kOnnxDomain":
    {
        "1":
        [
            "MemcpyToHost",
            "MemcpyFromHost"
        ],
        "11":
        [
            "Gemm"
        ]
    }
}
```

## More usage tips
- By default, the trimming happens only on cpu execution provider, with --use_cuda it will also be applied to cuda;
- If both are specified, operators referred from either argument will be kept active;
- The script is located under toos/ci_build/, and could go solo as:
```
python provider_rewriter.py --model_path d:\ReduceSize\models --file_path d:\ReduceSize\ops.json --ep_path d:\onnxruntime\onnxruntime\core\providers\cpu\cpu_execution_provider.cc
```