<!--
Keep in sync with doco generated from /docs/execution-providers/CoreML-ExecutionProvider.md on the gh_pages branch
-->
|Operator|Note|
|--------|------|
|ai.onnx:Add||
|ai.onnx:AveragePool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:Clip||
|ai.onnx:Conv|Only 1D/2D Conv is supported.<br/>Bias if provided must be constant.|
|ai.onnx:Div||
|ai.onnx:Gemm|Input B must be constant.|
|ai.onnx:GlobalAveragePool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:GlobalMaxPool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:MatMul|Only support for transA == 0, alpha == 1.0 and beta == 1.0 is currently implemented.|
|ai.onnx:MaxPool|Only 2D Pool is supported currently. 3D and 5D support can be added if needed.|
|ai.onnx:Mul||
|ai.onnx:Pow|Only supports cases when both inputs are fp32.|
|ai.onnx:Relu||
|ai.onnx:Reshape||
|ai.onnx:Sub||
