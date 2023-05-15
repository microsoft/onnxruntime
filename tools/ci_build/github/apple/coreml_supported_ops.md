<!--
Keep in sync with doco generated from /docs/execution-providers/CoreML-ExecutionProvider.md on the gh_pages branch
-->
|Operator|Note|
|--------|------|
|ai.onnx:Add||
|ai.onnx:ArgMax||
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:BatchNormalization||
|ai.onnx:Cast||
|ai.onnx:Clip||
|ai.onnx:Concat||
|ai.onnx:Conv|Only 1D/2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:DepthToSpace|Only DCR mode DepthToSpace is supported.|
|ai.onnx:Div||
|ai.onnx:Flatten||
|ai.onnx:Gemm|Input B should be constant.|
|ai.onnx:GlobalAveragePool|Only 2D Pool is supported.|
|ai.onnx:GlobalMaxPool|Only 2D Pool is supported.|
|ai.onnx:LeakyRelu||
|ai.onnx:LRN||
|ai.onnx:MatMul|Input B should be constant.|
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:Mul||
|ai.onnx:Pad|Only constant mode and last two dim padding is supported.<br/>Input pads and constant_value should be constant.<br/>If provided, axes should be constant.|
|ai.onnx:Pow|Only supports cases when both inputs are fp32.|
|ai.onnx:PRelu|Input slope should be constant.<br/>Input slope should either have shape [C, 1, 1] or have 1 element.|
|ai.onnx:Relu||
|ai.onnx:Reshape||
|ai.onnx:Resize||
|ai.onnx:Sigmoid||
|ai.onnx:Squeeze||
|ai.onnx:Sub||
|ai.onnx:Tanh||
|ai.onnx:Transpose||
