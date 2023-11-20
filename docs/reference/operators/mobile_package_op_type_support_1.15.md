---
title: ORT 1.15 Mobile Package Operators
parent: Operators
grand_parent: Reference
nav_exclude: true
---

# ONNX Runtime Mobile Pre-Built Package Operator and Type Support

## Supported operators and types

The supported operators and types are based on what is required to support float32 and quantized versions of popular models. The full list of input models used to determine this list is available [here](https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/android/mobile_package.required_operators.readme.txt)

## Supported data input types

  - float
  - int8_t
  - uint8_t

NOTE: Operators used to manipulate dimensions and indices will support int32 and int64.

## Supported Operators

|Operator|Opsets|
|--------|------|
|**ai.onnx**||
|ai.onnx:Abs|12, 13, 14, 15|
|ai.onnx:Add|12, 13, 14, 15|
|ai.onnx:And|12, 13, 14, 15|
|ai.onnx:ArgMax|12, 13, 14, 15|
|ai.onnx:ArgMin|12, 13, 14, 15|
|ai.onnx:AveragePool|12, 13, 14, 15|
|ai.onnx:Cast|12, 13, 14, 15|
|ai.onnx:Ceil|12, 13, 14, 15|
|ai.onnx:Clip|12, 13, 14, 15|
|ai.onnx:Concat|12, 13, 14, 15|
|ai.onnx:ConstantOfShape|12, 13, 14, 15|
|ai.onnx:Conv|12, 13, 14, 15|
|ai.onnx:ConvTranspose|12, 13, 14, 15|
|ai.onnx:Cos|12, 13, 14, 15|
|ai.onnx:CumSum|12, 13, 14, 15|
|ai.onnx:DepthToSpace|12, 13, 14, 15|
|ai.onnx:DequantizeLinear|12, 13, 14, 15|
|ai.onnx:Div|12, 13, 14, 15|
|ai.onnx:DynamicQuantizeLinear|12, 13, 14, 15|
|ai.onnx:Elu|12, 13, 14, 15|
|ai.onnx:Equal|12, 13, 14, 15|
|ai.onnx:Erf|12, 13, 14, 15|
|ai.onnx:Exp|12, 13, 14, 15|
|ai.onnx:Expand|12, 13, 14, 15|
|ai.onnx:Flatten|12, 13, 14, 15|
|ai.onnx:Floor|12, 13, 14, 15|
|ai.onnx:Gather|12, 13, 14, 15|
|ai.onnx:GatherND|12, 13, 14, 15|
|ai.onnx:Gemm|12, 13, 14, 15|
|ai.onnx:GlobalAveragePool|12, 13, 14, 15|
|ai.onnx:Greater|12, 13, 14, 15|
|ai.onnx:GreaterOrEqual|12, 13, 14, 15|
|ai.onnx:HardSigmoid|12, 13, 14, 15|
|ai.onnx:Identity|12, 13, 14, 15|
|ai.onnx:If|12, 13, 14, 15|
|ai.onnx:InstanceNormalization|12, 13, 14, 15|
|ai.onnx:LRN|12, 13, 14, 15|
|ai.onnx:LayerNormalization|1|
|ai.onnx:LeakyRelu|12, 13, 14, 15|
|ai.onnx:Less|12, 13, 14, 15|
|ai.onnx:LessOrEqual|12, 13, 14, 15|
|ai.onnx:Log|12, 13, 14, 15|
|ai.onnx:LogSoftmax|12, 13, 14, 15|
|ai.onnx:Loop|12, 13, 14, 15|
|ai.onnx:MatMul|12, 13, 14, 15|
|ai.onnx:MatMulInteger|12, 13, 14, 15|
|ai.onnx:Max|12, 13, 14, 15|
|ai.onnx:MaxPool|12, 13, 14, 15|
|ai.onnx:Mean|12, 13, 14, 15|
|ai.onnx:Min|12, 13, 14, 15|
|ai.onnx:Mul|12, 13, 14, 15|
|ai.onnx:Neg|12, 13, 14, 15|
|ai.onnx:NonMaxSuppression|12, 13, 14, 15|
|ai.onnx:NonZero|12, 13, 14, 15|
|ai.onnx:Not|12, 13, 14, 15|
|ai.onnx:Or|12, 13, 14, 15|
|ai.onnx:PRelu|12, 13, 14, 15|
|ai.onnx:Pad|12, 13, 14, 15|
|ai.onnx:Pow|12, 13, 14, 15|
|ai.onnx:QLinearConv|12, 13, 14, 15|
|ai.onnx:QLinearMatMul|12, 13, 14, 15|
|ai.onnx:QuantizeLinear|12, 13, 14, 15|
|ai.onnx:Range|12, 13, 14, 15|
|ai.onnx:Reciprocal|12, 13, 14, 15|
|ai.onnx:ReduceMax|12, 13, 14, 15|
|ai.onnx:ReduceMean|12, 13, 14, 15|
|ai.onnx:ReduceMin|12, 13, 14, 15|
|ai.onnx:ReduceProd|12, 13, 14, 15|
|ai.onnx:ReduceSum|12, 13, 14, 15|
|ai.onnx:Relu|12, 13, 14, 15|
|ai.onnx:Reshape|12, 13, 14, 15|
|ai.onnx:Resize|12, 13, 14, 15|
|ai.onnx:ReverseSequence|12, 13, 14, 15|
|ai.onnx:Round|12, 13, 14, 15|
|ai.onnx:Scan|12, 13, 14, 15|
|ai.onnx:ScatterND|12, 13, 14, 15|
|ai.onnx:Shape|12, 13, 14, 15|
|ai.onnx:Sigmoid|12, 13, 14, 15|
|ai.onnx:Sin|12, 13, 14, 15|
|ai.onnx:Size|12, 13, 14, 15|
|ai.onnx:Slice|12, 13, 14, 15|
|ai.onnx:Softmax|12, 13, 14, 15|
|ai.onnx:SpaceToDepth|12, 13, 14, 15|
|ai.onnx:Split|12, 13, 14, 15|
|ai.onnx:Sqrt|12, 13, 14, 15|
|ai.onnx:Squeeze|12, 13, 14, 15|
|ai.onnx:Sub|12, 13, 14, 15|
|ai.onnx:Sum|12, 13, 14, 15|
|ai.onnx:Tanh|12, 13, 14, 15|
|ai.onnx:ThresholdedRelu|12, 13, 14, 15|
|ai.onnx:Tile|12, 13, 14, 15|
|ai.onnx:TopK|12, 13, 14, 15|
|ai.onnx:Transpose|12, 13, 14, 15|
|ai.onnx:Unique|12, 13, 14, 15|
|ai.onnx:Unsqueeze|12, 13, 14, 15|
|ai.onnx:Where|12, 13, 14, 15|
|||
|**com.microsoft**||
|com.microsoft:DynamicQuantizeMatMul|1|
|com.microsoft:FusedConv|1|
|com.microsoft:FusedGemm|1|
|com.microsoft:FusedMatMul|1|
|com.microsoft:Gelu|1|
|com.microsoft:MatMulIntegerToFloat|1|
|com.microsoft:NhwcMaxPool|1|
|com.microsoft:QLinearAdd|1|
|com.microsoft:QLinearAveragePool|1|
|com.microsoft:QLinearConv|1|
|com.microsoft:QLinearGlobalAveragePool|1|
|com.microsoft:QLinearLeakyRelu|1|
|com.microsoft:QLinearMul|1|
|com.microsoft:QLinearSigmoid|1|
|||
