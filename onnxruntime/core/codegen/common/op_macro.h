// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
namespace onnxruntime {

#define LIST_BINARY_OPS() \
  BINARY_OP(Add)          \
  BINARY_OP(Div)          \
  BINARY_OP(Mul)          \
  BINARY_OP(PRelu)        \
  BINARY_OP(Sub)

#define LIST_BINARY_CMP_OPS() \
  BINARY_CMP_OP(Equal)        \
  BINARY_CMP_OP(Greater)      \
  BINARY_CMP_OP(Less)

#define LIST_POOL_OPS()  \
  POOL_OP(MaxPool)       \
  POOL_OP(AveragePool)   \
  POOL_OP(GlobalMaxPool) \
  POOL_OP(GlobalAveragePool)

#define LIST_REDUCE_OPS()    \
  REDUCE_INDEXED_OP(ArgMax)  \
  REDUCE_INDEXED_OP(ArgMin)  \
  REDUCE_OP(ReduceL1)        \
  REDUCE_OP(ReduceL2)        \
  REDUCE_OP(ReduceLogSum)    \
  REDUCE_OP(ReduceLogSumExp) \
  REDUCE_OP(ReduceMax)       \
  REDUCE_OP(ReduceMean)      \
  REDUCE_OP(ReduceMin)       \
  REDUCE_OP(ReduceProd)      \
  REDUCE_OP(ReduceSum)       \
  REDUCE_OP(ReduceSumSquare)

#define LIST_UNARY_OPS()       \
  UNARY_OP(Abs)                \
  UNARY_OP(Affine)             \
  UNARY_OP(Ceil)               \
  UNARY_OP(Elu)                \
  UNARY_OP(Exp)                \
  UNARY_OP(Floor)              \
  UNARY_OP(HardSigmoid)        \
  UNARY_OP(LeakyRelu)          \
  UNARY_OP(Log)                \
  UNARY_OP(Neg)                \
  UNARY_OP(ParametricSoftplus) \
  UNARY_OP(Reciprocal)         \
  UNARY_OP(Relu)               \
  UNARY_OP(ScaledTanh)         \
  UNARY_OP(Selu)               \
  UNARY_OP(Sigmoid)            \
  UNARY_OP(Softplus)           \
  UNARY_OP(Softsign)           \
  UNARY_OP(Sqrt)               \
  UNARY_OP(Tanh)               \
  UNARY_OP(ThresholdedRelu)

#define LIST_VARIADIC_OPS() \
  VARIADIC_OP(Max)          \
  VARIADIC_OP(Min)          \
  VARIADIC_OP(Sum)

#define LIST_ALL_GENERIC_OPS() \
  LIST_BINARY_OPS()            \
  LIST_BINARY_CMP_OPS()        \
  LIST_REDUCE_OPS()            \
  LIST_POOL_OPS()              \
  LIST_UNARY_OPS()             \
  LIST_VARIADIC_OPS()          \
  ADD_OP_ITEM(Cast)            \
  ADD_OP_ITEM(Clip)            \
  ADD_OP_ITEM(Concat)          \
  ADD_OP_ITEM(Conv)            \
  ADD_OP_ITEM(Crop)            \
  ADD_OP_ITEM(Dropout)         \
  ADD_OP_ITEM(Expand)          \
  ADD_OP_ITEM(Flatten)         \
  ADD_OP_ITEM(Gather)          \
  ADD_OP_ITEM(GatherElements)  \
  ADD_OP_ITEM(Gemm)            \
  ADD_OP_ITEM(Identity)        \
  ADD_OP_ITEM(LogSoftmax)      \
  ADD_OP_ITEM(LSTM)            \
  ADD_OP_ITEM(MatMul)          \
  ADD_OP_ITEM(MatMulInteger)   \
  ADD_OP_ITEM(Pad)             \
  ADD_OP_ITEM(Reshape)         \
  ADD_OP_ITEM(Shape)           \
  ADD_OP_ITEM(Slice)           \
  ADD_OP_ITEM(Softmax)         \
  ADD_OP_ITEM(Split)           \
  ADD_OP_ITEM(Squeeze)         \
  ADD_OP_ITEM(Transpose)       \
  ADD_OP_ITEM(Unsqueeze)       \
  ADD_OP_ITEM(Where)

}  // namespace onnxruntime
