/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#pragma once
#include <string>
#include <memory>
#include <map>
#include <utility>
#include "impl/activation_op_builder.h"
#include "impl/conv_op_builder.h"
#include "impl/elementwise_op_builder.h"
#include "impl/gemm_op_builder.h"
#include "impl/pool_op_builder.h"
#include "impl/qlinearconv_op_builder.h"
#include "impl/flatten_op_builder.h"
#include "impl/matmul_op_builder.h"
#include "impl/tensor_op_builder.h"
#include "impl/concat_op_builder.h"
#include "impl/softmax_op_builder.h"
#include "impl/norm_op_builder.h"
#include "impl/clip_op_builder.h"
#include "impl/reduce_op_builder.h"
#include "impl/quantize_op_builder.h"
#include "impl/dequantize_op_builder.h"
#include "impl/qlinearmatmul_op_builder.h"
#include "impl/qlinear_binary_op_builder.h"
#include "impl/qlinearconcat_op_builder.h"
#include "impl/gather_op_builder.h"
#include "impl/tile_op_builder.h"
#include "impl/squeeze_op_builder.h"
#include "impl/unsqueeze_op_builder.h"
#include "impl/resize_op_builder.h"
#include "impl/cast_op_builder.h"
#include "impl/dropout_op_builder.h"
#include "impl/slice_op_builder.h"
#include "impl/split_op_builder.h"
#include "impl/pad_op_builder.h"
namespace onnxruntime {
namespace vsi {
namespace npu {
using createIOpBuildItemFunc = std::function<std::unique_ptr<IOpBuilder>()>;
using OpBuildItemType = std::map<std::string, std::unique_ptr<IOpBuilder>>;

static const std::map<std::string, createIOpBuildItemFunc> reg = {
#define REGISTER_OP_BUILDER(ONNX_NODE_TYPE, BUILDER_TYPE) \
  {                                                       \
      ONNX_NODE_TYPE, [] { return std::make_unique<BUILDER_TYPE>(); }}

    REGISTER_OP_BUILDER("Add", AddOpBuilder),
    REGISTER_OP_BUILDER("Sub", SubOpBuilder),
    REGISTER_OP_BUILDER("Mul", MulOpBuilder),
    REGISTER_OP_BUILDER("Div", DivOpBuilder),
    REGISTER_OP_BUILDER("Abs", AbsOpBuilder),
    REGISTER_OP_BUILDER("Pow", PowOpBuilder),
    REGISTER_OP_BUILDER("Sqrt", SqrtOpBuilder),
    REGISTER_OP_BUILDER("Exp", ExpOpBuilder),
    REGISTER_OP_BUILDER("Floor", FloorOpBuilder),
    REGISTER_OP_BUILDER("Log", LogOpBuilder),
    REGISTER_OP_BUILDER("Sin", SinOpBuilder),
    REGISTER_OP_BUILDER("Conv", ConvOpBuilder),
    REGISTER_OP_BUILDER("Gemm", GemmOpBuilder),
    REGISTER_OP_BUILDER("Relu", ReluOpBuilder),
    REGISTER_OP_BUILDER("LeakyRelu", LeakyReluOpBuilder),
    REGISTER_OP_BUILDER("Tanh", TanhOpBuilder),
    REGISTER_OP_BUILDER("Sigmoid", SigmoidOpBuilder),
    REGISTER_OP_BUILDER("HardSigmoid", HardSigmoidOpBuilder),
    REGISTER_OP_BUILDER("HardSwish", HardSwishOpBuilder),
    REGISTER_OP_BUILDER("GlobalAveragePool", GlobalAveragePoolOpBuilder),
    REGISTER_OP_BUILDER("QLinearConv", QLinearConvOpBuilder),
    REGISTER_OP_BUILDER("Flatten", FlattenOpBuilder),
    REGISTER_OP_BUILDER("MatMul", MatMulOpBuilder),
    REGISTER_OP_BUILDER("GlobalMaxPool", GlobalMaxPoolOpBuilder),
    REGISTER_OP_BUILDER("AveragePool", AveragePoolOpBuilder),
    REGISTER_OP_BUILDER("MaxPool", MaxPoolOpBuilder),
    REGISTER_OP_BUILDER("Reshape", ReshapeOpBuilder),
    REGISTER_OP_BUILDER("Concat", ConcatOpBuilder),
    REGISTER_OP_BUILDER("Softmax", SoftmaxOpBuilder),
    REGISTER_OP_BUILDER("Transpose", TransposeOpBuilder),
    REGISTER_OP_BUILDER("BatchNormalization", BatchNormOpBuilder),
    REGISTER_OP_BUILDER("Clip", ClipOpBuilder),
    REGISTER_OP_BUILDER("ReduceMean", ReduceMeanOpBuilder),
    REGISTER_OP_BUILDER("QuantizeLinear", QuantizeLinearOpBuilder),
    REGISTER_OP_BUILDER("DequantizeLinear", DequantizeLinearOpBuilder),
    REGISTER_OP_BUILDER("QLinearMatMul", QLinearMatMulOpBuilder),
    REGISTER_OP_BUILDER("QLinearAdd", QLinearAddOpBuilder),
    REGISTER_OP_BUILDER("QLinearMul", QLinearMulOpBuilder),
    REGISTER_OP_BUILDER("QLinearConcat", QLinearConcatOpBuilder),
    REGISTER_OP_BUILDER("Gather", GatherOpBuilder),
    REGISTER_OP_BUILDER("Tile", TileOpBuilder),
    REGISTER_OP_BUILDER("Squeeze", SqueezeOpBuilder),
    REGISTER_OP_BUILDER("Unsqueeze", UnsqueezeOpBuilder),
    REGISTER_OP_BUILDER("Resize", ResizeOpBuilder),
    REGISTER_OP_BUILDER("Cast", CastOpBuilder),
    REGISTER_OP_BUILDER("Dropout", DropoutOpBuilder),
    REGISTER_OP_BUILDER("Slice", SliceOpBuilder),
    REGISTER_OP_BUILDER("Split", SplitOpBuilder),
    REGISTER_OP_BUILDER("Neg", NegOpBuilder),
    REGISTER_OP_BUILDER("Not", NotOpBuilder),
    REGISTER_OP_BUILDER("Ceil", CeilOpBuilder),
    REGISTER_OP_BUILDER("Round", RoundOpBuilder),
    REGISTER_OP_BUILDER("Min", MinOpBuilder),
    REGISTER_OP_BUILDER("Max", MaxOpBuilder),
    REGISTER_OP_BUILDER("Pad", PadOpBuilder)
#undef REGISTER_OP_BUILDER
};

template <typename T>
struct OpBuildConstructor {
  T supported_builtins;
  OpBuildConstructor(
      const std::map<typename T::key_type, createIOpBuildItemFunc> reg) {
    LOGS_DEFAULT(INFO) << "Initialize supported ops";
    for (const auto& kv : reg) {
      supported_builtins.insert(std::make_pair(kv.first, kv.second()));
    }
  }
};

inline const OpBuildItemType& SupportedBuiltinOps() {
  static OpBuildConstructor<OpBuildItemType> c(reg);
  return c.supported_builtins;
}
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
