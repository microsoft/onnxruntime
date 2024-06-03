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
#include <memory>
#include <string>
#include <vector>
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/node_unit.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
namespace util {

tim::vx::DataType OnnxDtypeToTIMVXDtype(const int32_t dtype);

tim::vx::DataType OnnxDtypeToTIMVXDtype(const ONNX_NAMESPACE::DataType type);

tim::vx::ShapeType OnnxShapeToTIMVXShape(const onnxruntime::TensorShape& ts);

std::string PrintNode(const onnxruntime::NodeArg& node_arg);

std::string PrintNode(const std::vector<int64_t> shape);

size_t GetTensorElementSize(const ONNXTensorElementDataType type);

size_t GetTensorBytes(const Ort::TensorTypeAndShapeInfo& info);

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg);

std::shared_ptr<uint8_t> UnpackTensor(
    const NodeArg* node, const ONNX_NAMESPACE::TensorProto& initializer);

tim::vx::PadType GetPadType(const std::string type);

int32_t ReverseAxis(int32_t origin_axis, int32_t length);

std::vector<int32_t> ReverseAxis(std::vector<int32_t> origin_axes, int32_t length);

bool IsTypeSupported(const NodeArg* node_arg);

enum class QuantizedOpType : uint8_t {
  Unknown,  // Unknown or not a quantized NodeUnit
  DequantizeLinear,
  QuantizeLinear,
  QLinearConv,
  QLinearMatMul,
  QLinearAdd,
  QLinearSigmoid,
  QLinearAveragePool,
  QLinearMul,
  // Not yet supported
  // QLinearReduceMean,
  QDQConv,
  QDQResize,
  QDQAveragePool,
  QDQAdd,
  QDQMul,
  QDQTranspose,
  QDQReshape,
  QDQSoftmax,
  QDQConcat,
  QDQGemm,
  QDQMatMul,
  // TODO, add other QDQ NodeUnit types
};

enum class ConvType : uint8_t {
  Regular,
  Depthwise,
  Grouped,
};
QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit);

ConvType GetConvType(const NodeUnit& node_unit, const InitializedTensorSet& initializers);

// If this is a quantized Conv (QLinearConv or QDQConv)
bool IsQuantizedConv(QuantizedOpType quant_op_type);

// If this is a quantized Pool (QLinearAveragePool or QDQAveragePool)
bool IsQuantizedPool(QuantizedOpType quant_op_type);

// If this is a quantized Gemm (QLinearMatMul or QDQMatMul/QDQGemm)
bool IsQuantizedGemm(QuantizedOpType quant_op_type);

// This quantized op is an operator or qdq node unit takes 2 inputs and produces 1 output
// Such as QLinearConv, QLinearMatMul, QLinearAdd, QDQConv,...
bool IsQuantizedBinaryOp(QuantizedOpType quant_op_type);

// Check if a qlinear binary op has valid inputs, Qlinear[Conv/MatMul/Add]
bool HasValidBinaryOpQuantizedInputTypes(const NodeUnit& node_unit);

void GetQuantizationScaleAndZeroPoint(
    const InitializedTensorSet& initializers, const NodeUnitIODef& io_def, const Path& model_path,
    float& scale, int32_t& zero_point,
    std::optional<std::vector<float>>& pcq_scales,
    std::optional<std::vector<int32_t>>& pcq_zps);

bool GetType(const NodeArg& node_arg, int32_t& type);

}  // namespace util
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
