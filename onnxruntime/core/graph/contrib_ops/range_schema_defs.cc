// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "range_schema_defs.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include <cmath>
#include <type_traits>

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::OPTIONAL_VALUE;
using ::ONNX_NAMESPACE::OpSchema;
using ::ONNX_NAMESPACE::InferenceContext;
using ::ONNX_NAMESPACE::TensorShapeProto;
using ::ONNX_NAMESPACE::TensorProto;
using ::ONNX_NAMESPACE::TensorProto_DataType;

// This Doc based on LSTM_ver7, and modification
static const char* Range_ver1_doc = R"DOC(
Creates a sequence of numbers that begins at `start` and extends by increments of `delta`
up to but not including `limit`.
)DOC";

template <typename T>
static T get_data(const TensorProto*) {
    fail_shape_inference("Unsupported non-raw-data data type!");
}

template <>
int32_t get_data<int32_t>(const TensorProto* shapeInitializer) {
    if (shapeInitializer->int32_data_size() > 0) return shapeInitializer->int32_data(0);
    fail_shape_inference("Can not get shape initializer data!");
}

template <>
int64_t get_data<int64_t>(const TensorProto* shapeInitializer) {
    if (shapeInitializer->int64_data_size() > 0) return shapeInitializer->int64_data(0);
    fail_shape_inference("Can not get shape initializer data!");
}

template <>
float get_data<float>(const TensorProto* shapeInitializer) {
    if (shapeInitializer->float_data_size() > 0) return shapeInitializer->float_data(0);
    fail_shape_inference("Can not get shape initializer data!");
}

template <>
double get_data<double>(const TensorProto* shapeInitializer) {
    if (shapeInitializer->double_data_size() > 0) return shapeInitializer->double_data(0);
    fail_shape_inference("Can not get shape initializer data!");
}

template <typename T>
static T GetFirstElement(const TensorProto* shapeInitializer) {
    if (shapeInitializer == nullptr) return T{1};

    if (utils::HasRawData(*shapeInitializer)) {
        const std::string& bytes = shapeInitializer->raw_data();
        return *reinterpret_cast<const T*>(bytes.c_str());
    }
    return get_data<T>(shapeInitializer);
}

template <typename T>
static int64_t CalcRangeDim(const TensorProto* startShapeInitializer,
                            const TensorProto* limitShapeInitializer,
                            const TensorProto* deltaShapeInitializer) {
    auto start = static_cast<double>(GetFirstElement<T>(startShapeInitializer));
    auto limit = static_cast<double>(GetFirstElement<T>(limitShapeInitializer));
    auto delta = static_cast<double>(GetFirstElement<T>(deltaShapeInitializer));
    if (delta == 0) {
        fail_shape_inference("delta in Range operator can not be zero!");
    }
    return static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
}

static int64_t CalcResultDim(const TensorProto* startShapeInitializer,
                             const TensorProto* limitShapeInitializer,
                             const TensorProto* deltaShapeInitializer,
                             int dtype) {
    int64_t dim = -1LL;    
    if (dtype == TensorProto::FLOAT) {
        dim = CalcRangeDim<float>(startShapeInitializer, limitShapeInitializer, deltaShapeInitializer);
    }
    else if (dtype == TensorProto::INT32) {
        dim = CalcRangeDim<int32_t>(startShapeInitializer, limitShapeInitializer, deltaShapeInitializer);
    }
    else if (dtype == TensorProto::INT64) {
        dim = CalcRangeDim<int64_t>(startShapeInitializer, limitShapeInitializer, deltaShapeInitializer);
    }
    else if (dtype == TensorProto::INT16) {
        dim = CalcRangeDim<int16_t>(startShapeInitializer, limitShapeInitializer, deltaShapeInitializer);
    }
    else if (dtype == TensorProto::DOUBLE) {
        dim = CalcRangeDim<double>(startShapeInitializer, limitShapeInitializer, deltaShapeInitializer);
    }
    else {
        fail_shape_inference("Unsupported type:", dtype);
    }
    return dim;
}

OpSchema& RegisterRangeOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)", "tensor(int16)", "tensor(int32)", "tensor(int64)" },
        "Constrain input and output types.")
    .Input(
        0,
        "start",
        "Tensor(scalar, or dims=[1]). First entry in the range.",
        "T")
    .Input(
        1,
        "limit",
        "Tensor(scalar, or dims=[1]). Upper limit of sequence, exclusive.",
        "T")
    .Input(
        2,
        "delta",
        "Tensor(scalar, or dims=[1]). Number that increments start. Defaults to 1.",
        "T",
        OpSchema::Optional)
    .Output(
        0,
        "Y",
        "1-D Tensor of the range.",
        "T")
    .SetDoc(Range_ver1_doc)
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        TensorShapeProto::Dimension dim;
        bool enoughShapeInfo = ctx.getInputData(0) != nullptr 
            && ctx.getInputData(1) != nullptr 
            && (ctx.getNumInputs() == 2 || ctx.getInputData(2) != nullptr);
        if (enoughShapeInfo) {
            const TensorProto* startShapeInitializer = ctx.getInputData(0);
            const TensorProto* limitShapeInitializer = ctx.getInputData(1);
            const TensorProto* deltaShapeInitializer = (ctx.getNumInputs() > 2) ? ctx.getInputData(2) : nullptr;
            const auto& startTensorType = ctx.getInputType(0)->tensor_type();
            int dtype = startTensorType.elem_type();

            int64_t n = CalcResultDim(startShapeInitializer, limitShapeInitializer, deltaShapeInitializer, dtype);
            dim.set_dim_value(n);
        } // else unknown 1-D tensor

        updateOutputShape(ctx, 0, {dim});
    });
}

}  // namespace contrib
}  // namespace onnxruntime
