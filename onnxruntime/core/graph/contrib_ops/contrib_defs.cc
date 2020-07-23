// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/attn_lstm_schema_defs.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/nchwc_schema_defs.h"
#include "core/graph/contrib_ops/range_schema_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"
#include "core/mlas/inc/mlas.h"

template<typename... args>
struct StaticArray {
    static constexpr typename std::common_type<args...>::type data[] = {args::make...};
};

template<typename... T>
constexpr auto static_array() {
    return StaticArray<T...>::data;
}

template<const char* const *... args>
struct StaticStrings {
    static constexpr const char * const data[] = {*args...};
};

template<const char * const *... S>
constexpr auto static_strings() {
    return StaticStrings<S...>::data;
}

namespace ONNX_NAMESPACE {

template<typename U>
struct span {
    class iterator {
    public:
        iterator(const U *ptr): ptr(ptr) {}
        iterator operator++() { ++ptr; return *this; }
        bool operator!=(const iterator& other) const { return ptr != other.ptr; }
        const U& operator*() const { return *ptr; }
    private:
        U* ptr;
    };

    const U *data;
    size_t n;

    constexpr span() : data(), n() {}

    template <size_t N>
    constexpr span(const U (&arr)[N])
    : data(arr), n(N) {}

    iterator begin() const { return iterator(data); }
    iterator end() const { return iterator(data + n); }
    size_t size() const { return n; }
    const U& operator[](size_t i) const { return data[i]; }
};

struct FormalParameter {
    constexpr FormalParameter(
            const char* name,
            const char*,
            const char* type_str,
            OpSchema::FormalParameterOption param_option = OpSchema::FormalParameterOption::Single,
            bool is_homogeneous = true,
            int min_arity = 1) :
            name(name),
            type_str(type_str),
            param_option(param_option),
            is_homogeneous(is_homogeneous),
            min_arity(min_arity)
    {}

    const char* name;
    const char* type_str;
    OpSchema::FormalParameterOption param_option;
    bool is_homogeneous;
    int min_arity;
};

struct Attribute {
    constexpr Attribute(
            const char* name,
            const char*,
            AttributeProto::AttributeType type,
            bool required = true)
            : name_(name),
              type_(type),
              required_(required) {}

    constexpr Attribute(
            const char* name,
            const char*,
            AttributeProto::AttributeType type,
            float /*default_value*/)
            : name_(name),
              type_(type) {
        // TODO default_value
    }

    constexpr Attribute(
            const char* name,
            const char*,
            AttributeProto::AttributeType type,
            int64_t /*default_value*/)
            : name_(name),
              type_(type) {
        // TODO default_value
    }

    const char* name_;
    AttributeProto::AttributeType type_;
    bool required_{};

    ONNX_NAMESPACE::OpSchema::Attribute toAttribute() const {
        return ONNX_NAMESPACE::OpSchema::Attribute(name_, "", type_, required_);
    }
};

struct TypeConstraint {
    constexpr TypeConstraint(
            const char* type_param_str_,
            span<const char* const> allowed_type_strs_,
            const char*)
            : type_param_str(type_param_str_),
              allowed_type_strs(allowed_type_strs_) {}

    const char* type_param_str;
    span<const char * const> allowed_type_strs;

    constexpr static auto TENSOR_INT8 = "tensor(int8)";
    constexpr static auto TENSOR_UINT8 = "tensor(uint8)";
    constexpr static auto TENSOR_INT16 = "tensor(int16)";
    constexpr static auto TENSOR_UINT16 = "tensor(uint16)";
    constexpr static auto TENSOR_INT32 = "tensor(int32)";
    constexpr static auto TENSOR_UINT32 = "tensor(int32)";
    constexpr static auto TENSOR_INT64 = "tensor(int64)";
    constexpr static auto TENSOR_UINT64 = "tensor(int64)";
    constexpr static auto TENSOR_FLOAT = "tensor(float)";
    constexpr static auto TENSOR_FLOAT16 = "tensor(float16)";
    constexpr static auto TENSOR_DOUBLE = "tensor(double)";
    constexpr static auto TENSOR_STRING = "tensor(string)";
    constexpr static auto TENSOR_BOOL = "tensor(bool)";
    constexpr static auto TENSOR_COMPLEX64 = "tensor(complex64)";
    constexpr static auto TENSOR_COMPLEX128 = "tensor(complex128)";
};
#define AllTensorTypes StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_DOUBLE, &ONNX_NAMESPACE::TypeConstraint::TENSOR_STRING, &ONNX_NAMESPACE::TypeConstraint::TENSOR_BOOL, &ONNX_NAMESPACE::TypeConstraint::TENSOR_COMPLEX64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_COMPLEX128>::data
#define NumericTypesForMathReduction StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_DOUBLE>::data

class CxOpSchema final {
public:
    constexpr CxOpSchema(const char* name, const char* file, int line)
            : name_(name),
              file_(file),
              line_(line) {}

    constexpr CxOpSchema& SetDomain(const char* domain) {
        domain_ = domain;
        return *this;
    }

    constexpr CxOpSchema& SetDoc(const char* doc) {
        doc_ = doc;
        return *this;
    }

    constexpr CxOpSchema& SinceVersion(OperatorSetVersion n) {
        since_version_ = n;
        return *this;
    }

    constexpr CxOpSchema& SetSupportLevel(OpSchema::SupportType supportType) {
        support_ = supportType;
        return *this;
    }

    constexpr CxOpSchema& Attrs(span<const Attribute> attributes) {
        attributes_ = attributes;
        return *this;
    }

    constexpr CxOpSchema& Inputs(span<const FormalParameter> inputs) {
        inputs_ = inputs;
        return *this;
    }

    constexpr CxOpSchema& Outputs(span<const FormalParameter> outputs) {
        outputs_ = outputs;
        return *this;
    }

    constexpr CxOpSchema& TypeConstraints(span<const TypeConstraint> type_constraints) {
        type_constraints_ = type_constraints;
        return *this;
    }

    using InferenceFunctionP = void(*)(InferenceContext&);
    constexpr CxOpSchema& TypeAndShapeInferenceFunction(const InferenceFunctionP inference_function) {
        inference_function_ = inference_function;
        return *this;
    }

    constexpr CxOpSchema& Deprecate() {
        // TODO
        return *this;
    }

    const char * name_;
    const char * file_;
    const char * doc_ = "";
    const char * domain_ = ONNX_DOMAIN;
    span<const FormalParameter> inputs_;
    span<const FormalParameter> outputs_;
    span<const Attribute> attributes_;
    span<const TypeConstraint> type_constraints_;
    bool allows_unchecked_attributes_ = false;
    int line_ = 0;
    OpSchema::SupportType support_{};
    int min_input_ = 0;
    int max_input_ = 0;
    int min_output_ = 0;
    int max_output_ = 0;
    OperatorSetVersion since_version_ = 1;
    InferenceFunctionP inference_function_{};
};

void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
void matmulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx);

void convTransposeWithDynamicPadsShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;  // Input tensor should have at least two dimensions.
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - size_t{2});

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      return;
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      return;
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  } else {
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] =
        (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;

  // Infer output shape if 'pads' tensor is available
  const auto* pads_initializer = ctx.getInputData(2);
  if (nullptr == pads_initializer) {
    return;
  }

  if (pads_initializer->dims_size() != 1 ||
      pads_initializer->data_type() != TensorProto::INT64)
    fail_shape_inference(
        "'pads' input must be a 1D (shape: [2 * n_input_dims]) tensor of type int64");

  pads = ParseData<int64_t>(pads_initializer);

  if (pads.size() != static_cast<size_t>(2 * n_input_dims))
    fail_shape_inference("Pads has incorrect number of values");

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) {  // Added only to one side.
      return;
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto final_output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1) *
      group;  // channels should be the second dim of second input multiply
              // group.

  int size_of_output;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
          // TODO: throw exception?
          return;  // output shape value cannot be smaller than the input shape
                   // value
        }
      }
      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim =
            strides[i] * (input_shape.dim(i + 2).dim_value() - 1) +
            output_padding[i] + effective_kernel_shape[i] - pads[i] -
            pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}

constexpr const char* operator "" _docstring(const char* string, size_t) {
#ifdef __ONNX_NO_DOC_STRINGS
    (void)string;
    return "";
#else
    return string;
#endif
}
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;
using ONNX_NAMESPACE::operator "" _docstring;

void ValidateTypeAndShapeForScaleAndZP(ONNX_NAMESPACE::InferenceContext& ctx, int index, ::google::protobuf::int32 expectedType, bool isScalar, int expectedTensorSize = 0) {
  if (ctx.getNumInputs() > static_cast<size_t>(index)) {
    auto data_type = ctx.getInputType(index);
    if (nullptr == data_type) {
      fail_type_inference("Input data type does not match the expected data type");
    }
    if (data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
        data_type->tensor_type().elem_type() != expectedType) {
      fail_type_inference(
          "Input data type does not match the expected data type. Current data type is ", data_type->tensor_type().elem_type());
    }
  }

  if (hasInputShape(ctx, index)) {
    ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(index)->tensor_type().shape();
    if (isScalar) {
      if (shape.dim_size() != 0) {
        fail_type_inference("Scale and Zero-point must be a scalar");
      }
    } else {
      if (shape.dim_size() != 1) {
        fail_type_inference("Scale and Zero-point must be of rank 1");
      }

      if (shape.dim((int)0).has_dim_value() && shape.dim((int)0).dim_value() != expectedTensorSize) {
        fail_type_inference(
            "Scale and Zero-point must be of rank 1 and the number of elements should be equal to the number of rows of the corresponding input.");
      }
    }
  }
}

std::function<void(OpSchema&)> QLinearMathDocGenerator(const char* name, const char* additionalDocumentation) {
#ifdef __ONNX_NO_DOC_STRINGS
  (void)name;
  (void)additionalDocumentation;
#endif
  return [=](OpSchema& schema) {
#ifndef __ONNX_NO_DOC_STRINGS
    std::string doc = R"DOC(
Performs element-wise binary {name} on 8 bit data types (with Numpy-style broadcasting support).

{additionalDocumentation}
)DOC";
    ONNX_NAMESPACE::ReplaceAll(doc, "{name}", name);
    ONNX_NAMESPACE::ReplaceAll(doc, "{additionalDocumentation}", additionalDocumentation);
    schema.SetDoc(doc);
#endif
    schema.Input(0, "A", "First operand."_docstring, "T");
    schema.Input(
        1,
        "A_scale",
        "Input A's scale. It's a scalar, which means a per-tensor/layer quantization."_docstring,
        "tensor(float)");
    schema.Input(
        2,
        "A_zero_point",
        "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring,
        "T",
        OpSchema::Optional);
    schema.Input(3, "B", "Second operand.", "T");
    schema.Input(
        4,
        "B_scale",
        "Input B's scale. It's a scalar, which means a per-tensor/layer quantization."_docstring,
        "tensor(float)");
    schema.Input(
        5,
        "B_zero_point",
        "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring,
        "T",
        OpSchema::Optional);
    schema.Input(
        6,
        "C_scale",
        "Output scale. It's a scalar, which means a per-tensor/layer quantization."_docstring,
        "tensor(float)");
    schema.Input(
        7,
        "C_zero_point",
        "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring,
        "T",
        OpSchema::Optional);
    schema.Output(0, "C", "Result, has same element type as two inputs"_docstring, "T");
    schema.TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"}, "Constrain input and output types to 8 bit signed and unsigned tensors."_docstring);
    schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      auto a_type = ctx.getInputType(0);
      auto b_type = ctx.getInputType(3);

      if (nullptr == a_type || nullptr == b_type ||
          a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
          b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
      }

      // validate scale and zero points
      ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 2, a_type->tensor_type().elem_type(), true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 4, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 5, b_type->tensor_type().elem_type(), true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 6, ONNX_NAMESPACE::TensorProto::FLOAT, true);
      ValidateTypeAndShapeForScaleAndZP(ctx, 7, a_type->tensor_type().elem_type(), true);

      if (hasInputShape(ctx, 0) && hasInputShape(ctx, 3))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(3)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

constexpr static ONNX_NAMESPACE::Attribute AttentionAttributes[] {
        { "num_heads", "Number of attention heads", AttributeProto::INT},
};
constexpr static ONNX_NAMESPACE::FormalParameter AttentionInputs[] {
        ONNX_NAMESPACE::FormalParameter("input", "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * head_size", "T"),
        ONNX_NAMESPACE::FormalParameter("weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "T"),
        ONNX_NAMESPACE::FormalParameter("bias", "1D input tensor with shape (3 * hidden_size)", "T"),
        ONNX_NAMESPACE::FormalParameter("mask_index", "Attention mask with shape (batch_size, past_sequence_length + sequence_length), or index with shape (batch_size) or (2 * batch_size).", "M", OpSchema::Optional),
        ONNX_NAMESPACE::FormalParameter("past", "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).", "T", OpSchema::Optional)
};
constexpr static ONNX_NAMESPACE::FormalParameter AttentionOutputs[] {
        ONNX_NAMESPACE::FormalParameter("output", "3D output tensor with shape (batch_size, append_length, hidden_size)", "T"),
        ONNX_NAMESPACE::FormalParameter("present", "present state for key and value with shape (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)", "T", OpSchema::Optional)
};
constexpr static ONNX_NAMESPACE::TypeConstraint AttentionTypeConstraints[] {
        ONNX_NAMESPACE::TypeConstraint("T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16>::data, "Constrain input and output types to float tensors."),
        ONNX_NAMESPACE::TypeConstraint("M", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32>::data, "Constrain mask index to integer types")
};
static void AttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (ctx.getNumOutputs() > 1) {
        propagateElemTypeFromInputToOutput(ctx, 0, 1);
    }

    if (hasInputShape(ctx, 0)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);

        if (ctx.getNumOutputs() > 1) {
            auto& input_shape = getInputShape(ctx, 0);
            auto& input_dims = input_shape.dim();
            if (input_dims.size() != 3) {
                fail_shape_inference("Inputs 0 shall be 3 dimensions");
            }

            if (hasInputShape(ctx, 4)) {
                auto& past_shape = getInputShape(ctx, 4);
                auto& past_dims = past_shape.dim();
                if (past_dims.size() != 5) {
                    fail_shape_inference("Inputs 4 shall be 5 dimensions");
                }

                if (past_dims[3].has_dim_value() && input_dims[1].has_dim_value()) {
                    auto all_sequence_length = past_shape.dim(3).dim_value() + input_shape.dim(1).dim_value();

                    ONNX_NAMESPACE::TensorShapeProto present_shape;
                    for (auto& dim : past_dims) {
                        *present_shape.add_dim() = dim;
                    }
                    present_shape.mutable_dim(3)->set_dim_value(all_sequence_length);

                    updateOutputShape(ctx, 1, present_shape);
                }
            }
        }
    }
};

constexpr static ONNX_NAMESPACE::Attribute QAttentionAttributes[] {
        { "num_heads", "Number of attention heads"_docstring, AttributeProto::INT },
        { "unidirectional", "Whether every token can only attend to previous tokens. Default value is 0."_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
};
constexpr static ONNX_NAMESPACE::FormalParameter QAttentionInputs[] {
        ONNX_NAMESPACE::FormalParameter("input", "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * head_size"_docstring, "T1"),
        ONNX_NAMESPACE::FormalParameter("weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)"_docstring, "T2"),
        ONNX_NAMESPACE::FormalParameter("bias", "1D input tensor with shape (3 * hidden_size)"_docstring, "T3"),
        ONNX_NAMESPACE::FormalParameter("input_scale", "scale of quantized input tensor. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T3"),
        ONNX_NAMESPACE::FormalParameter("weight_scale", "scale of weight scale. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T3"),
        ONNX_NAMESPACE::FormalParameter("mask_index", "Attention mask index with shape (batch_size)"_docstring, "T4", OpSchema::Optional),
        ONNX_NAMESPACE::FormalParameter("input_zero_point", "zero point of quantized input tensor.It's a scalar, which means a per-tensor/layer quantization."_docstring, "T1", OpSchema::Optional),
        ONNX_NAMESPACE::FormalParameter("weight_zero_point", "zero point of quantized weight tensor. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T2", OpSchema::Optional),
        ONNX_NAMESPACE::FormalParameter("past", "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size)."_docstring, "T3", OpSchema::Optional)
};
constexpr static ONNX_NAMESPACE::FormalParameter QAttentionOutputs[] {
        ONNX_NAMESPACE::FormalParameter("output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)"_docstring, "T3"),
        ONNX_NAMESPACE::FormalParameter("present", "present state for key and value with shape (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)"_docstring, "T3", OpSchema::Optional)
};
constexpr static ONNX_NAMESPACE::TypeConstraint QAttentionTypeConstraints[] {
        ONNX_NAMESPACE::TypeConstraint("T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input and output types to int8 tensors."_docstring),
        ONNX_NAMESPACE::TypeConstraint("T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input and output types to int8 tensors."_docstring),
        ONNX_NAMESPACE::TypeConstraint("T3", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16>::data, "Constrain input and output types to float tensors."_docstring),
        ONNX_NAMESPACE::TypeConstraint("T4", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32>::data, "Constrain mask index to integer types"_docstring)
};
static void QAttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    // Type inference
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 2, 0);

    // Shape inference
    // if the input shape doesn't exist, further shape inference is not possible
    if (!hasNInputShapes(ctx, 1)) {
        return;
    }

    ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 0);
}

constexpr static ONNX_NAMESPACE::Attribute EmbedLayerNormalizationAttributes[] {
        { "epsilon", "The epsilon value to use to avoid division by zero."_docstring, AttributeProto::FLOAT, kDefaultEmbedLayerNormEpsilon },
};
constexpr static ONNX_NAMESPACE::FormalParameter EmbedLayerNormalizationInputs[] {
        ONNX_NAMESPACE::FormalParameter("input_ids", "2D words IDs with shape (batch_size, sequence_length)"_docstring, "T1"),
        ONNX_NAMESPACE::FormalParameter("segment_ids", "2D segment IDs with shape (batch_size, sequence_length)"_docstring, "T1"),
        ONNX_NAMESPACE::FormalParameter("word_embedding", "2D with shape (,hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("position_embedding", "2D with shape (, hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("segment_embedding", "2D with shape (, hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("gamma", "1D gamma tensor for layer normalization with shape (hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("beta", "1D beta tensor for layer normalization  with shape (hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("mask", "2D attention mask with shape (batch_size, sequence_length)"_docstring, "T1", OpSchema::Optional)
};
constexpr static ONNX_NAMESPACE::FormalParameter EmbedLayerNormalizationOutputs[] {
        ONNX_NAMESPACE::FormalParameter("output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("mask_index", "1D mask_index tensor with shape (batch_size)"_docstring, "T1")
};
constexpr static ONNX_NAMESPACE::TypeConstraint EmbedLayerNormalizationTypeConstraints[] {
        ONNX_NAMESPACE::TypeConstraint("T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32>::data, "Constrain input and output integer tensors types"_docstring),
        ONNX_NAMESPACE::TypeConstraint("T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16>::data, "Constrain input and output float tensors types."_docstring)
};
static void EmbedLayerNormalizationTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 2, 0);
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
    if (!hasInputShape(ctx, 0))
        return;

    auto& input_ids_shape = getInputShape(ctx, 0);
    auto& input_ids_dims = input_ids_shape.dim();

    // Note that both batch size and sequence length could be symbolic.
    // So we only check dimension size here.
    if (input_ids_dims.size() != 2) {
        fail_shape_inference("Inputs 0 shall be 2 dimensions");
    }

    // get hidden_size from the last dimension of embedding
    auto& word_embedding_shape = getInputShape(ctx, 3);
    auto& word_embedding_dims = word_embedding_shape.dim();
    if (word_embedding_dims.size() != 2 ||
        !word_embedding_dims[1].has_dim_value() ||
        word_embedding_shape.dim(1).dim_value() <= 0) {
        fail_shape_inference("word_embedding should have 2 dimensions and dimension size is known.");
    }
    int64_t hidden_size = word_embedding_shape.dim(1).dim_value();

    // input shape is (batch_size, sequence_length), output shape is (batch_size, sequence_length, hidden_size)
    ONNX_NAMESPACE::TensorShapeProto output_shape;
    for (auto& dim : input_ids_dims) {
        *output_shape.add_dim() = dim;
    }
    output_shape.add_dim();
    output_shape.mutable_dim(2)->set_dim_value(hidden_size);

    updateOutputShape(ctx, 0, output_shape);

    // mask_index shape is (batch_size)
    ONNX_NAMESPACE::TensorShapeProto mask_index_shape;
    *mask_index_shape.add_dim() = input_ids_dims[0];
    updateOutputShape(ctx, 1, mask_index_shape);
}

constexpr static ONNX_NAMESPACE::FormalParameter FastGeluInputs[] {
        ONNX_NAMESPACE::FormalParameter("X", "input tensor"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("bias", "bias tensor"_docstring, "T", OpSchema::Optional)
};
constexpr static ONNX_NAMESPACE::FormalParameter FastGeluOutputs[] {
        ONNX_NAMESPACE::FormalParameter("Y", "output tensor"_docstring, "T")
};
constexpr static ONNX_NAMESPACE::TypeConstraint FastGeluTypeConstraints[] {
        ONNX_NAMESPACE::TypeConstraint("T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16>::data, "Constrain input and output types to float or half tensors."_docstring)
};

constexpr static ONNX_NAMESPACE::Attribute SkipLayerNormalizationAttributes[] {
        { "epsilon", "The epsilon value to use to avoid division by zero."_docstring, AttributeProto::FLOAT, kDefaultSkipLayerNormEpsilon },
};
constexpr static ONNX_NAMESPACE::FormalParameter SkipLayerNormalizationInputs[] {
        ONNX_NAMESPACE::FormalParameter("input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("skip", "3D skip tensor with shape (batch_size, sequence_length, hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("gamma", "1D input tensor with shape (hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("beta", "1D skip tensor with shape (hidden_size"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("bias", "1D bias tensor with shape (hidden_size"_docstring, "T", OpSchema::Optional),
};
constexpr static ONNX_NAMESPACE::FormalParameter SkipLayerNormalizationOutputs[] {
        ONNX_NAMESPACE::FormalParameter("output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)"_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("mean", "Saved mean used during training to speed up gradient computation"_docstring, "U", OpSchema::Optional),
        ONNX_NAMESPACE::FormalParameter("inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation."_docstring, "U", OpSchema::Optional),
};
constexpr static ONNX_NAMESPACE::TypeConstraint SkipLayerNormalizationTypeConstraints[] {
        ONNX_NAMESPACE::TypeConstraint("T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16>::data, "Constrain input and output types to float or half tensors."_docstring),
        ONNX_NAMESPACE::TypeConstraint("U", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT>::data, "Constrain mean and inv_std_var to float tensors."_docstring)
};

static constexpr ONNX_NAMESPACE::CxOpSchema BertSchemas[] {
    ONNX_NAMESPACE::CxOpSchema("Attention", "", 0)
        .SetDomain(onnxruntime::kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .Attrs(AttentionAttributes)
        .Inputs(AttentionInputs)
        .Outputs(AttentionOutputs)
        .TypeConstraints(AttentionTypeConstraints)
        .TypeAndShapeInferenceFunction(AttentionTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("QAttention", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .Attrs(QAttentionAttributes)
        .Inputs(QAttentionInputs)
        .Outputs(QAttentionOutputs)
        .TypeConstraints(QAttentionTypeConstraints)
        .TypeAndShapeInferenceFunction(QAttentionTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("EmbedLayerNormalization", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .Attrs(EmbedLayerNormalizationAttributes)
        .Inputs(EmbedLayerNormalizationInputs)
        .Outputs(EmbedLayerNormalizationOutputs)
        .TypeConstraints(EmbedLayerNormalizationTypeConstraints)
        .TypeAndShapeInferenceFunction(EmbedLayerNormalizationTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("FastGelu", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .Inputs(FastGeluInputs)
        .Outputs(FastGeluOutputs)
        .TypeConstraints(FastGeluTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("SkipLayerNormalization", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .Attrs(SkipLayerNormalizationAttributes)
        .Inputs(SkipLayerNormalizationInputs)
        .Outputs(SkipLayerNormalizationOutputs)
        .TypeConstraints(SkipLayerNormalizationTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
};

void Register(const ONNX_NAMESPACE::CxOpSchema& cx_schema) {
    ONNX_NAMESPACE::OpSchema schema(cx_schema.name_, "", 0);
    schema
        .SetDomain(cx_schema.domain_)
        .SinceVersion(cx_schema.since_version_)
        .SetSupportLevel(cx_schema.support_);
    for (const auto& attr : cx_schema.attributes_) {
        schema.Attr(attr.toAttribute());
    }
    for (size_t i = 0; i < cx_schema.inputs_.size(); ++i) {
        const auto& input = cx_schema.inputs_[i];
        schema.Input(i, input.name, "", input.type_str, input.param_option, input.is_homogeneous, input.min_arity);
    }
    for (size_t i = 0; i < cx_schema.outputs_.size(); ++i) {
        const auto& output = cx_schema.outputs_[i];
        schema.Output(i, output.name, "", output.type_str, output.param_option, output.is_homogeneous, output.min_arity);
    }
    for (const auto& constrain : cx_schema.type_constraints_) {
        std::vector<std::string> allowed_types;
        allowed_types.reserve(constrain.allowed_type_strs.size());
        for (const char * type : constrain.allowed_type_strs) {
            allowed_types.emplace_back(type);
        }
        schema.TypeConstraint(constrain.type_param_str, allowed_types, "");
    }
    schema.TypeAndShapeInferenceFunction(cx_schema.inference_function_);
    ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce{schema};
}

void RegisterBertSchemas() {
    for (auto& cx_schema : BertSchemas) {
        Register(cx_schema);
    }
}

constexpr static ONNX_NAMESPACE::Attribute AffineAttributes[] {
        { "alpha", "Value of alpha"_docstring, AttributeProto::FLOAT, 1.0f},
        { "beta", "Value of beta"_docstring, AttributeProto::FLOAT, 0.0f},
};
constexpr static ONNX_NAMESPACE::FormalParameter AffineInputs[] {
        ONNX_NAMESPACE::FormalParameter("X", "1D input tensor"_docstring, "T"),
};
constexpr static ONNX_NAMESPACE::FormalParameter AffineOutputs[] {
        ONNX_NAMESPACE::FormalParameter("Y", "1D output tensor"_docstring, "T"),
};
constexpr static ONNX_NAMESPACE::TypeConstraint FloatingPointTTypeConstraints[] {
    {"T",
        StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_DOUBLE>::data,
        "Constrain input and output types to float tensors."_docstring},
};

constexpr static ONNX_NAMESPACE::Attribute ParametricSoftplusAttributes[] {
        { "alpha", "Value of alpha"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE},
        { "beta", "Value of beta"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE},
};
constexpr static ONNX_NAMESPACE::FormalParameter ParametricSoftplusInputs[] {
        ONNX_NAMESPACE::FormalParameter("X", "1D input tensor"_docstring, "T"),
};
constexpr static ONNX_NAMESPACE::FormalParameter ParametricSoftplusOutputs[] {
        ONNX_NAMESPACE::FormalParameter("Y", "1D output tensor"_docstring, "T"),
};

constexpr static ONNX_NAMESPACE::Attribute ImageScalerAttributes[] {
        { "bias", "Bias applied to each channel, same size as C."_docstring, AttributeProto::FLOATS, OPTIONAL_VALUE},
        { "scale", "The scale to apply."_docstring, AttributeProto::FLOAT, 1.0f},
};
constexpr static ONNX_NAMESPACE::FormalParameter ImageScalerInputs[] {
        ONNX_NAMESPACE::FormalParameter("input", "Input tensor of shape [N,C,H,W]"_docstring, "T"),
};
constexpr static ONNX_NAMESPACE::FormalParameter ImageScalerOutputs[] {
        ONNX_NAMESPACE::FormalParameter("output", "Result, has same shape and type as input"_docstring, "T"),
};

constexpr static ONNX_NAMESPACE::Attribute CropAttributes[] {
        { "border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder)."_docstring, AttributeProto::INTS, OPTIONAL_VALUE},
        { "scale", "A 1-D values of (height, width)."_docstring, AttributeProto::INTS, OPTIONAL_VALUE},
};
constexpr static ONNX_NAMESPACE::FormalParameter CropInputs[] {
        ONNX_NAMESPACE::FormalParameter("input", "Input tensor of shape [N,C,H,W]"_docstring, "T"),
};
constexpr static ONNX_NAMESPACE::FormalParameter CropOutputs[] {
        ONNX_NAMESPACE::FormalParameter("output", "Result, has same type as input, with H and W dimensions reduced."_docstring, "T"),
};

constexpr static ONNX_NAMESPACE::Attribute ThresholdedReluAttributes[] {
        { "alpha", "Threshold value"_docstring, AttributeProto::FLOAT, 1.0f},
};
constexpr static ONNX_NAMESPACE::FormalParameter ThresholdedReluInputs[] {
        ONNX_NAMESPACE::FormalParameter("X", "Input tensor"_docstring, "T"),
};
constexpr static ONNX_NAMESPACE::FormalParameter ThresholdedReluOutputs[] {
        ONNX_NAMESPACE::FormalParameter("Y", "Output tensor"_docstring, "T"),
};

constexpr static ONNX_NAMESPACE::FormalParameter DynamicSliceInputs[] {
        ONNX_NAMESPACE::FormalParameter("data", "Tensor of data to extract slices from."_docstring, "T"),
        ONNX_NAMESPACE::FormalParameter("starts", "1-D tensor of starting indices of corresponding axis in `axes`"_docstring, "Tind"),
        ONNX_NAMESPACE::FormalParameter("ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes"_docstring, "Tind"),
        ONNX_NAMESPACE::FormalParameter("axes", "1-D tensor of axes that `starts` and `ends` apply to."_docstring, "Tind", OpSchema::Optional),
};
constexpr static ONNX_NAMESPACE::FormalParameter DynamicSliceOutputs[] {
        ONNX_NAMESPACE::FormalParameter("output", "Sliced data tensor."_docstring, "T"),
};
constexpr static ONNX_NAMESPACE::TypeConstraint DynamicSliceTypeConstraints[] {
        ONNX_NAMESPACE::TypeConstraint("T", AllTensorTypes, "Constrain input and output types to all tensor types."_docstring),
        ONNX_NAMESPACE::TypeConstraint("Tind", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT64>::data, "Constrain indices to integer types"_docstring),
};

constexpr static ONNX_NAMESPACE::Attribute GivenTensorFillAttributes[] {
        { "values", "", AttributeProto::FLOATS, OPTIONAL_VALUE },
        { "shape", "", AttributeProto::INTS, OPTIONAL_VALUE },
        { "input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE },
        { "extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter GivenTensorFillInputs[] {
        { "shape", "The shape of filled tensor"_docstring, "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter GivenTensorFillOutputs[] {
        { "X", "The filled tensor"_docstring, "T" },
};
static void GivenTensorFillTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (ctx.getAttribute("shape") != nullptr) {
        propagateShapeFromAttributeToOutput(ctx, "shape", 0);
        return;
    }
    // The type constraints above do not allow for input_as_shape
    // and may need to be fixed.
    if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
        return;
    std::vector<int64_t> extra_shape;
    getRepeatedAttribute(ctx, "extra_shape", extra_shape);
    if (hasInputShape(ctx, 0)) {
        ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
        for (auto extra_dim_val : extra_shape) {
            if (extra_dim_val < 0)
                fail_shape_inference(
                        "Negative values are not allowed in a shape specification");
            shape.add_dim()->set_dim_value(extra_dim_val);
        }
        updateOutputShape(ctx, 0, shape);
    }
}

static constexpr const char* Scale_ver1_doc = R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.)DOC"_docstring;
constexpr static ONNX_NAMESPACE::Attribute ScaleAttributes[] {
        { "scale", "The scale to apply."_docstring, AttributeProto::FLOAT, 1.0f },
};
constexpr static ONNX_NAMESPACE::FormalParameter ScaleInputs[] {
        { "input", "Input data to be scaled"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ScaleOutputs[] {
        { "output", "Output data after scaling"_docstring, "T" },
};

static constexpr const char* GRUUnit_ver1_doc = R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].)DOC"_docstring;
constexpr static ONNX_NAMESPACE::Attribute GRUUnitAttributes[] {
        { "drop_states",
            "Bool to determine if hidden state is zeroes or passed "
            "along for timesteps past the given sequence_length."_docstring,
            AttributeProto::INT, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter GRUUnitInputs[] {
        { "hidden_prev", "The previous GRU hidden state.", "T" },
        {
        "gates",
         "Unactivated gate outputs from forget, update, "
         "and output gates, pre-activation."_docstring,
        "T"
        },
        {
            "seq_lengths",
            "Array of sequence lengths. "
            "len(seq_lengths) should equal batch size N."_docstring,
            "T"
        },
        { "t", "The timestep for this operation."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter GRUUnitOutputs[] {
        { "hidden",
                "The new GRU hidden state calculated by this op."_docstring,
                "T" },
};

constexpr static ONNX_NAMESPACE::Attribute GivenTensorFill10Attributes[] {
        { "values", "", AttributeProto::FLOATS, OPTIONAL_VALUE },
        { "shape", "", AttributeProto::INTS, OPTIONAL_VALUE },
        { "input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE },
        { "extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter GivenTensorFill10Inputs[] {
        { "shape", "The shape of filled tensor"_docstring, "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter GivenTensorFill10Outputs[] {
        { "X", "The filled tensor"_docstring, "T" },
};
static void GivenTensorFill10TypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (ctx.getAttribute("shape") != nullptr) {
        propagateShapeFromAttributeToOutput(ctx, "shape", 0);
        return;
    }
    // The type constraints above do not allow for input_as_shape
    // and may need to be fixed.
    if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
        return;
    std::vector<int64_t> extra_shape;
    getRepeatedAttribute(ctx, "extra_shape", extra_shape);
    if (hasInputShape(ctx, 0)) {
        ONNX_NAMESPACE::TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
        for (auto extra_dim_val : extra_shape) {
            if (extra_dim_val < 0)
                fail_shape_inference(
                        "Negative values are not allowed in a shape specification");
            shape.add_dim()->set_dim_value(extra_dim_val);
        }
        updateOutputShape(ctx, 0, shape);
    }
}

constexpr static ONNX_NAMESPACE::Attribute Scale10Attributes[] {
        { "scale", "The scale to apply."_docstring, AttributeProto::FLOAT, 1.0f },
};
constexpr static ONNX_NAMESPACE::FormalParameter Scale10Inputs[] {
        { "input", "Input data to be scaled"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter Scale10Outputs[] {
        { "output", "Output data after scaling"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute GRUUnit10Attributes[] {
        { "drop_states",
                "Bool to determine if hidden state is zeroes or passed "
                "along for timesteps past the given sequence_length."_docstring,
                AttributeProto::INT, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter GRUUnit10Inputs[] {
        { "hidden_prev", "The previous GRU hidden state."_docstring, "T" },
        { "gates",
                         "Unactivated gate outputs from forget, update, "
                         "and output gates, pre-activation."_docstring,
                                                                     "T" },
        { "seq_lengths",
                         "Array of sequence lengths.  "
                         "len(seq_lengths) should equal batch size N."_docstring,
                                                                     "T" },
        { "t", "The timestep for this operation."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter GRUUnit10Outputs[] {
        { "hidden", "The new GRU hidden state calculated by this op."_docstring, "T" },
};

static constexpr const char* MeanVarienceNormalizationDoc = R"DOC(Perform mean variance normalization.)DOC"_docstring;
constexpr static ONNX_NAMESPACE::Attribute MeanVarienceNormalizationAttributes[] {
    { "across_channels", "If 1, mean and variance are computed across channels. Default is 0."_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
    { "normalize_variance", "If 0, normalize the mean only.  Default is 1."_docstring, AttributeProto::INT, static_cast<int64_t>(1) },
};
constexpr static ONNX_NAMESPACE::FormalParameter MeanVarienceNormalizationInputs[] {
    { "input", "Input tensor of shape [N,C,H,W]"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter MeanVarienceNormalizationOutputs[] {
    { "output", "Result, has same shape and type as input"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute ScaledTanhAttributes[] {
    { "alpha", "Scaling value"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE },
    { "beta", "Scaling value"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter ScaledTanhInputs[] {
    { "input", "Input tensor"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ScaledTanhOutputs[] {
    { "output", "The scaled hyperbolic tangent values of the input tensor computed element-wise"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute Affine10Attributes[] {
    { "alpha", "Value of alpha"_docstring, AttributeProto::FLOAT, 1.0f },
    { "beta", "Value of beta"_docstring, AttributeProto::FLOAT, 0.0f },
};
constexpr static ONNX_NAMESPACE::FormalParameter Affine10Inputs[] {
    { "X", "1D input tensor"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter Affine10Outputs[] {
    { "Y", "1D output tensor"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute ParametricSoftplus10Attributes[] {
    { "alpha", "Value of alpha"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE },
    { "beta", "Value of beta"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter ParametricSoftplus10Inputs[] {
    { "X", "1D input tensor"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ParametricSoftplus10Outputs[] {
    { "Y", "1D input tensor"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute ImageScaler10Attributes[] {
    { "bias", "Bias applied to each channel, same size as C."_docstring, AttributeProto::FLOATS, OPTIONAL_VALUE },
    { "scale", "The scale to apply."_docstring, AttributeProto::FLOAT, 1.0f },
};
constexpr static ONNX_NAMESPACE::FormalParameter ImageScaler10Inputs[] {
    { "input", "Input tensor of shape [N,C,H,W]"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ImageScaler10Outputs[] {
    { "output", "Result, has same shape and type as input"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute Crop10Attributes[] {
    { "border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder)."_docstring, AttributeProto::INTS },
    { "scale", "A 1-D values of (height, width)."_docstring, AttributeProto::INTS, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter Crop10Inputs[] {
    { "input", "Input tensor of shape [N,C,H,W]"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter Crop10Outputs[] {
    { "output", "Result, has same type as input, with H and W dimensions reduced."_docstring, "T" },
};
static void Crop10TypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
  // Type inference
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (ONNX_NAMESPACE::hasNInputShapes(ctx, 1)) {
    const auto& input_shape =
        ctx.getInputType(0)->tensor_type().shape();
    const auto input_rank =
        input_shape.dim_size();
    if (input_rank != 4)
      fail_shape_inference("Input's shape must be 4-D");

    // parse necessary attributes for futher processing
    std::vector<int64_t> border;
    bool border_present =
        getRepeatedAttribute(ctx, "border", border);
    if (!border_present || border.size() != 4)
      fail_shape_inference(
          "'Border' attribute must be present and must contain exactly 4 values - "
          "(left_border, top_border, right_border, bottom_border)");

    std::vector<int64_t> scale;
    bool scale_present =
        getRepeatedAttribute(ctx, "scale", scale);
    if (scale_present && scale.size() != 2)
      fail_shape_inference("'Scale' must contain exactly 2 values - (height, width)");

    // actual shape inference processing
    // [N, C] can be copied over from the input as is
    *output_shape->mutable_dim(static_cast<int>(0)) = input_shape.dim(static_cast<int>(0));
    *output_shape->mutable_dim(static_cast<int>(1)) = input_shape.dim(static_cast<int>(1));

    // process 'H' and 'W'
    if (!utils::HasDimValue(input_shape.dim(static_cast<int>(2))) ||
        !utils::HasDimValue(input_shape.dim(static_cast<int>(3)))) {
      // either height and width input has symbolic dims, so can't proceed further
      // add two dims as placeholders for output_H and output_W and return
      output_shape->add_dim();
      output_shape->add_dim();
      return;
    }

    int64_t H = input_shape.dim(static_cast<int>(2)).dim_value();
    int64_t W = input_shape.dim(static_cast<int>(3)).dim_value();

    int64_t left_border = border[0],
        top_border = border[1],
        right_border = border[2],
        bottom_border = border[3];

    if (H < top_border + bottom_border)
      fail_shape_inference("Input's height (", H,
                           ") needs to be greater than or equal to "
                           "the top_border (",
                           top_border, ") + bottom_border (", bottom_border, ")");

    if (W < left_border + right_border)
      fail_shape_inference("Input's width (", W,
                           ") needs to be greater than or equal to "
                           "the left_border (",
                           left_border, ") + right_border (", right_border, ")");

    int64_t bottom_limit = H - bottom_border;
    int64_t right_limit = W - right_border;

    // scale = (height, width)
    if (!scale.empty()) {
      bottom_limit = top_border + scale[0];
      right_limit = left_border + scale[1];

      if (H < bottom_limit)
        fail_shape_inference("Input's height (", H, ") needs to be greater than or equal to the top_border (", top_border, ") + scale[0] (", scale[0], ")");

      if (W < right_limit)
        fail_shape_inference("Input's width (", W, ") needs to be greater than or equal to the left_border (", left_border, ") + scale[1] (", scale[1], ")");
    }

    auto* h_output_dim = output_shape->add_dim();
    h_output_dim->set_dim_value(bottom_limit - top_border);

    auto* w_output_dim = output_shape->add_dim();
    w_output_dim->set_dim_value(right_limit - left_border);

  } else {
    // Rank Inference at the very least
    // (We know that the output is going to be 4-D)
    for (int i = 0; i < 4; ++i) {
      output_shape->add_dim();
    }
  }
}

constexpr static ONNX_NAMESPACE::Attribute DynamicSlice10Attributes[] {
    { "border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder)."_docstring, AttributeProto::INTS },
    { "scale", "A 1-D values of (height, width)."_docstring, AttributeProto::INTS, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter DynamicSlice10Inputs[] {
    { "data", "Tensor of data to extract slices from."_docstring, "T" },
    { "starts", "1-D tensor of starting indices of corresponding axis in `axes`"_docstring, "Tind" },
    { "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes"_docstring, "Tind" },
    { "axes", "1-D tensor of axes that `starts` and `ends` apply to."_docstring, "Tind", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter DynamicSlice10Outputs[] {
    { "output", "Sliced data tensor."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint DynamicSlice10TypeConstraints[] {
    { "T", AllTensorTypes, "Constrain input and output types to all tensor types."_docstring },
    { "Tind", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT64>::data, "Constrain indices to integer types"_docstring },
};

constexpr static ONNX_NAMESPACE::Attribute ScaledTanh10Attributes[] {
    { "alpha", "Scaling value"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE },
    { "beta", "Scaling value"_docstring, AttributeProto::FLOAT, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter ScaledTanh10Inputs[] {
    { "input", "Input tensor"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ScaledTanh10Outputs[] {
    { "output", "The scaled hyperbolic tangent values of the input tensor computed element-wise"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::FormalParameter SampleOpInputs[] {
    { "X", "input"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter SampleOpOutputs[] {
    { "Y", "output"_docstring, "T" },
};
static constexpr const char* SampleOpDoc = R"DOC(Sample echo operator.)DOC";
constexpr static ONNX_NAMESPACE::TypeConstraint SampleOpTypeConstraints[] {
    { "T", NumericTypesForMathReduction, "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type."_docstring }
};

constexpr static ONNX_NAMESPACE::Attribute MaxpoolWithMaskAttributes[]{
    {"auto_pad", "", AttributeProto::STRING, "NOTSET"},
    {"kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE},
    {"pads", "", AttributeProto::INTS, OPTIONAL_VALUE},
    {"storage_order", "", AttributeProto::INT, static_cast<int64_t>(0)},
    {"strides", "", AttributeProto::INTS, OPTIONAL_VALUE},
};
constexpr static ONNX_NAMESPACE::FormalParameter MaxpoolWithMaskInputs[] {
    { "X", "", "T" },
    { "M", "mask", "tensor(int32)" },
};
constexpr static ONNX_NAMESPACE::FormalParameter MaxpoolWithMaskOutputs[] {
    { "Y", "", "T" },
};
static void MaxpoolWithMaskTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    ONNX_NAMESPACE::convPoolShapeInference(ctx, false, true, 0, 1);
}

constexpr static ONNX_NAMESPACE::Attribute RfftAttributes[]{
    { "signal_ndim", "", AttributeProto::INT },
    { "normalized", "", AttributeProto::INT, static_cast<int64_t>(0) },
    { "onesided", "", AttributeProto::INT, static_cast<int64_t>(1) },
};
constexpr static ONNX_NAMESPACE::FormalParameter RfftInputs[] {
    { "X", "input tensor"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter RfftOutputs[] {
    { "Y", "output tensor"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute IrfftAttributes[]{
    { "signal_ndim", "", AttributeProto::INT },
    { "normalized", "", AttributeProto::INT, static_cast<int64_t>(0) },
    { "onesided", "", AttributeProto::INT, static_cast<int64_t>(1) },
};
constexpr static ONNX_NAMESPACE::FormalParameter IrfftInputs[] {
    { "X", "input tensor"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter IrfftOutputs[] {
    { "Y", "output tensor"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::FormalParameter ComplexMulInputs[] {
    { "A", "input_0"_docstring, "T" },
    { "B", "input_1"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ComplexMulOutputs[] {
    { "C", "output tensor"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::FormalParameter ComplexMulConjInputs[] {
    { "A", "input_0"_docstring, "T" },
    { "B", "input_1"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ComplexMulConjOutputs[] {
    { "C", "output tensor"_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::Attribute ConvTransposeWithDynamicPadsAttributes[]{
    { "kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "output_padding", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "dilations", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "strides", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "auto_pad", "", AttributeProto::STRING, "NOTSET" },
    { "group", "", AttributeProto::INT, static_cast<int64_t>(1) },
};
constexpr static ONNX_NAMESPACE::FormalParameter ConvTransposeWithDynamicPadsInputs[] {
    { "X", "", "T" },
    { "W", "", "T" },
    { "Pads", "", "tensor(int64)", OpSchema::Optional },
    { "B", "", "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter ConvTransposeWithDynamicPadsOutputs[] {
    { "Y", "", "T" },
};

constexpr static ONNX_NAMESPACE::Attribute FusedConvAttributes[]{
    { "auto_pad", "", AttributeProto::STRING, "NOTSET" },
    { "kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "dilations", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "strides", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "pads", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "group", "", AttributeProto::INT, static_cast<int64_t>(1) },
    { "activation", "", AttributeProto::STRING, OPTIONAL_VALUE },
    { "activation_params", "", AttributeProto::FLOATS, OPTIONAL_VALUE }
};
constexpr static ONNX_NAMESPACE::FormalParameter FusedConvInputs[] {
    { "X", "", "T" },
    { "W", "", "T" },
    { "B", "", "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter FusedConvOutputs[] {
    { "Y", "", "T" },
};
static void FusedConvTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
}

constexpr static ONNX_NAMESPACE::Attribute FusedGemmAttributes[]{
    { "transA", "Whether A should be transposed"_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
    { "transB", "Whether B should be transposed"_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
    { "alpha", "Scalar multiplier for the product of input tensors A * B."_docstring, AttributeProto::FLOAT, 1.0f },
    { "beta" , "Scalar multiplier for input tensor C."_docstring, AttributeProto::FLOAT, 1.0f },
    { "activation", "", AttributeProto::STRING, OPTIONAL_VALUE },
    { "activation_alpha", "", AttributeProto::FLOAT, OPTIONAL_VALUE },
    { "activation_beta", "", AttributeProto::FLOAT, OPTIONAL_VALUE },
    { "activation_gamma", "", AttributeProto::FLOAT, OPTIONAL_VALUE },
};
constexpr static ONNX_NAMESPACE::FormalParameter FusedGemmInputs[] {
  { "A", "Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero."_docstring, "T" },
  { "B", "Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero."_docstring, "T" },
  { "C", "Input tensor C. The shape of C should be unidirectional broadcastable to (M, N)."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter FusedGemmOutputs[] {
    { "Y", "Output tensor of shape (M, N)."_docstring, "T" },
};
static void FusedGemmTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (hasNInputShapes(ctx, 2)) {
        auto transAAttr = ctx.getAttribute("transA");
        bool transA =
                transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transB =
                transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
        auto& first_input_shape = getInputShape(ctx, 0);
        auto& second_input_shape = getInputShape(ctx, 1);
        if (first_input_shape.dim_size() != 2)
            fail_shape_inference("First input does not have rank 2");
        if (second_input_shape.dim_size() != 2)
            fail_shape_inference("Second input does not have rank 2");
        updateOutputShape(
                ctx,
                0,
                {first_input_shape.dim(transA ? 1 : 0),
                 second_input_shape.dim(transB ? 0 : 1)});
    }
}
constexpr static ONNX_NAMESPACE::TypeConstraint FloatingPointIntTTypeConstraints[] {
    { "T",
        StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_DOUBLE, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT64>::data,
        "Constrain input and output types to float/int tensors."_docstring },
};

constexpr static ONNX_NAMESPACE::FormalParameter ExpandDimsInputs[] {
    { "X", "input"_docstring, "T" },
    { "axis", "Specified axis to insert a dimension"_docstring, "tensor(int32)" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ExpandDimsOutputs[] {
    { "Y", "output"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint ExpandDimsTypeConstraints[] {
    { "T", AllTensorTypes, "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type."_docstring },
};
static void ExpandDimsTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    // Type inference
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    // Shape inference
    if (!hasInputShape(ctx, 0))
        return;

    auto& input_shape = getInputShape(ctx, 0);
    const int rank = input_shape.dim_size();
    const ONNX_NAMESPACE::TensorProto* axis_initializer = ctx.getInputData(1);
    if (!axis_initializer)
        return;
    const int axis = axis_initializer->int32_data()[0];
    if (axis > rank || axis < -rank - 1) {
        fail_shape_inference("Input axis is invalid: ", axis);
    }
    int pos = axis >= 0 ? axis : rank + axis - 1;
    ONNX_NAMESPACE::TensorShapeProto output_shape;
    for (int i = 0; i < pos; ++i) {
        output_shape.add_dim();
        *(output_shape.mutable_dim(i)) = input_shape.dim(i);
    }
    output_shape.add_dim();
    output_shape.mutable_dim(pos)->set_dim_value(1);
    for (int i = pos + 1; i < rank + 1; ++i) {
        output_shape.add_dim();
        *(output_shape.mutable_dim(i)) = input_shape.dim(i - 1);
    }
    updateOutputShape(ctx, 0, output_shape);
}

constexpr static ONNX_NAMESPACE::Attribute QuantizeLinearAttributes[]{
    { "axis",
        "The axis along which same quantization parameters are applied. It's optional."
        "If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars."
        "If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors."_docstring, AttributeProto::INT, false },
};
constexpr static ONNX_NAMESPACE::FormalParameter QuantizeLinearInputs[] {
    { "x", "N-D full precision Input tensor to be quantized."_docstring, "T1" },
    { "y_scale", "Scale for doing quantization to get 'y'. It could be a scalar or a 1-D tensor," "which means a per-tensor or per-axis quantization. If it's a 1-D tensor, " "its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'."_docstring, "T1" },
    { "y_zero_point", "Zero point for doing quantization to get 'y'. It could be a scalar or a 1-D tensor, which means a per-tensor" "or per-axis quantization. If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'."_docstring, "T2" },
};
constexpr static ONNX_NAMESPACE::FormalParameter QuantizeLinearOutputs[] {
    { "y", "N-D quantized output tensor. It has same shape as input 'x'."_docstring, "T2" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint QuantizeLinearTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT>::data, "Constrain 'x', 'y_scale' to float tensors."_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain 'y_zero_point' and 'y' to 8-bit integer tensors."_docstring },
};
static void QuantizeLinearTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 2, 0);

    if (!hasInputShape(ctx, 0))
        return;

    auto& input_shape = getInputShape(ctx, 0);
    updateOutputShape(ctx, 0, input_shape);
}


constexpr static ONNX_NAMESPACE::Attribute DequantizeLinearAttributes[]{
    { "axis",
        "The axis along which same quantization parameters are applied. It's optional."
        "If it's not specified, it means per-tensor quantization and input 'x_scale' and 'x_zero_point' must be scalars."
        "If it's specified, it means per 'axis' quantization and input 'x_scale' and 'x_zero_point' must be 1-D tensors."_docstring, AttributeProto::INT, false },
};
constexpr static ONNX_NAMESPACE::FormalParameter DequantizeLinearInputs[] {
    { "x", "N-D quantized Input tensor to be de-quantized."_docstring, "T1" },
    { "x_scale", "Scale for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization."
        "If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'."_docstring,
        "T2" },
    { "x_zero_point",
        "Zero point for input 'x'. It could be a scalar or a 1-D tensor, which means a per-tensor or per-axis quantization."
        "If it's a 1-D tensor, its number of elements should be equal to the dimension value of 'axis' dimension of input 'x'."_docstring,
        "T1" },
};
constexpr static ONNX_NAMESPACE::FormalParameter DequantizeLinearOutputs[] {
    { "y", "N-D full precision output tensor. It has same shape as input 'x'."_docstring, "T2" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint DequantizeLinearTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain 'x' and 'x_zero_point' to 8-bit integer tensors."_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT>::data, "Constrain 'y', 'x_scale' to float tensors."_docstring },
};
static void DequantizeLinearTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    auto y_type = ctx.getOutputType(0);
    // only float is supported
    y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

    if (!hasInputShape(ctx, 0))
        return;

    auto& input_shape = getInputShape(ctx, 0);
    updateOutputShape(ctx, 0, input_shape);
}

constexpr static ONNX_NAMESPACE::Attribute TokenizerAttributes[]{
    { "mark", "Boolean whether to mark the beginning/end character with start of text character (0x02)/end of text character (0x03)."_docstring, AttributeProto::INT },
    { "pad_value", "The string used to pad output tensors when the tokens extracted doesn't match the maximum number of tokens found. If start/end markers are needed, padding will appear outside the markers."_docstring, AttributeProto::STRING },
    { "tokenexp", "An optional string. Token's regular expression in basic POSIX format"
        " (http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html#tag_09_03)."
        " If set, tokenizer may produce tokens matching the specified pattern. Note that one and only of"
        " 'tokenexp' and 'separators' should be set."_docstring,
        AttributeProto::STRING,
        OPTIONAL_VALUE },
    { "separators",
        "an optional list of strings attribute that contains a list of separators - regular expressions to match separators"
        " Two consecutive segments in X connected by a separator would be divided into two tokens."
        " For example, if the input is \"Hello World!\" and this attribute contains only one space character,"
        " the corresponding output would be [\"Hello\", \"World!\"]. To achieve character-level tokenization,"
        " one should set the 'separators' to [\"\"], which contains an empty string."_docstring,
        AttributeProto::STRINGS,
        OPTIONAL_VALUE },
    { "mincharnum",
        "Minimum number of characters allowed in the output. For example, if mincharnum is 2, tokens such as \"A\" and \"B\" would be ignored"_docstring,
        AttributeProto::INT },
};
constexpr static ONNX_NAMESPACE::FormalParameter TokenizerInputs[] {
    { "X", "Strings to tokenize"_docstring, "tensor(string)" },
};
constexpr static ONNX_NAMESPACE::FormalParameter TokenizerOutputs[] {
    { "Y", "Tokenized strings"_docstring, "tensor(string)" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint TokenizerTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain 'x' and 'x_zero_point' to 8-bit integer tensors."_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT>::data, "Constrain 'y', 'x_scale' to float tensors."_docstring },
};
static void TokenizerTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    // Shape inference
    if (!hasInputShape(ctx, 0))
        return;

    ONNX_NAMESPACE::TensorShapeProto output_shape;
    auto& input_shape = getInputShape(ctx, 0);
    auto& dims = input_shape.dim();
    if (dims.size() < 1 || dims.size() > 2) {
        fail_shape_inference("Input dimensions are either [C] or [N][C] allowed");
    }

    int64_t size = 1;
    for (auto& dim : dims) {
        if (utils::HasDimValue(dim)) {
            size *= dim.dim_value();
        }
    }

    if (size > 0) {
        for (auto& dim : dims) {
            *output_shape.add_dim() = dim;
        }
        // Add the last unknown dimension
        // only if the input is not empty
        output_shape.add_dim();
    } else if (size == 0) {
        if (dims.size() == 2) {
            *output_shape.add_dim() = dims[0];
        }
        output_shape.add_dim()->set_dim_value(0);
    }
    updateOutputShape(ctx, 0, output_shape);
}

constexpr static ONNX_NAMESPACE::Attribute MatMulInteger16Attributes[]{
    { "mark", "Boolean whether to mark the beginning/end character with start of text character (0x02)/end of text character (0x03)."_docstring, AttributeProto::INT },
    { "pad_value", "The string used to pad output tensors when the tokens extracted doesn't match the maximum number of tokens found. If start/end markers are needed, padding will appear outside the markers."_docstring, AttributeProto::STRING },
    { "tokenexp", "An optional string. Token's regular expression in basic POSIX format"
                  " (http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html#tag_09_03)."
                  " If set, tokenizer may produce tokens matching the specified pattern. Note that one and only of"
                  " 'tokenexp' and 'separators' should be set."_docstring,
        AttributeProto::STRING,
        OPTIONAL_VALUE },
    { "separators",
        "an optional list of strings attribute that contains a list of separators - regular expressions to match separators"
        " Two consecutive segments in X connected by a separator would be divided into two tokens."
        " For example, if the input is \"Hello World!\" and this attribute contains only one space character,"
        " the corresponding output would be [\"Hello\", \"World!\"]. To achieve character-level tokenization,"
        " one should set the 'separators' to [\"\"], which contains an empty string."_docstring,
        AttributeProto::STRINGS,
        OPTIONAL_VALUE },
    { "mincharnum",
        "Minimum number of characters allowed in the output. For example, if mincharnum is 2, tokens such as \"A\" and \"B\" would be ignored"_docstring,
        AttributeProto::INT },
};
constexpr static ONNX_NAMESPACE::FormalParameter MatMulInteger16Inputs[] {
    { "A", "N-dimensional matrix A"_docstring, "T1" },
    { "B", "N-dimensional matrix B"_docstring, "T2" },
};
constexpr static ONNX_NAMESPACE::FormalParameter MatMulInteger16Outputs[] {
    { "Y", "Matrix multiply results from A * B"_docstring, "T3" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint MatMulInteger16TypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT16>::data, "Constrain input A data types as 16-bit integer tensor"_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT16>::data, "Constrain input B data types as 16-bit integer tensor"_docstring },
    { "T3", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT32>::data,
        "Constrain output Y data types as 32-bit integer tensor."
        "T3 must be tensor(uint32) when both T1 and T2 are tensor(uint16),"
        "or must be tensor(int32) when either T1 or T2 is tensor(int16)."_docstring},
};
static void MatMulInteger16TypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    auto a_type = ctx.getInputType(0);
    auto b_type = ctx.getInputType(1);
    auto y_type = ctx.getOutputType(0);
    if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
        a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
        b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference(
                "inputs are expected to have tensor type and output type should not be null.");
    }

    // Right now we only support int32
    y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

    ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
}

constexpr static ONNX_NAMESPACE::FormalParameter DynamicQuantizeMatMulInputs[] {
    { "A", "N-dimensional matrix A"_docstring, "T1" },
    { "B", "N-dimensional matrix B"_docstring, "T2" },
    { "b_scale", "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor,  ""which means a per-tensor or per-column quantization. If it's a 1-D tensor,  its number ""of elements should be equal to the number of columns of input 'B'."_docstring, "T1" },
    { "b_zero_point", "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor,  ""which means a per-tensor or per-column quantization. If it's a 1-D tensor,  its number ""of elements should be equal to the number of columns of input 'B'."_docstring, "T2", OpSchema::Optional },
    { "bias", "1D input tensor,  whose dimension is same as B's last dimension"_docstring, "T1", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter DynamicQuantizeMatMulOutputs[] {
    { "Y", "Matrix multiply results from A * B"_docstring, "T1" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint DynamicQuantizeMatMulTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT>::data, "Constrain input A, b_scale and output Y data type as float tensor."_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input B data type to 8-bit integer tensor."_docstring },
};
static void DynamicQuantizeMatMulTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
}


constexpr static ONNX_NAMESPACE::FormalParameter MatMulIntegerToFloatInputs[] {
    {"A", "N-dimensional matrix A"_docstring, "T1" },
    {"B", "N-dimensional matrix B"_docstring, "T2" },
    {
            "a_scale",
            "Scale of quantized input 'A'. It could be a scalar or a 1-D tensor, "
            "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
            "of elements should be equal to the number of columns of input 'A'."_docstring,
            "T3" },
    {
            "b_scale",
            "Scale of quantized input 'B'. It could be a scalar or a 1-D tensor, "
            "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
            "of elements should be equal to the number of columns of input 'B'."_docstring,
            "T3" },
    {
            "a_zero_point",
            "Zero point tensor for input 'A'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
            "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
            "of elements should be equal to the number of columns of input 'A'."_docstring,
            "T1",
            OpSchema::Optional },
    {
            "b_zero_point",
            "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
            "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
            "of elements should be equal to the number of columns of input 'B'."_docstring,
            "T2",
            OpSchema::Optional },
    {
            "bias",
            "1D input tensor, whose dimension is same as B's last dimension"_docstring,
            "T3",
            OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter MatMulIntegerToFloatOutputs[] {
    { "Y", "Matrix multiply results from A * B"_docstring, "T3" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint MatMulIntegerToFloatTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input A data type to 8-bit integer tensor."_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input B data type to 8-bit integer tensor."_docstring },
    { "T3", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT>::data, "Constrain input a_scale, b_scale and output Y data type as float tensor."_docstring },
};
static void MatMulIntegerToFloatTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 2, 0);
    ONNX_NAMESPACE::matmulShapeInference(ctx, 0, 1);
}

constexpr static ONNX_NAMESPACE::Attribute TransposeMatMulAttributes[]{
    { "transA", "Whether A should be transposed on the last two dimensions before doing multiplication"_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
    { "transB", "Whether B should be transposed on the last two dimensions before doing multiplication"_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
};
constexpr static ONNX_NAMESPACE::FormalParameter TransposeMatMulInputs[] {
    { "A", "N-dimensional matrix A"_docstring, "T" },
    { "B", "N-dimensional matrix B"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter TransposeMatMulOutputs[] {
    { "Y", "Matrix multiply results"_docstring, "T" },
};
static void TransposeMatMulTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    auto transAAttr = ctx.getAttribute("transA");
    bool transa = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
    auto transBAttr = ctx.getAttribute("transB");
    bool transb = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
    int input1Idx = 0;
    int input2Idx = 1;
    if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
        return;
    }

    const auto shape0_raw = getInputShape(ctx, input1Idx);
    const auto shape1_raw = getInputShape(ctx, input2Idx);

    if (shape0_raw.dim_size() == 0 || shape1_raw.dim_size() == 0) {
        fail_shape_inference("Input tensors of wrong rank (0).");
    }

    // numpy transpose on a vector does not change anything.
    if (shape0_raw.dim_size() == 1) {
        transa = false;
    }
    if (shape1_raw.dim_size() == 1) {
        transb = false;
    }

    ONNX_NAMESPACE::TensorShapeProto shape0, shape1;
    auto rank0 = shape0_raw.dim_size();
    if (rank0 == 1) {
        // for vector input, transa does not make impact on the dim.
        shape0 = shape0_raw;
    } else {
        for (int i = 0; i < rank0 - 2; ++i) {
            *shape0.add_dim() = shape0_raw.dim(i);
        }
        *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 1 : rank0 - 2);
        *shape0.add_dim() = shape0_raw.dim(transa ? rank0 - 2 : rank0 - 1);
    }

    auto rank1 = shape1_raw.dim_size();
    if (rank1 == 1) {
        // for vector input, transb does not make impact on the dim.
        shape1 = shape1_raw;
    } else {
        for (int i = 0; i < rank1 - 2; ++i) {
            *shape1.add_dim() = shape1_raw.dim(i);
        }
        *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 1 : rank1 - 2);
        *shape1.add_dim() = shape1_raw.dim(transb ? rank1 - 2 : rank1 - 1);
    }

    ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

    // First promote each shape to at least rank-2. This logic is
    // specific to matmul, not generic broadcasting.
    {
        if (shape0.dim_size() == 1) {
            shapeL.add_dim()->set_dim_value(1);
            *shapeL.add_dim() = shape0.dim(0);
        } else {
            *shapeL.mutable_dim() = shape0.dim();
        }
        if (shape1.dim_size() == 1) {
            *shapeR.add_dim() = shape1.dim(0);
            shapeR.add_dim()->set_dim_value(1);
        } else {
            *shapeR.mutable_dim() = shape1.dim();
        }
    }

    // Check for compatible matrix multiply dimensions
    {
        auto dimL = shapeL.dim(shapeL.dim_size() - 1);
        auto dimR = shapeR.dim(shapeR.dim_size() - 2);
        if (dimL.has_dim_value() && dimR.has_dim_value() &&
            dimL.dim_value() != dimR.dim_value()) {
            fail_shape_inference("Incompatible dimensions for matrix multiplication");
        }
    }

    ONNX_NAMESPACE::TensorShapeProto resultShape;

    // Now call out to generic multidimensional broadcasting for
    // the broadcastable prefixes.
    {
        ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
        for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
            *prefixShapeL.add_dim() = shapeL.dim(i);
        }
        for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
            *prefixShapeR.add_dim() = shapeR.dim(i);
        }
        bidirectionalBroadcastShapeInference(
                prefixShapeL, prefixShapeR, resultShape);
    }

    // Back to matmul-specific. Add the trailing dimensions back in.
    {
        if (shape0.dim_size() != 1) {
            *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
        }
        if (shape1.dim_size() != 1) {
            *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
        }
    }
    updateOutputShape(ctx, 0, resultShape);
}

constexpr static ONNX_NAMESPACE::Attribute ReduceSumIntegerAttributes[]{
    { "axes", "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor."_docstring, AttributeProto::INTS },
    { "keepdims", "Keep the reduced dimension or not, default 1 mean keep reduced dimension."_docstring, AttributeProto::INT },
};
constexpr static ONNX_NAMESPACE::FormalParameter ReduceSumIntegerInputs[] {
    { "data", "An input tensor."_docstring, "T1" },
};
constexpr static ONNX_NAMESPACE::FormalParameter ReduceSumIntegerOutputs[] {
    { "reduced", "Reduced output tensor."_docstring, "T2" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint ReduceSumIntegerOutputsTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input type to 8-bit integer tensor."_docstring },
    { "T2",
        StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT32>::data,
        "Constrain output data type to 32-bit integer tensor."
        "T2 must be tensor(uint32) when T1 is tensor(uint8),"
        "or must be tensor(int32) when T1 is tensor(int8)."_docstring },
};

constexpr static ONNX_NAMESPACE::Attribute QLinearReduceMeanAttributes[]{
    { "axes", "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor."_docstring, AttributeProto::INTS },
    { "keepdims", "Keep the reduced dimension or not, default 1 mean keep reduced dimension."_docstring, AttributeProto::INT },
};
constexpr static ONNX_NAMESPACE::FormalParameter QLinearReduceMeanInputs[] {
    { "data", "An input tensor."_docstring, "T" },
    {"data_scale", "Input scale. It's a scalar, which means a per-tensor/layer quantization."_docstring, "tensor(float)"},
    {"data_zero_point", "Input zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional},
    {"reduced_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization."_docstring, "tensor(float)"},
    {"reduced_zero_point", "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional},
};
constexpr static ONNX_NAMESPACE::FormalParameter QLinearReduceMeanOutputs[] {
    { "reduced", "Reduced output tensor."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint QLinearReduceMeanOutputsTypeConstraints[] {
    { "T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input type to 8-bit integer tensor."_docstring },
};
static void QLinearTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    if (!hasNInputShapes(ctx, 1)) {
        return;
    }

    auto data_type = ctx.getInputType(0);
    if (nullptr == data_type || data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
    }

    // validate scale and zero points
    ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
    ValidateTypeAndShapeForScaleAndZP(ctx, 2, data_type->tensor_type().elem_type(), true);
    ValidateTypeAndShapeForScaleAndZP(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT, true);
    ValidateTypeAndShapeForScaleAndZP(ctx, 4, data_type->tensor_type().elem_type(), true);

    int64_t keep_dims = 1;
    auto attr_proto = ctx.getAttribute("keepdims");
    if (attr_proto) {
        keep_dims = attr_proto->i();
    }

    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t input_ndim = input_shape.dim_size();
    auto output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    std::vector<int64_t> axes;
    auto axes_proto = ctx.getAttribute("axes");
    if (axes_proto)
        axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < -input_ndim || axes[i] >= input_ndim) {
            fail_shape_inference(
                    "axis must be in [-rank, rank-1]. input rank was ", input_ndim);
        }
        if (axes[i] < 0)
            axes[i] += input_ndim;
    }
    // do we need to handle negative axis?
    for (int i = 0; i < input_ndim; ++i) {
        // axes empty means reduce all dim
        if (!axes.empty() &&
            std::find(axes.begin(), axes.end(), i) == axes.end()) {
            auto dim = output_shape->add_dim();
            dim->CopyFrom(input_shape.dim(i));
        } else {
            if (keep_dims == 1) {
                auto dim = output_shape->add_dim();
                dim->set_dim_value(1);
            }
        }
    }
}

constexpr static ONNX_NAMESPACE::FormalParameter MulIntegerInputs[] {
    { "A", "First operand."_docstring, "T" },
    { "A_zero_point", "Input A zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional },
    { "B", "Second operand.", "T" },
    { "B_zero_point", "Input B zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter MulIntegerOutputs[] {
    { "C", "Constrain output to 32 bit tensor"_docstring, "T1" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint MulIntegerOutputsTypeConstraints[] {
    { "T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input type to 8-bit integer tensor."_docstring },
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32>::data, "Constrain output types to 32 bit tensors."_docstring },
};
static void MulIntegerTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    auto c_type = ctx.getOutputType(0);
    c_type->mutable_tensor_type()->set_elem_type(
            ONNX_NAMESPACE::TensorProto::INT32);

    auto a_type = ctx.getInputType(0);
    auto b_type = ctx.getInputType(3);
    if (nullptr == a_type || nullptr == b_type ||
        a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
        b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
    }

    ValidateTypeAndShapeForScaleAndZP(ctx, 1, a_type->tensor_type().elem_type(), true);
    ValidateTypeAndShapeForScaleAndZP(ctx, 3, b_type->tensor_type().elem_type(), true);

    if (hasInputShape(ctx, 0) && hasInputShape(ctx, 2)) {
        bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(2)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }
}

constexpr static ONNX_NAMESPACE::Attribute QLinearAveragePoolAttributes[] {
    { "count_include_pad", "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad."_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
    { "kernel_shape", "The size of the kernel along each axis."_docstring, AttributeProto::INTS },
    { "strides", "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis."_docstring, AttributeProto::INTS, OPTIONAL_VALUE },
    { "auto_pad", "", AttributeProto::STRING, "NOTSET" },
    { "pads", "", AttributeProto::INTS, OPTIONAL_VALUE },
    { "ceil_mode", "Whether to use ceil or floor (default) to compute the output shape."_docstring, AttributeProto::INT, static_cast<int64_t>(0) },
};
constexpr static ONNX_NAMESPACE::FormalParameter QLinearAveragePoolInputs[] {
    { "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size. Optionally, if dimension denotation is "
        "in effect, the operation expects the input "
        "data tensor to arrive with the dimension denotation "
        "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]."_docstring,
        "T" },
    { "x_scale", "Input scale. It's a scalar, which means a per-tensor/layer quantization."_docstring, "tensor(float)" },
    { "x_zero_point", "Input zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional },
    { "y_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization."_docstring, "tensor(float)" },
    { "y_zero_point", "Output zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter QLinearAveragePoolOutputs[] {
    { "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used"_docstring,
        "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint QLinearAveragePoolOutputsTypeConstraints[] {
    { "T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input and output types to 8-bit integer tensor."_docstring },
};

constexpr static ONNX_NAMESPACE::Attribute QLinearLeakyReluAttributes[] {
    { "alpha", "Coefficient of leakage."_docstring, AttributeProto::FLOAT, 0.01f },
};
constexpr static ONNX_NAMESPACE::FormalParameter QLinearLeakyReluInputs[] {
    { "X", "Input tensor", "T" },
    { "X_scale", "Input X's scale. It's a scalar, which means a per-tensor/layer quantization."_docstring, "tensor(float)" },
    { "X_zero_point", "Input X's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional },
    { "Y_scale", "Output Y's scale. It's a scalar, which means a per-tensor/layer quantization."_docstring, "tensor(float)" },
    { "Y_zero_point", "Output Y's zero point. Default value is 0 if it's not specified. It's a scalar, which means a per-tensor/layer quantization."_docstring, "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter QLinearLeakyReluOutputs[] {
    { "Y", "Output tensor"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint QLinearLeakyReluOutputsTypeConstraints[] {
    { "T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT8, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT8>::data, "Constrain input and output types to 8-bit integer tensor."_docstring },
};
static void QLinearAveragePoolTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

    auto data_type = ctx.getInputType(0);
    if (nullptr == data_type || data_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
    }

    // validate scale and zero points
    ValidateTypeAndShapeForScaleAndZP(ctx, 1, ONNX_NAMESPACE::TensorProto::FLOAT, true);
    ValidateTypeAndShapeForScaleAndZP(ctx, 2, data_type->tensor_type().elem_type(), true);
    ValidateTypeAndShapeForScaleAndZP(ctx, 3, ONNX_NAMESPACE::TensorProto::FLOAT, true);
    ValidateTypeAndShapeForScaleAndZP(ctx, 4, data_type->tensor_type().elem_type(), true);

    ONNX_NAMESPACE::convPoolShapeInference(ctx, false, true, 0, 5);
}

constexpr static ONNX_NAMESPACE::FormalParameter MurmurHash3Inputs[] {
    { "X", "An input tensor to hash."_docstring, "T1" },
};
constexpr static ONNX_NAMESPACE::FormalParameter MurmurHash3Outputs[] {
    { "Y", "32-bit hash value."_docstring, "T2" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint MurmurHash3OutputsTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT64, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_DOUBLE, &ONNX_NAMESPACE::TypeConstraint::TENSOR_STRING>::data, "Constrain input type to unsigned or signed 32-bit integer tensor, or string tensor. It should be utf-8 encoded if using unicode."_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_UINT32>::data, "Constrain output type to unsigned and signed 32-bit integer tensor."_docstring },
};
constexpr static ONNX_NAMESPACE::Attribute MurmurHash3Attributes[] {
    { "seed", "Seed for the hashing algorithm, unsigned 32-bit integer, default to 0."_docstring, AttributeProto::INT, (int64_t)0LL },
    { "positive", "If value is 1, output type is uint32_t, else int32_t. Default value is 1."_docstring, AttributeProto::INT, (int64_t)1LL },
};
static void MurmurHash3TypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    // type inference
    auto positive_attr = ctx.getAttribute("positive");
    bool is_positive =
            positive_attr ? (static_cast<int>(positive_attr->i()) == 1 ? true : false) : true /* default value if attribute not present */;
    auto output_data_type = ctx.getOutputType(0)->mutable_tensor_type();
    if (is_positive) {
        output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32);
    } else {
        output_data_type->set_elem_type(::ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32);
    }

    // Shape inference
    if (!hasInputShape(ctx, 0))
        return;

    auto& input_shape = getInputShape(ctx, 0);
    updateOutputShape(ctx, 0, input_shape);
}

constexpr static ONNX_NAMESPACE::FormalParameter GatherNDInputs[] {
    { "data", "Tensor of rank r >= 1."_docstring, "T" },
    { "indices", "Tensor of rank q >= 1."_docstring, "Tind" },
};
constexpr static ONNX_NAMESPACE::FormalParameter GatherNDOutputs[] {
    { "output", "Tensor of rank q-1+r-indices[-1]."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint GatherNDOutputsTypeConstraints[] {
    { "T", AllTensorTypes, "Constrain input and output types to any tensor type."_docstring },
    { "Tind", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32, &ONNX_NAMESPACE::TypeConstraint::TENSOR_INT64>::data, "Constrain indice type to int32 or int64"_docstring },
};
static void GatherNDTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (!hasNInputShapes(ctx, 2)) {
        return;
    }
    auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
    auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
    auto data_rank = data_shape.dim_size();
    auto indices_rank = indices_shape.dim_size();
    if (data_rank < 1 || indices_rank < 1) {
        fail_shape_inference("both data and indices tensor need to have rank larger than zero.");
    }
    auto last_indice_dimension = indices_shape.dim(indices_rank - 1).dim_value();
    if (last_indice_dimension > data_rank) {
        fail_shape_inference("last dimension of indices must not be larger and rank of data tensor");
    }
    for (int i = 0; i < indices_rank - 1; ++i) {
        *ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim() = indices_shape.dim(i);
    }
    for (int i = static_cast<int>(last_indice_dimension); i < data_rank; ++i) {
        *ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->add_dim() = data_shape.dim(i);
    }
}

constexpr static ONNX_NAMESPACE::FormalParameter WordConvEmbeddingInputs[] {
    { "Sequence", "Specify batchs of sequence words to embedding"_docstring, "T" },
    { "W", "Specify weights of conv"_docstring, "T1" },
    { "B", "Specify bias of conv"_docstring, "T1" },
    { "C", "Specify embedding vector of char"_docstring, "T1" },
};
constexpr static ONNX_NAMESPACE::FormalParameter WordConvEmbeddingOutputs[] {
    { "Y", "output"_docstring, "T1" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint WordConvEmbeddingOutputsTypeConstraints[] {
    { "T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32>::data, "Constrain to tensor(int32)."_docstring },
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT>::data, "Constrain to tensor(float)."_docstring },
};
constexpr static ONNX_NAMESPACE::Attribute WordConvEmbeddingAttributes[] {
    { "embedding_size", "Integer representing the embedding vector size for each word."
        "If not provide, use the fileter size of conv weight"_docstring, AttributeProto::INT, OPTIONAL_VALUE },
    { "conv_window_size", "This operator applies convolution to word from left to right with window equal to conv_window_size and stride to 1."
        "Take word 'example' for example, with conv_window_size equal to 2, conv is applied to [ex],[xa], [am], [mp]..."
        "If not provide, use the first dimension of conv kernal shape."_docstring, AttributeProto::INT, OPTIONAL_VALUE },
    { "char_embedding_size", "Integer representing the embedding vector size for each char."
        "If not provide, use the char embedding size of embedding vector."_docstring, AttributeProto::INT, OPTIONAL_VALUE },
};

constexpr static ONNX_NAMESPACE::FormalParameter PadInputs[] {
    { "data", "Input tensor.", "T" },
    {
      "pads", "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
              "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
              "`pads` should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]. "
              "`pads` format (1D example) should be as follow [x1_begin, x2_begin,...,x1_end, x2_end,...], "
              "where xi_begin is the number of pixels added at the beginning of axis `i` and "
              "xi_end, the number of pixels added at the end of axis `i`."_docstring, "tensor(int64)" },
    { "value", "(Optional) A scalar or rank 1 tensor containing a single value to be filled if the mode chosen is `constant` (by default it is 0.0)."_docstring, "T", OpSchema::Optional },
};
constexpr static ONNX_NAMESPACE::FormalParameter PadOutputs[] {
    { "output", "Tensor after padding."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::Attribute PadAttributes[] {
    { "mode",
        "Three modes: `constant`(default) - pads with a given constant value, "
        "`reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis, "
        "`edge` - pads with the edge values of array"_docstring,
        AttributeProto::STRING,
        "constant" },
};
static void PadTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    // Type inference
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    // Shape inference needs the input data shape
    if (!hasNInputShapes(ctx, 1)) {
        return;
    }
    const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    const auto input_rank = input_shape.dim_size();

    // Infer output shape if 'pads' tensor is available
    const auto* pads_initializer = ctx.getInputData(1);
    if (nullptr != pads_initializer) {
        const auto& pads_shape = ctx.getInputType(1)->tensor_type().shape();
        if ((pads_initializer->dims_size() != 1 &&
             pads_initializer->dims_size() != 2) ||
            (pads_initializer->dims_size() == 2 &&
             pads_shape.dim(static_cast<int>(0)).dim_value() != 1) ||
            pads_initializer->data_type() != ONNX_NAMESPACE::TensorProto::INT64)
            fail_shape_inference(
                    "'pads' input must be a 1D (shape: [input_rank]) "
                    "or 2D tensor (shape: [1, input_rank]) of type int64");

        // make a copy of the returned const vector - may have to resize
        // this in next step
        std::vector<int64_t> pads_data;
        if (utils::HasRawData(*pads_initializer))
            return;
        else
            pads_data.insert(
                    pads_data.end(),
                    pads_initializer->int64_data().begin(),
                    pads_initializer->int64_data().end());

        // fill with zeros if needed to reach appropriate size
        if (pads_data.size() != 2 * static_cast<size_t>(input_rank))
            pads_data.resize(size_t{2} * input_rank, 0);

        const auto& output_shape =
                ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
            const auto& input_dim = input_shape.dim(static_cast<int>(i));
            auto* output_dim = output_shape->add_dim();
            if (utils::HasDimValue(input_dim)) {
                output_dim->set_dim_value(
                        input_dim.dim_value() + pads_data[i] + pads_data[i + input_rank]);
            } else if (pads_data[i] + pads_data[i + input_rank] == 0) {
                *output_dim = input_dim;
            }
        }
    } else {
        // Infer output shapes' rank in any case
        auto* output_shape_0 = getOutputShape(ctx, 0);
        for (size_t i = 0; static_cast<int64_t>(i) < input_rank; ++i) {
            output_shape_0->add_dim();
        }
    }
}

constexpr static ONNX_NAMESPACE::FormalParameter UniqueInputs[] {
    { "x", "A 1-D input tensor that is to be processed."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter UniqueOutputs[] {
    { "y",
        "A 1-D tensor of the same type as 'x' "
        "containing all the unique values in 'x' sorted "
        "in the same order that they occur in the input 'x'"_docstring,
        "T" },
    { "idx",
         "A 1-D INT64 tensor of the same size as 'x' "
         "containing the indices for each value in 'x' "
         "in the output 'uniques'"_docstring,
         "tensor(int64)" },
    { "counts",
         "A 1-D INT64 tensor containing the "
         "the count of each element "
         "of 'uniques' in the input 'x'"_docstring,
         "tensor(int64)" },
    { "output", "Tensor after padding."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint UniqueTypeConstraints[] {
    { "T", AllTensorTypes, "Input can be of any tensor type."_docstring },
};
static void UniqueTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    // Type inference
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    ONNX_NAMESPACE::updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::INT64);
    ONNX_NAMESPACE::updateOutputElemType(ctx, 2, ONNX_NAMESPACE::TensorProto::INT64);

    // Shape inference

    // shape of output 'uniques' and 'counts'
    // depends on actual input data, but the rank is always 1
    ctx.getOutputType(0)
            ->mutable_tensor_type()
            ->mutable_shape()
            ->add_dim();

    ctx.getOutputType(2)
            ->mutable_tensor_type()
            ->mutable_shape()
            ->add_dim();

    // if the input shape doesn't exist, further shape inference is not possible
    if (!hasNInputShapes(ctx, 1)) {
        return;
    }

    // 'idx' output has same shape as input
    ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 1);
}

constexpr static ONNX_NAMESPACE::Attribute CDistAttributes[] {
    { "metric",
            "The distance metric to use. If a string, the distance function can be \"braycurtis\", \"canberra\", "
            "\"chebyshev\", \"cityblock\", \"correlation\", \"cosine\", \"dice\", \"euclidean\", \"hamming\", \"jaccard\", "
            "\"jensenshannon\", \"kulsinski\", \"mahalanobis\", \"matching\", \"minkowski\", \"rogerstanimoto\", \"russellrao\", "
            "\"seuclidean\", \"sokalmichener\", \"sokalsneath\", \"sqeuclidean\", \"wminkowski\", \"yule\"."_docstring,
            AttributeProto::STRING, "sqeuclidean" },
};
constexpr static ONNX_NAMESPACE::FormalParameter CDistInputs[] {
    { "A", "2D matrix with shape (M,N)"_docstring, "T" },
    { "B", "2D matrix with shape (K,N)"_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter CDistOutputs[] {
    { "C", "A 2D Matrix that represents the distance between each pair of the two collections of inputs."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint CDistTypeConstraints[] {
    { "T", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_DOUBLE>::data, "Constrains input to only numeric types."_docstring },
};

constexpr static ONNX_NAMESPACE::Attribute CropAndResizeAttributes[] {
    { "mode", "The pooling method. Two modes are supported: 'bilinear' and 'nearest'. " "Default is 'bilinear'."_docstring, AttributeProto::STRING, "bilinear" },
    { "extrapolation_value", "Value used for extrapolation, when applicable. Default is 0.0f. "_docstring, AttributeProto::FLOAT, 0.f },
};
constexpr static ONNX_NAMESPACE::FormalParameter CropAndResizeInputs[] {
    { "X", "Input data tensor from the previous operator; "
                "4-D feature map of shape (N, C, H, W), "
                "where N is the batch size, C is the number of channels, "
                "and H and W are the height and the width of the data."_docstring,
                "T1" },
    { "rois", "RoIs (Regions of Interest) to pool over; rois is "
                "2-D input of shape (num_rois, 4) given as "
                "[[y1, x1, y2, x2], ...]. "
                "The RoIs' coordinates are normalized in the coordinate system of the input image. "
                "Each coordinate set has a 1:1 correspondence with the 'batch_indices' input."_docstring,
                "T1" },
    { "batch_indices", "1-D tensor of shape (num_rois,) with each element denoting "
                "the index of the corresponding image in the batch."_docstring,
                "T2" },
    { "crop_size", "1-D tensor of 2 elements: [crop_height, crop_width]. "
                "All cropped image patches are resized to this size. Both crop_height and crop_width need to be positive."_docstring,
                "T2" },
};
constexpr static ONNX_NAMESPACE::FormalParameter CropAndResizeOutputs[] {
    { "Y",
            "RoI pooled output, 4-D tensor of shape "
            "(num_rois, C, crop_height, crop_width). The r-th batch element Y[r-1] "
            "is a pooled feature map corresponding to the r-th RoI X[r-1]."_docstring,
            "T1" },
};
constexpr static ONNX_NAMESPACE::TypeConstraint CropAndResizeTypeConstraints[] {
    { "T1", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT16, &ONNX_NAMESPACE::TypeConstraint::TENSOR_FLOAT, &ONNX_NAMESPACE::TypeConstraint::TENSOR_DOUBLE>::data, "Constrain types to float tensors."_docstring },
    { "T2", StaticStrings<&ONNX_NAMESPACE::TypeConstraint::TENSOR_INT32>::data, "Constrain types to int tensors."_docstring },
};
static void CropAndResizeTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    if (!hasNInputShapes(ctx, 4)) {
        return;
    }
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    auto& input_shape = getInputShape(ctx, 0);
    auto& rois_shape = getInputShape(ctx, 1);
    auto& batch_index_shape = getInputShape(ctx, 2);
    auto& crop_size_shape = getInputShape(ctx, 3);

    if (input_shape.dim_size() != 4) {
        fail_shape_inference("first input tensor has wrong dimension");
    }
    if (rois_shape.dim_size() != 2) {
        fail_shape_inference("rois input tensor has wrong dimension");
    }
    if (batch_index_shape.dim_size() != 1) {
        fail_shape_inference("batch_indices shape input tensor has wrong dimension");
    }
    if (crop_size_shape.dim_size() != 1) {
        fail_shape_inference("crop_size shape input tensor has wrong dimension");
    }
}

constexpr static ONNX_NAMESPACE::FormalParameter GeluInputs[] {
    { "X", "The input data as Tensor."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter GeluOutputs[] {
    { "Y", "The output."_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::FormalParameter BiasGeluInputs[] {
    { "A", "The normal input data."_docstring, "T" },
    { "B", "The bias input data that is a 1D tensor."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter BiasGeluOutputs[] {
    { "C", "The output."_docstring, "T" },
};

constexpr static ONNX_NAMESPACE::FormalParameter InverseInputs[] {
    { "X", "Input tensor. Every matrix in the batch must be invertible."_docstring, "T" },
};
constexpr static ONNX_NAMESPACE::FormalParameter InverseOutputs[] {
    { "Y", "Output tensor of the same type and shape as the input tensor."_docstring, "T" },
};
static void InverseTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    // Type inference
    using namespace ONNX_NAMESPACE;
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    // Shape inference
    if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& input_shape =
                ctx.getInputType(0)->tensor_type().shape();
        const int rank = static_cast<int>(input_shape.dim_size());

        if (rank < 2) {
            fail_shape_inference("Input rank must be >= 2.")
        }

        const auto mat_w = input_shape.dim(rank - 1);
        const auto mat_h = input_shape.dim(rank - 2);
        if (mat_w.has_dim_value() && mat_h.has_dim_value() &&
            (mat_w.dim_value() != mat_h.dim_value())) {
            fail_shape_inference(
                    "The inner-most 2 dimensions must have the same size (mat_w:",
                    mat_w.dim_value(),
                    " != mat_h:",
                    mat_h.dim_value(),
                    ").");
        }

        // Shape inference
        propagateShapeFromInputToOutput(ctx, 0, 0);
    }
}

static constexpr ONNX_NAMESPACE::CxOpSchema ContribSchemas[]{
    ONNX_NAMESPACE::CxOpSchema("Affine", "", 0)
        .SinceVersion(1)
        .Attrs(AffineAttributes)
        .Inputs(AffineInputs)
        .Outputs(AffineOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("ParametricSoftplus", "", 0)
        .SinceVersion(1)
        .Attrs(ParametricSoftplusAttributes)
        .Inputs(ParametricSoftplusInputs)
        .Outputs(ParametricSoftplusOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("ImageScaler", "", 0)
        .SinceVersion(1)
        .Attrs(ImageScalerAttributes)
        .Inputs(ImageScalerInputs)
        .Outputs(ImageScalerOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("Crop", "", 0)
        .SinceVersion(1)
        .Attrs(CropAttributes)
        .Inputs(CropInputs)
        .Outputs(CropOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("ThresholdedRelu", "", 0)
        .SinceVersion(1)
        .Attrs(ThresholdedReluAttributes)
        .Inputs(ThresholdedReluInputs)
        .Outputs(ThresholdedReluOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("DynamicSlice", "", 0)
        .SinceVersion(1)
        .Inputs(DynamicSliceInputs)
        .Outputs(DynamicSliceOutputs)
        .TypeConstraints(DynamicSliceTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("GivenTensorFill", "", 0)
        .SinceVersion(1)
        .Attrs(GivenTensorFillAttributes)
        .Inputs(GivenTensorFillInputs)
        .Outputs(GivenTensorFillOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(GivenTensorFillTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("Scale", "", 0)
        .SinceVersion(1)
        .Inputs(ScaleInputs)
        .Outputs(ScaleOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .SetDoc(Scale_ver1_doc)
        .Attrs(ScaleAttributes)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("GRUUnit", "", 0)
        .SinceVersion(1)
        .SetDoc(GRUUnit_ver1_doc)
        .Attrs(GRUUnitAttributes)
        .Inputs(GRUUnitInputs)
        .Outputs(GRUUnitOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("GivenTensorFill", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Inputs(GivenTensorFill10Inputs)
        .Outputs(GivenTensorFill10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .Attrs(GivenTensorFill10Attributes)
        .TypeAndShapeInferenceFunction(GivenTensorFill10TypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("Scale", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Inputs(Scale10Inputs)
        .Outputs(Scale10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .SetDoc(Scale_ver1_doc)
        .Attrs(Scale10Attributes)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("GRUUnit", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .SetDoc(GRUUnit_ver1_doc)
        .Attrs(GRUUnit10Attributes)
        .Inputs(GRUUnit10Inputs)
        .Outputs(GRUUnit10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("MeanVarianceNormalization", "", 0)
        .SinceVersion(1)
        .SetDoc(MeanVarienceNormalizationDoc)
        .Attrs(MeanVarienceNormalizationAttributes)
        .Inputs(MeanVarienceNormalizationInputs)
        .Outputs(MeanVarienceNormalizationOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("ScaledTanh", "", 0)
        .SinceVersion(1)
        .Attrs(ScaledTanhAttributes)
        .Inputs(ScaledTanhInputs)
        .Outputs(ScaledTanhOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("Affine", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Attrs(Affine10Attributes)
        .Inputs(Affine10Inputs)
        .Outputs(Affine10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("ParametricSoftplus", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Attrs(ParametricSoftplus10Attributes)
        .Inputs(ParametricSoftplus10Inputs)
        .Outputs(ParametricSoftplus10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("ImageScaler", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Attrs(ImageScaler10Attributes)
        .Inputs(ImageScaler10Inputs)
        .Outputs(ImageScaler10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("Crop", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Attrs(Crop10Attributes)
        .Inputs(Crop10Inputs)
        .Outputs(Crop10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(Crop10TypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("DynamicSlice", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Inputs(DynamicSlice10Inputs)
        .Outputs(DynamicSlice10Outputs)
        .TypeConstraints(DynamicSlice10TypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("ScaledTanh", "", 0)
        .SinceVersion(10)
        .Deprecate()
        .Attrs(ScaledTanh10Attributes)
        .Inputs(ScaledTanh10Inputs)
        .Outputs(ScaledTanh10Outputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),

        // End of ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old version history maintenance

    ONNX_NAMESPACE::CxOpSchema("SampleOp", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Inputs(SampleOpInputs)
        .Outputs(SampleOpOutputs)
        .TypeConstraints(SampleOpTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
        .SetDoc(SampleOpDoc),

        // register schemas for more operators here

    ONNX_NAMESPACE::CxOpSchema("MaxpoolWithMask", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(For internal use.)DOC"_docstring)
        .Attrs(MaxpoolWithMaskAttributes)
        .Inputs(MaxpoolWithMaskInputs)
        .Outputs(MaxpoolWithMaskOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(MaxpoolWithMaskTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("Rfft", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC()DOC")
        .Inputs(RfftInputs)
        .Attrs(RfftAttributes)
        .Outputs(RfftOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("Irfft", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC()DOC")
        .Inputs(IrfftInputs)
        .Attrs(IrfftAttributes)
        .Outputs(IrfftOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("ComplexMul", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC()DOC")
        .Inputs(ComplexMulInputs)
        .Outputs(ComplexMulOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("ComplexMulConj", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC()DOC")
        .Inputs(ComplexMulConjInputs)
        .Outputs(ComplexMulConjOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("ConvTransposeWithDynamicPads", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC()DOC")
        .Attrs(ConvTransposeWithDynamicPadsAttributes)
        .Inputs(ConvTransposeWithDynamicPadsInputs)
        .Outputs(ConvTransposeWithDynamicPadsOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::convTransposeWithDynamicPadsShapeInference),
    ONNX_NAMESPACE::CxOpSchema("FusedConv", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(
The fused convolution operator schema is the same as Conv besides it includes an attribute
activation.)DOC"_docstring)
        .Attrs(FusedConvAttributes)
        .Inputs(FusedConvInputs)
        .Outputs(FusedConvOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(FusedConvTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("FusedGemm", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(
The FusedGemm operator schema is the same as Gemm besides it includes attributes
activation and leaky_relu_alpha.)DOC"_docstring)
        .Inputs(FusedGemmInputs)
        .Outputs(FusedGemmOutputs)
        .TypeConstraints(FloatingPointIntTTypeConstraints)
        .Attrs(FusedGemmAttributes)
        .TypeAndShapeInferenceFunction(FusedGemmTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("ExpandDims", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Inputs(ExpandDimsInputs)
        .Outputs(ExpandDimsOutputs)
        .TypeConstraints(ExpandDimsTypeConstraints)
        .TypeAndShapeInferenceFunction(ExpandDimsTypeAndShapeInference)
        .SetDoc(R"DOC(ExpandDims echo operator.)DOC"_docstring),
    ONNX_NAMESPACE::CxOpSchema("QuantizeLinear", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Attrs(QuantizeLinearAttributes)
        .Inputs(QuantizeLinearInputs)
        .Outputs(QuantizeLinearOutputs)
        .TypeConstraints(QuantizeLinearTypeConstraints)
        .SetDoc(R"DOC(
The linear quantization operator. It consumes a full precision data, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC"_docstring)
        .TypeAndShapeInferenceFunction(QuantizeLinearTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("DequantizeLinear", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Attrs(DequantizeLinearAttributes)
        .Inputs(DequantizeLinearInputs)
        .Outputs(DequantizeLinearOutputs)
        .TypeConstraints(DequantizeLinearTypeConstraints)
        .SetDoc(R"DOC(
The linear dequantization operator. It consumes a quantized data, a scale, a zero point and computes the full precision data.
The dequantization formula is y = (x - x_zero_point) * x_scale.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').)DOC"_docstring)
        .TypeAndShapeInferenceFunction(DequantizeLinearTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("Tokenizer", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Inputs(TokenizerInputs)
        .Outputs(TokenizerOutputs)
        .Attrs(TokenizerAttributes)
        .SetDoc(R"DOC(
  Tokenizer divides each string in X into a vector of strings along the last axis. Allowed input shapes are [C] and [N, C].
  If the maximum number of tokens found per input string is D, the output shape would be [N, C, D] when input shape is [N, C].
  Similarly, if input shape is [C] then the output should be [C, D]. Tokenizer has two different operation modes.
  The first mode is selected when "tokenexp" is not set and "separators" is set. If "tokenexp" is set and "separators" is not set,
  the second mode will be used. The first mode breaks each input string into tokens by matching and removing separators.
  "separators" is a list of strings which are regular expressions. "tokenexp" is a single regular expression.
  Let's assume "separators" is [" "] and consider an example.
  If input is
  ["Hello World", "I love computer science !"] whose shape is [2],
  then the output would be
 [["Hello", "World", padvalue, padvalue, padvalue],
 ["I", "love", "computer", "science", "!"]]
 whose shape is [2, 5] because you can find at most 5 tokens per input string.
 Note that the input at most can have two axes, so 3-D and higher dimension are not supported.
 If "separators" contains a single empty string, the Tokenizer will enter into character tokenezation mode. This means all strings
 will be broken part into individual characters.
 For each input string, the second mode searches matches of "tokenexp" and each match will be a token in Y.
 The matching of "tokenexp" is conducted greedily (i.e., a match should be as long as possible).
 This operator searches for the first match starting from the beginning of the considered string,
 and then launches another search starting from the first remained character after the first matched token.
 If no match found, this operator will remove the first character from the remained string and do another search.
 This procedure will be repeated until reaching the end of the considered string.
  Let's consider another example to illustrate the effect of setting "mark" to true.
  If input is ["Hello", "World"],
  then the corresponding output would be [0x02, "Hello", "World", 0x03].
  This implies that if mark is true, [C]/[N, C] - input's output shape becomes [C, D+2]/[N, C, D+2].
If tokenizer removes the entire content of [C]-input, it will produce [[]].
I.e. the output shape should be [C][0] or [N][C][0] if input shape was [N][C].
If the tokenizer receives empty input of [0] then the output is [0] if empty input
of [N, 0] then [N, 0].)DOC"_docstring)
        .TypeAndShapeInferenceFunction(TokenizerTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("MatMulInteger16", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
 The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.)DOC"_docstring)
        .Inputs(MatMulInteger16Inputs)
        .Outputs(MatMulInteger16Outputs)
        .TypeConstraints(MatMulInteger16TypeConstraints)
        .TypeAndShapeInferenceFunction(MatMulInteger16TypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("DynamicQuantizeMatMul", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Inputs(DynamicQuantizeMatMulInputs)
        .Outputs(DynamicQuantizeMatMulOutputs)
        .TypeConstraints(DynamicQuantizeMatMulTypeConstraints)
        .TypeAndShapeInferenceFunction(DynamicQuantizeMatMulTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("MatMulIntegerToFloat", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Inputs(MatMulIntegerToFloatInputs)
        .Outputs(MatMulIntegerToFloatOutputs)
        .TypeConstraints(MatMulIntegerToFloatTypeConstraints)
        .TypeAndShapeInferenceFunction(MatMulIntegerToFloatTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("TransposeMatMul", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetDoc("TransposeMatMul"_docstring)
        .Attrs(TransposeMatMulAttributes)
        .Inputs(TransposeMatMulInputs)
        .Outputs(TransposeMatMulOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html)DOC"_docstring)
        .TypeAndShapeInferenceFunction(TransposeMatMulTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("ReduceSumInteger", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(
Computes the sum of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.)DOC"_docstring)
        .Inputs(ReduceSumIntegerInputs)
        .Outputs(ReduceSumIntegerOutputs)
        .TypeConstraints(ReduceSumIntegerOutputsTypeConstraints)
        .Attrs(ReduceSumIntegerAttributes),
    ONNX_NAMESPACE::CxOpSchema("QLinearReduceMean", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(
Computes the mean of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.
Input and Output scales and zero points are used to requantize the output in a new range.
This helps to improve accuracy as after ReduceMean operation the range of the output is expected to decrease.

```
"Output = Dequantize(Input) -> ReduceMean on fp32 data -> Quantize(output)",

```)DOC"_docstring)
        .Inputs(QLinearReduceMeanInputs)
        .Outputs(QLinearReduceMeanOutputs)
        .TypeConstraints(QLinearReduceMeanOutputsTypeConstraints)
        .Attrs(QLinearReduceMeanAttributes)
        .TypeAndShapeInferenceFunction(QLinearTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("MulInteger", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(Performs element-wise binary quantized multiplication (with Numpy-style broadcasting support).
"This operator supports **multidirectional (i.e., Numpy-style) broadcasting**"
The output of this op is the int32 accumulated result of the mul operation

```
C (int32) = (A - A_zero_point) * (B - B_zero_point)
```)DOC"_docstring)
        .Inputs(MulIntegerInputs)
        .Outputs(MulIntegerOutputs)
        .TypeConstraints(MulIntegerOutputsTypeConstraints)
        .TypeAndShapeInferenceFunction(MulIntegerTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("QLinearAveragePool", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(
 QLinearAveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```

The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).

Input and output scales and zero points are used to convert the output to a new quantization range.
Output = Dequantize(Input) -> AveragePool on fp32 data -> Quantize(output))DOC"_docstring)
        .Attrs(QLinearAveragePoolAttributes)
        .Inputs(QLinearAveragePoolInputs)
        .Outputs(QLinearAveragePoolOutputs)
        .TypeConstraints(QLinearAveragePoolOutputsTypeConstraints)
        .TypeAndShapeInferenceFunction(QLinearAveragePoolTypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("QLinearLeakyRelu", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(
QLinearLeakyRelu takes quantized input data (Tensor), an argument alpha, and quantize parameter for output,
and produces one output data (Tensor<T>) where the function `f(x) = quantize(alpha * dequantize(x)) for dequantize(x) < 0`,
`f(x) = quantize(dequantize(x)) for dequantize(x) >= 0`, is applied to the data tensor elementwise.)DOC"_docstring)
        .Attrs(QLinearLeakyReluAttributes)
        .Inputs(QLinearLeakyReluInputs)
        .Outputs(QLinearLeakyReluOutputs)
        .TypeConstraints(QLinearLeakyReluOutputsTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("MurmurHash3", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(The underlying implementation is MurmurHash3_x86_32 generating low latency 32bits hash suitable for implementing lookup tables, Bloom filters, count min sketch or feature hashing.)DOC"_docstring)
        .Inputs(MurmurHash3Inputs)
        .Outputs(MurmurHash3Outputs)
        .TypeConstraints(MurmurHash3OutputsTypeConstraints)
        .Attrs(MurmurHash3Attributes)
        .TypeAndShapeInferenceFunction(MurmurHash3TypeAndShapeInference),
    ONNX_NAMESPACE::CxOpSchema("GatherND", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Inputs(GatherNDInputs)
        .Outputs(GatherNDOutputs)
        .TypeConstraints(GatherNDOutputsTypeConstraints)
        .TypeAndShapeInferenceFunction(GatherNDTypeAndShapeInference)
        .SetDoc(R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q >= 1, gather
slices of `data` into an output tensor of rank q - 1 + r - indices[-1].
Example 1:
  data    = [[0,1],[2,3]]
  indices = [[0,0],[1,1]]
  output  = [0,3]
Example 2:
  data    = [[0,1],[2,3]]
  indices = [[1],[0]]
  output  = [[2,3],[0,1]]
Example 3:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[0,1],[1,0]]
  output  = [[2,3],[4,5]]
Example 4:
  data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
  indices = [[[0,1]],[[1,0]]]
  output  = [[[2,3]],[[4,5]]])DOC"_docstring),
    ONNX_NAMESPACE::CxOpSchema("WordConvEmbedding", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Attrs(WordConvEmbeddingAttributes)
        .Inputs(WordConvEmbeddingInputs)
        .Outputs(WordConvEmbeddingOutputs)
        .TypeConstraints(WordConvEmbeddingOutputsTypeConstraints)
        .SetDoc(R"DOC(The WordConvEmbedding takes in a batch of sequence words and embed each word to a vector.)DOC"_docstring),
    ONNX_NAMESPACE::CxOpSchema("Pad", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Attrs(PadAttributes)
        .Inputs(PadInputs)
        .Outputs(PadOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(PadTypeAndShapeInference)
        .SetDoc(R"DOC(
            Given `data` tensor, pads, mode, and value.
            Example:
            Insert 0 pads to the beginning of the second dimension.
            data = [
                    [1.0, 1.2],
                    [2.3, 3.4],
                    [4.5, 5.7],
                    ]
            pads = [0, 2, 0, 0]
            output = [
                    [
                    [0.0, 0.0, 1.0, 1.2],
                    [0.0, 0.0, 2.3, 3.4],
                    [0.0, 0.0, 4.5, 5.7],
                    ],
                    ])DOC"_docstring),
    ONNX_NAMESPACE::CxOpSchema("Unique", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Inputs(UniqueInputs)
        .Outputs(UniqueOutputs)
        .TypeConstraints(UniqueTypeConstraints)
        .TypeAndShapeInferenceFunction(UniqueTypeAndShapeInference)
        .SetDoc(R"DOC(
              Finds all the unique values (deduped list) present in the given input tensor.
              This operator returns 3 outputs.
              The first output tensor 'uniques' contains all of the unique elements of the input,
              sorted in the same order that they occur in the input.
              The second output tensor 'idx' is the same size as the input and it contains the index
              of each value of the input in 'uniques'.
              The third output tensor 'counts' contains the count of each element of 'uniques' in the input.
              Example:
                input_x = [2, 1, 1, 3, 4, 3]
                output_uniques = [2, 1, 3, 4]
                output_idx = [0, 1, 1, 2, 3, 2]
                output_counts = [1, 2, 2, 1])DOC"_docstring),
        //see:https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    ONNX_NAMESPACE::CxOpSchema("CDist", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Attrs(CDistAttributes)
        .Inputs(CDistInputs)
        .Outputs(CDistOutputs)
        .TypeConstraints(CDistTypeConstraints),
    ONNX_NAMESPACE::CxOpSchema("CropAndResize", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .Attrs(CropAndResizeAttributes)
        .Inputs(CropAndResizeInputs)
        .Outputs(CropAndResizeOutputs)
        .TypeConstraints(CropAndResizeTypeConstraints)
        .TypeAndShapeInferenceFunction(CropAndResizeTypeAndShapeInference)
        .SetDoc(R"DOC(
        Extracts crops from the input image tensor and resizes them using bilinear sampling or nearest neighbor sampling
        (possibly with aspect ratio change) to a common output size specified by crop_height and crop_width.
        Returns a tensor with crops from the input image at positions defined at the bounding box locations in boxes.
        The cropped boxes are all resized (with bilinear or nearest neighbor interpolation) to
        a fixed size = [crop_height, crop_width]. The result is a 4-D tensor [num_boxes, crop_height, crop_width, depth].
        The resizing is corner aligned.)DOC"_docstring),
    ONNX_NAMESPACE::CxOpSchema("Gelu", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetDoc(R"DOC(Gaussian Error Linear Unit.
A high-performing neural network activation function.The GELU nonlinearity is
the expected transformation of a stochastic regularizer which randomly applies
the identity or zero map to a neuron's input. The GELU nonlinearity weights
inputs by their magnitude, rather than gates inputs by their sign as in ReLUs.)DOC"_docstring)
        .Inputs(GeluInputs)
        .Outputs(GeluOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),
    ONNX_NAMESPACE::CxOpSchema("BiasGelu", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetDoc(R"DOC(Bias Gelu.
It's an extension of Gelu. It takes the sum of input A and bias input B as the input of Gelu activation. )DOC"_docstring)
        .Inputs(BiasGeluInputs)
        .Outputs(BiasGeluOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput),

        // Used to be ONNX 1.7 Inverse(12)
        // Comment out docs not to increase the binary size
        //
        //  static constexpr const char* Inverse_ver1_doc = R"DOC(
        //Calculates inverse of a square matrix or batches of square matrices.
        //Inverse takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
        //and the inner-most 2 dimensions form square matrices. These matrices must be invertible (full-rank).
        //The behavior where one of the matrices is not invertible is undefined. The implementation can choose
        //to throw an error or output (garbage) results as is. The output is a tensor of shape `[*, M, M]`,
        //containing the individual inverses of all input submatrices.
        //)DOC";

    ONNX_NAMESPACE::CxOpSchema("Inverse", "", 0)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .Inputs(InverseInputs)
        .Outputs(InverseOutputs)
        .TypeConstraints(FloatingPointTTypeConstraints)
        .TypeAndShapeInferenceFunction(InverseTypeAndShapeInference),
};

void RegisterContribSchemas() {
  // Register removed experimental ops for backward compatibility.
  // Experimental operators do not have version history. However, RS5 takes bunch of experimental operators
  // as production ops. In order to maintain backward compatibility when the experimental ops are removed from ONNX
  // they need to be added in onnxruntime as contrib ops.
  // ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old version history maintenance

  for (auto& cx_schema : ContribSchemas) {
    Register(cx_schema);
  }

  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(AttnLSTM, RegisterAttnLSTMContribOpSchema);
  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(Range, RegisterRangeOpSchema);

  ONNX_NAMESPACE::OpSchema("QLinearAdd", "", 0)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .FillUsing(QLinearMathDocGenerator("addition",
                                         "C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point"_docstring));

  ONNX_NAMESPACE::OpSchema("QLinearMul", "", 0)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .FillUsing(QLinearMathDocGenerator("multiplication",
                                         "C = ((A - A_zero_point) * (B - B_zero_point)) * (A_scale * B_scale)/C_scale + C_zero_point"_docstring));

  ONNX_NAMESPACE::OpSchema("LayerNormalization", "", 0)
    .SetDomain(kOnnxDomain)
    .SinceVersion(1)
    .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
    .SetDoc("LayerNormalization")
    .Attr("axis",
           "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs)."_docstring,
           AttributeProto::INT, static_cast<int64_t>(-1))
    .Attr("epsilon",
           "The epsilon value to use to avoid division by zero."_docstring,
           AttributeProto::FLOAT, 1e-5f)
    .AllowUncheckedAttributes()
    .Input(0, "X", "Input data tensor from the previous layer."_docstring, "T")
    .Input(1, "scale", "Scale tensor."_docstring, "T")
    .Input(2, "B", "Bias tensor."_docstring, "T")
    .Output(0, "Y", "Output data tensor."_docstring, "T")
    .Output(1, "mean", "Saved mean used during training to speed up gradient computation"_docstring, "U", OpSchema::Optional)
    .Output(2, "inv_std_var", "Saved inverse standard variance used during training to speed up gradient computation."_docstring, "U", OpSchema::Optional)
    .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types (except mean and inv_std_var) to float tensors."_docstring)
    .TypeConstraint(
            "U",
            {"tensor(float)"},
            "Constrain mean and inv_std_var to be float tensors."_docstring)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
            return;
        }
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        int64_t input_ndim = input_shape.dim_size();
        int64_t axis = -1;
        auto axis_proto = ctx.getAttribute("axis");
        if (axis_proto) {
            axis = axis_proto->i();
        }
        if (axis < 0) {
            axis += input_ndim;
        }

        if (ctx.getNumOutputs() > 1) {
            auto saved_mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
            saved_mean_shape->CopyFrom(input_shape);
            saved_mean_shape->mutable_dim(static_cast<int>(axis))->set_dim_value(1);
        }

        if (ctx.getNumOutputs() > 2) {
            auto saved_inv_std_var_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
            saved_inv_std_var_shape->CopyFrom(input_shape);
            saved_inv_std_var_shape->mutable_dim(static_cast<int>(axis))->set_dim_value(1);
        }
    });

    // Register the NCHWc schemas if supported by the platform.
    if (MlasNchwcGetBlockSize() > 1) {
        RegisterNchwcSchemas();
    }

  RegisterBertSchemas();
}
}  // namespace contrib
}  // namespace onnxruntime
