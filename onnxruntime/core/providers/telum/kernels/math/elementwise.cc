// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

#include <algorithm>

namespace onnxruntime {
namespace telum {

/**
 * @brief Base class for elementwise binary operations
 */
template <typename OpFunc>
class BinaryElementwise : public TelumKernel {
 public:
  explicit BinaryElementwise(const OpKernelInfo& info) : TelumKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = context->Input<Tensor>(1);

    ORT_RETURN_IF_NOT(A != nullptr, "Input A is null");
    ORT_RETURN_IF_NOT(B != nullptr, "Input B is null");

    // Validate static shapes
    ORT_RETURN_IF_ERROR(ValidateStaticShape(A->Shape()));
    ORT_RETURN_IF_ERROR(ValidateStaticShape(B->Shape()));

    // Compute broadcasted output shape using ONNX/Numpy broadcasting rules.
    std::vector<int64_t> out_dims;
    bool needs_broadcast = false;
    ORT_RETURN_IF_ERROR(ComputeBroadcastShape(A->Shape().GetDims(), B->Shape().GetDims(), out_dims, needs_broadcast));

    TensorShape output_shape(out_dims);
    Tensor* Y = context->Output(0, output_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // zDNN elementwise ops do not do broadcasting. If no broadcasting is needed and rank <= 4, use zDNN.
    // Otherwise, compute on CPU (still within Telum EP) for correctness.
    if (!needs_broadcast) {
      if (output_shape.NumDimensions() > 4) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "Telum EP: elementwise ops only support rank <= 4, got rank ",
                               output_shape.NumDimensions());
      }

      // Determine layout
      const zdnn_data_layouts layout = TensorConverter::GetLayoutForShape(output_shape);

      // Convert tensors
      zdnn_ztensor zdnn_a, zdnn_b, zdnn_y;
      ORT_RETURN_IF_ERROR(ConvertToZTensor(*A, zdnn_a, layout));
      ZTensorGuard guard_a(&zdnn_a);

      ORT_RETURN_IF_ERROR(ConvertToZTensor(*B, zdnn_b, layout));
      ZTensorGuard guard_b(&zdnn_b);

      ORT_RETURN_IF_ERROR(InitZTensorForOutput(*Y, zdnn_y, layout));
      ZTensorGuard guard_y(&zdnn_y);

      // Execute operation
      const zdnn_status status = OpFunc::Execute(&zdnn_a, &zdnn_b, &zdnn_y);
      ORT_RETURN_IF_ERROR(CheckStatus(status, OpFunc::Name()));

      // Convert result
      ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

      return Status::OK();
    }

    // CPU broadcast path: support up to rank 4 for now (sufficient for common transformer patterns).
    if (output_shape.NumDimensions() > 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Telum EP: broadcast elementwise ops only support rank <= 4, got rank ",
                             output_shape.NumDimensions());
    }

    ORT_RETURN_IF_NOT(A->GetElementType() == B->GetElementType(), "Telum EP: elementwise input types must match");
    ORT_RETURN_IF_NOT(A->GetElementType() == Y->GetElementType(), "Telum EP: elementwise output type mismatch");

    const int32_t ort_type = A->GetElementType();
    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return BroadcastBinaryOp<float>(*A, *B, *Y);
    }
    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      return BroadcastBinaryOp<MLFloat16>(*A, *B, *Y);
    }
    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      return BroadcastBinaryOp<BFloat16>(*A, *B, *Y);
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported elementwise type for Telum EP");
  }

 private:
  static Status ComputeBroadcastShape(const std::vector<int64_t>& a_dims,
                                      const std::vector<int64_t>& b_dims,
                                      std::vector<int64_t>& out_dims,
                                      bool& needs_broadcast) {
    const size_t rank_a = a_dims.size();
    const size_t rank_b = b_dims.size();
    const size_t out_rank = std::max(rank_a, rank_b);
    out_dims.assign(out_rank, 1);
    needs_broadcast = (rank_a != rank_b);

    for (size_t i = 0; i < out_rank; ++i) {
      const int64_t da = (i < rank_a) ? a_dims[rank_a - 1 - i] : 1;
      const int64_t db = (i < rank_b) ? b_dims[rank_b - 1 - i] : 1;

      if (da == db) {
        out_dims[out_rank - 1 - i] = da;
      } else if (da == 1) {
        out_dims[out_rank - 1 - i] = db;
        needs_broadcast = true;
      } else if (db == 1) {
        out_dims[out_rank - 1 - i] = da;
        needs_broadcast = true;
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Telum EP: incompatible broadcast dimensions for ",
                               OpFunc::Name(), ": ", TensorShape(a_dims).ToString(),
                               " and ", TensorShape(b_dims).ToString());
      }
    }

    return Status::OK();
  }

  static std::vector<int64_t> AlignDims(const std::vector<int64_t>& dims, size_t out_rank) {
    if (dims.size() >= out_rank) return dims;
    std::vector<int64_t> aligned(out_rank - dims.size(), 1);
    aligned.insert(aligned.end(), dims.begin(), dims.end());
    return aligned;
  }

  static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 0);
    int64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
      strides[i] = stride;
      stride *= dims[i];
    }
    return strides;
  }

  template <typename T>
  static Status BroadcastBinaryOp(const Tensor& a, const Tensor& b, Tensor& y) {
    const auto& out_dims = y.Shape().GetDims();
    const size_t out_rank = out_dims.size();

    const auto a_aligned = AlignDims(a.Shape().GetDims(), out_rank);
    const auto b_aligned = AlignDims(b.Shape().GetDims(), out_rank);

    auto a_strides = ComputeStrides(a_aligned);
    auto b_strides = ComputeStrides(b_aligned);
    const auto out_strides = ComputeStrides(out_dims);

    // Broadcast dims use stride 0 so the same element is reused.
    for (size_t d = 0; d < out_rank; ++d) {
      if (a_aligned[d] == 1) a_strides[d] = 0;
      if (b_aligned[d] == 1) b_strides[d] = 0;
    }

    const int64_t out_size = y.Shape().Size();
    const T* a_data = a.Data<T>();
    const T* b_data = b.Data<T>();
    T* y_data = y.MutableData<T>();

    for (int64_t idx = 0; idx < out_size; ++idx) {
      int64_t t = idx;
      int64_t off_a = 0;
      int64_t off_b = 0;

      for (size_t d = 0; d < out_rank; ++d) {
        const int64_t coord = (out_strides[d] == 0) ? 0 : (t / out_strides[d]);
        t = (out_strides[d] == 0) ? 0 : (t % out_strides[d]);
        off_a += coord * a_strides[d];
        off_b += coord * b_strides[d];
      }

      const float av = static_cast<float>(a_data[static_cast<size_t>(off_a)]);
      const float bv = static_cast<float>(b_data[static_cast<size_t>(off_b)]);
      const float yv = OpFunc::Compute(av, bv);
      y_data[static_cast<size_t>(idx)] = T(yv);
    }

    return Status::OK();
  }
};

// Operation functors
struct AddOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_add(a, b, y);
  }
  static const char* Name() { return "zdnn_add"; }
  static float Compute(float a, float b) { return a + b; }
};

struct SubOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_sub(a, b, y);
  }
  static const char* Name() { return "zdnn_sub"; }
  static float Compute(float a, float b) { return a - b; }
};

struct MulOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_mul(a, b, y);
  }
  static const char* Name() { return "zdnn_mul"; }
  static float Compute(float a, float b) { return a * b; }
};

struct DivOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_div(a, b, y);
  }
  static const char* Name() { return "zdnn_div"; }
  static float Compute(float a, float b) { return a / b; }
};

struct MinOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_min(a, b, y);
  }
  static const char* Name() { return "zdnn_min"; }
  static float Compute(float a, float b) { return std::min(a, b); }
};

struct MaxOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_max(a, b, y);
  }
  static const char* Name() { return "zdnn_max"; }
  static float Compute(float a, float b) { return std::max(a, b); }
};

// Concrete kernel classes
using Add = BinaryElementwise<AddOp>;
using Sub = BinaryElementwise<SubOp>;
using Mul = BinaryElementwise<MulOp>;
using Div = BinaryElementwise<DivOp>;
using Min = BinaryElementwise<MinOp>;
using Max = BinaryElementwise<MaxOp>;

// Register Add kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Add,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Add);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Add,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Add);

ONNX_OPERATOR_KERNEL_EX(
    Add,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Add);

// Register Sub kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sub,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sub);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sub,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sub);

ONNX_OPERATOR_KERNEL_EX(
    Sub,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sub);

// Register Mul kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Mul);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Mul);

ONNX_OPERATOR_KERNEL_EX(
    Mul,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Mul);

// Register Div kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Div,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Div);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Div,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Div);

ONNX_OPERATOR_KERNEL_EX(
    Div,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Div);

// Register Min kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Min,
    kOnnxDomain,
    8, 11,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Min);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Min,
    kOnnxDomain,
    12, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Min);

ONNX_OPERATOR_KERNEL_EX(
    Min,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Min);

// Register Max kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Max,
    kOnnxDomain,
    8, 11,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Max);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Max,
    kOnnxDomain,
    12, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Max);

ONNX_OPERATOR_KERNEL_EX(
    Max,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Max);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
