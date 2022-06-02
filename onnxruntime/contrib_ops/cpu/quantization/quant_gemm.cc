// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cpu/math/gemm_base.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/quantization/matmul_integer_base.h"
#include "core/quantization/quantization.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

class QGemm : protected GemmBase, public MatMulIntegerBase {
 public:
  QGemm(const OpKernelInfo& info) : GemmBase(info), MatMulIntegerBase(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* a = context->Input<Tensor>(IN_A);
    const auto* b = packed_b_ ? nullptr : context->Input<Tensor>(IN_B);
    const auto& b_shape = b ? b->Shape() : b_shape_;

    const auto* c = context->Input<Tensor>(IN_C);
    GemmHelper helper(a->Shape(), trans_A_ != CblasNoTrans,
                      b_shape, trans_B_ != CblasNoTrans,
                      c != nullptr ? c->Shape() : TensorShape({}));
    if (!helper.State().IsOK())
      return helper.State();

    size_t M = SafeInt<size_t>(helper.M());
    size_t N = SafeInt<size_t>(helper.N());
    size_t K = SafeInt<size_t>(helper.K());

    //validate scales and zero points
    const auto* a_zp = context->Input<Tensor>(IN_A_ZERO_POINT);
    const auto* b_zp = context->Input<Tensor>(IN_B_ZERO_POINT);
    const auto* y_zp = context->Input<Tensor>(IN_Y_ZERO_POINT);
    const auto* a_scale = context->Input<Tensor>(IN_A_SCALE);
    const auto* b_scale = context->Input<Tensor>(IN_B_SCALE);
    const auto* y_scale = context->Input<Tensor>(IN_Y_SCALE);
    CheckInputs(a_zp, b_zp, y_zp, a_scale, b_scale, y_scale, helper);

    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

    bool a_is_signed = a->IsDataType<int8_t>();
    const uint8_t* a_data = static_cast<const uint8_t*>(a->DataRaw());

    BufferUniquePtr a_trans_buffer;
    if (trans_A_ == CblasTrans) {
      a_data = quantization::TransPoseInputData(a_data, a_trans_buffer, allocator, K, M);
    }

    bool b_is_signed;
    const uint8_t* b_data = nullptr;
    BufferUniquePtr b_trans_buffer;
    if (nullptr == b) {
      b_data = static_cast<const uint8_t*>(packed_b_.get());
      b_is_signed = b_is_signed_;
    } else {
      b_data = static_cast<const uint8_t*>(b->DataRaw());
      b_is_signed = b->IsDataType<int8_t>();
      if (trans_B_ == CblasTrans) {
        b_data = quantization::TransPoseInputData(b_data, b_trans_buffer, allocator, N, K);
      }
    }

    auto y = context->Output(OUT_Y, {SafeInt<int64_t>(M), SafeInt<int64_t>(N)});
    if (M == 0 || N == 0) return Status::OK();

    // prepare output buffer of GEMM
    int32_t* gemm_output_data = nullptr;
    BufferUniquePtr gemm_output_buffer;
    bool need_requant = y_scale != nullptr;
    if (need_requant) {
      gemm_output_data = static_cast<int32_t*>(allocator->Alloc(SafeInt<size_t>(M * N) * sizeof(int32_t)));
      gemm_output_buffer.reset(gemm_output_data);
    } else {
      gemm_output_data = static_cast<int32_t*>(y->MutableDataRaw());
    }

    if (c != nullptr) {
      GemmBroadcastBias(M, N, 1.f, c->template Data<int32_t>(), &(c->Shape()), gemm_output_data);
    }

    MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape{M, N, K, a_is_signed, b_is_signed, c != nullptr};
    MLAS_GEMM_QUANT_DATA_PARAMS gemm_param;

    gemm_param.A = a_data;
    gemm_param.lda = gemm_shape.K;
    gemm_param.ZeroPointA = *(static_cast<const uint8_t*>(a_zp->DataRaw()));

    gemm_param.B = b_data;
    gemm_param.ldb = gemm_shape.N;
    gemm_param.BIsPacked = bool(packed_b_);
    gemm_param.ZeroPointB = static_cast<const uint8_t*>(b_zp->DataRaw());

    gemm_param.C = gemm_output_data;
    gemm_param.ldc = gemm_shape.N;

    gemm_param.PerColumnZeroPoints = !IsScalarOr1ElementVector(b_zp);

    std::vector<float> output_scales = ComputeOutputScale(a_scale, b_scale, y_scale);
    std::unique_ptr<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> scale_bias_proc_ptr;
    std::unique_ptr<MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR> requant_proc_ptr;
    std::unique_ptr<MLAS_REQUANT_PARAM> requant_param;
    SetPostProcessor(y_zp, N, output_scales, y, gemm_param, scale_bias_proc_ptr, requant_proc_ptr, requant_param);

    MlasGemmBatch(gemm_shape, &gemm_param, 1, context->GetOperatorThreadPool());
    return Status::OK();
  }

 protected:
  int GetBIdx() const override {
    return IN_B;
  }

  virtual bool IsBTransposed() const override {
    return trans_B_ == CblasTrans;
  }

 private:
  enum InputTensors : int {
    IN_A = 0,
    IN_A_SCALE = 1,
    IN_A_ZERO_POINT = 2,
    IN_B = 3,
    IN_B_SCALE = 4,
    IN_B_ZERO_POINT = 5,
    IN_C = 6,
    IN_Y_SCALE = 7,
    IN_Y_ZERO_POINT = 8
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  static void CheckInputs(const Tensor* a_zp, const Tensor* b_zp, const Tensor* y_zp,
                          const Tensor* a_scale, const Tensor* b_scale, const Tensor* y_scale, const GemmHelper& helper) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
                "QGemm : scale of input a must be a scalar or 1D tensor of size 1");
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zp),
                "QGemm : zero point of input a must be a scalar or 1D tensor of size 1");

    const auto& b_zp_shape = b_zp->Shape();
    const auto& b_scale_shape = b_scale->Shape();
    ORT_ENFORCE(b_zp_shape.NumDimensions() == 0 ||
                    (b_zp_shape.NumDimensions() == 1 && (b_zp_shape[0] == 1 || b_zp_shape[0] == helper.N())),
                "QGemm : zero point of input b must be a scalar or 1D tensor of size 1 or N");
    ORT_ENFORCE(b_scale_shape.NumDimensions() == 0 ||
                    (b_scale_shape.NumDimensions() == 1 && (b_scale_shape[0] == 1 || b_scale_shape[0] == helper.N())),
                "QGemm : scale of input b must be a scalar or 1D tensor of size 1 or N");
    ORT_ENFORCE(b_scale_shape.NumDimensions() == b_zp_shape.NumDimensions() &&
                    (b_scale_shape.NumDimensions() == 0 || (b_scale_shape[0] == b_zp_shape[0])),
                "QGemm : zero point and scale of input b should have same shape size");

    ORT_ENFORCE(y_zp == nullptr || IsScalarOr1ElementVector(y_zp),
                "QGemm : zero point of y must be null or a scalar or 1D tensor of size 1");
    ORT_ENFORCE(y_scale == nullptr || IsScalarOr1ElementVector(y_scale),
                "QGemm : scale of y must be null or a scalar or 1D tensor of size 1");
  }

  std::vector<float> ComputeOutputScale(const Tensor* a_scale, const Tensor* b_scale, const Tensor* y_scale) const {
    const int64_t output_scale_size = b_scale->Shape().Size();
    std::vector<float> output_scales(output_scale_size);
    auto a_scale_value = *(a_scale->template Data<float>());
    const auto* b_scale_data = b_scale->template Data<float>();
    for (int64_t i = 0; i < output_scale_size; i++) {
      output_scales[i] = (alpha_ * a_scale_value * b_scale_data[i]);
      if (nullptr != y_scale) {
        output_scales[i] /= *(y_scale->template Data<float>());
      }
    }
    return output_scales;
  }

  static void SetPostProcessor(const Tensor* y_zp,
                               size_t out_lda,
                               const std::vector<float>& output_scales,
                               Tensor* y,
                               MLAS_GEMM_QUANT_DATA_PARAMS& gemm_param,
                               std::unique_ptr<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR>& scale_bias_proc_ptr,
                               std::unique_ptr<MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR>& requant_proc_ptr,
                               std::unique_ptr<MLAS_REQUANT_PARAM>& requant_param) {
    if (nullptr != y_zp) {
      bool is_y_signed = y->IsDataType<int8_t>();
      int32_t y_zero_point = is_y_signed ? *y_zp->template Data<int8_t>() : *y_zp->template Data<uint8_t>();
      requant_param = std::make_unique<MLAS_REQUANT_PARAM>(output_scales.data(), output_scales.size(), y_zero_point);
      requant_proc_ptr = std::make_unique<MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR>(
          y->MutableDataRaw(),
          out_lda,
          nullptr,
          requant_param.get(),
          is_y_signed);
      gemm_param.OutputProcessor = requant_proc_ptr.get();
    } else {
      scale_bias_proc_ptr = std::make_unique<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR>(
          static_cast<float*>(y->MutableDataRaw()),
          out_lda,
          output_scales.data(),
          nullptr,
          MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
          output_scales.size() > 1 ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
      gemm_param.OutputProcessor = scale_bias_proc_ptr.get();
    }
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QGemm,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TA", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TB", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("TC", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TYZ", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TY", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<uint8_t>()}),
    QGemm);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QGemm,
    kMSDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TA", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TB", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TC", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TYZ", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TY", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<int8_t>()}),
    QGemm);

}  // namespace contrib
}  // namespace onnxruntime
