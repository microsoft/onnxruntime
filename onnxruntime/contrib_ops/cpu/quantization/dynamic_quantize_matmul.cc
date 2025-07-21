// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/quantization/matmul_integer_base.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {

namespace {
void ScaleOutput(const Tensor& scale, Tensor& output) {
  ProcessBroadcastSpanFuncs funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.ScalarInput0<float>() * per_iter_bh.EigenInput1<float>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().array() * per_iter_bh.ScalarInput1<float>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<float>() = per_iter_bh.EigenInput0<float>().cwiseProduct(per_iter_bh.EigenInput1<float>());
      }};

  InputBroadcaster input_broadcaster(scale, output);
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(),
                                       output);
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster);

  BroadcastLooper(broadcast_helper, funcs);
}
}  // namespace

class MatMulIntegerToFloatBase : public MatMulIntegerBase {
 public:
  MatMulIntegerToFloatBase(const OpKernelInfo& info) : MatMulIntegerBase(info) {
  }

  enum OutputTensors : int { OUT_Y = 0 };

 protected:
  Status ComputeCommon(OpKernelContext* ctx,
                       const uint8_t* a_data,
                       const TensorShape& a_shape,
                       float a_scale,
                       uint8_t a_zp,
                       bool a_is_signed,
                       const Tensor* b_tensor,
                       const Tensor* b_scale,
                       const Tensor* b_zp,
                       const Tensor* bias_tensor) const;
};

Status MatMulIntegerToFloatBase::ComputeCommon(OpKernelContext* ctx,
                                               const uint8_t* a_data,
                                               const TensorShape& a_shape,
                                               float a_scale,
                                               uint8_t a_zp,
                                               bool a_is_signed,
                                               const Tensor* b_tensor,
                                               const Tensor* b_scale_tensor,
                                               const Tensor* b_zp_tensor,
                                               const Tensor* bias_tensor) const {
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape,
                                     b_tensor ? b_tensor->Shape() : b_shape_,
                                     b_scale_tensor ? &b_scale_tensor->Shape() : nullptr,
                                     b_zp_tensor ? &b_zp_tensor->Shape() : nullptr));
  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();
  const auto* bias_data = bias_tensor != nullptr ? bias_tensor->Data<float>() : nullptr;

  // process zero point of b
  bool is_b_zp_per_column = false;
  uint8_t b_zp_default = 0;
  const uint8_t* b_zp_ptr = &b_zp_default;
  if (nullptr != b_zp_tensor) {
    ORT_ENFORCE(IsBQuantParamSupported(b_zp_tensor->Shape(), b_tensor ? b_tensor->Shape() : b_shape_),
                "MatmulInteger : b zero point is not valid");

    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zp_tensor);
    b_zp_ptr = static_cast<const uint8_t*>(b_zp_tensor->DataRaw());
  }

  // process scale of b
  bool is_b_scale_per_column = false;
  float multiplier_per_tensor = a_scale;
  const float* b_scale_data = &multiplier_per_tensor;
  std::vector<float> multipliers_per_column;
  if (nullptr != b_scale_tensor) {
    is_b_scale_per_column = !IsScalarOr1ElementVector(b_scale_tensor);
    const float* b_scale_tensor_data = b_scale_tensor->Data<float>();

    if (is_b_scale_per_column) {
      multipliers_per_column.reserve(narrow<size_t>(b_scale_tensor->Shape().Size()));
      std::transform(b_scale_tensor_data,
                     b_scale_tensor_data + b_scale_tensor->Shape().Size(),
                     std::back_inserter(multipliers_per_column),
                     [&a_scale](float b_scale) {
                       return a_scale * b_scale;
                     });
      b_scale_data = multipliers_per_column.data();
    } else {
      multiplier_per_tensor *= *b_scale_tensor_data;
    }
  }

  // batch gemm
  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.AIsSigned = a_is_signed;
  gemm_shape.BIsSigned = b_tensor ? b_tensor->IsDataType<int8_t>() : b_is_signed_;

  const size_t num_gemms = helper.OutputOffsets().size();
  std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> gemm_scale_procs;
  gemm_scale_procs.reserve(num_gemms);
  std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(num_gemms);

  for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
    gemm_scale_procs.emplace_back(y_data + helper.OutputOffsets()[gemm_idx],
                                  gemm_shape.N,
                                  b_scale_data + helper.RightScaleOffsets()[gemm_idx],
                                  bias_data,
                                  MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                  is_b_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
    auto& params = gemm_data_vec[gemm_idx];
    params.OutputProcessor = &(gemm_scale_procs[gemm_idx]);
    params.A = a_data + helper.LeftOffsets()[gemm_idx];
    params.lda = gemm_shape.K;
    params.ZeroPointA = a_zp;
    params.BIsPacked = bool(packed_b_);
    params.B = b_tensor ? static_cast<const uint8_t*>(b_tensor->DataRaw()) + helper.RightOffsets()[gemm_idx] : packed_b_.get();
    params.ldb = gemm_shape.N;
    params.ZeroPointB = b_zp_ptr + helper.RightZeroPointOffsets()[gemm_idx];
    params.PerColumnZeroPoints = is_b_zp_per_column;
    params.C = reinterpret_cast<int32_t*>(y_data + helper.OutputOffsets()[gemm_idx]);
    params.ldc = gemm_shape.N;
  }

  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, ctx->GetOperatorThreadPool());

  return Status::OK();
}

class DynamicQuantizeMatMul final : public MatMulIntegerToFloatBase {
 public:
  DynamicQuantizeMatMul(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override {
    // only pack Matrix B
    if (input_idx == GetBIdx()) {
      const Tensor* b_zp_tensor{nullptr};

      // assume zero is present
      auto b_zp_present{true};

      // zero point tensor could be provided as a direct input to the kernel and not as a constant so this
      // test is not sufficient
      const OrtValue* b_zp;
      if (Info().TryGetConstantInput(IN_B_ZERO_POINT, &b_zp)) {
        b_zp_tensor = &b_zp->Get<Tensor>();
      }

      // MlasDynamicQgemm requires symmetric quantization for B, so no zero point should exist or it should
      // have a zero value
      if (b_zp_tensor != nullptr) {
        b_zp_present = (b_zp_tensor->Shape().Size() != 1) ||
                       (static_cast<const uint8_t*>(b_zp_tensor->DataRaw())[0] != 0);
      }

      // MlasDynamicQgemm requires scale data to be available at packing stage
      const OrtValue* b_scale;
      auto b_scale_available = Info().TryGetConstantInput(IN_B_SCALE, &b_scale);

      can_use_dynamic_quant_mlas_ = (!b_zp_present && b_scale_available);

      // Can we use the MLAS dynamic Q gemm interface supported with float output ?
      if (!can_use_dynamic_quant_mlas_) {
        // default to piece wise mlas interface with separate int matmul, quantize and float conversion
        return MatMulIntegerToFloatBase::PrePack(tensor, input_idx, alloc, is_packed, prepacked_weights);

      } else {
        is_packed = false;

        auto b_scale_tensor = &b_scale->Get<Tensor>();

        // Default to all zeros for bias
        const Tensor* bias_tensor{nullptr};

        // Enable this if fix in onnxruntime/core/optimizer/matmul_integer_to_float.cc has been applied
#if 1
        const OrtValue* bias;
        if (Info().TryGetConstantInput(IN_BIAS, &bias)) {
          bias_tensor = &bias->Get<Tensor>();
          dynamic_quant_mlas_packed_ = true;
        }
#endif

        // Only handle the common case of a 2D weight matrix. Additional matrices
        // could be handled by stacking the packed buffers.
        b_shape_ = tensor.Shape();
        if (b_shape_.NumDimensions() != 2) {
          return Status::OK();
        }

        size_t K = static_cast<size_t>(b_shape_[0]);
        size_t N = static_cast<size_t>(b_shape_[1]);

        const auto* b_data = static_cast<const uint8_t*>(tensor.DataRaw());

        std::optional<Tensor> b_trans_buffer;
        if (IsBTransposed()) {
          std::swap(K, N);
          b_data = quantization::TransPoseInputData(b_data, b_trans_buffer, alloc, N, K);
        }

        const size_t packed_b_size = MlasDynamicQgemmPackBSize(N, K);
        if (packed_b_size == 0) {
          return Status::OK();
        }

        packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size, true);
        // Initialize memory to 0 as there could be some padding associated with pre-packed
        // buffer memory and we do not want it uninitialized and generate different hashes
        // if and when we try to cache this pre-packed buffer for sharing between sessions.
        memset(packed_b_.get(), 0, packed_b_size);

        const auto scales = static_cast(b_scale_tensor->Shape().Size()) == N ? std::vector<float>(&b_scale_tensor->Data<float>()[0],
                                                                                             &b_scale_tensor->Data<float>()[N])
                                                                        :
                                                                        // Broadcast matrix scale to all channels
                                std::vector<float>(N, b_scale_tensor->Data<float>()[0]);

        const auto biases = bias_tensor != nullptr ? std::vector<float>(&bias_tensor->Data<float>()[0],
                                                                        &bias_tensor->Data<float>()[N])
                                                   :
                                                   // Broadcast zero to all channels - no bias data is available
                                std::vector<float>(N, 0.f);

        MlasDynamicQgemmPackB(N, K, reinterpret_cast<const int8_t*>(b_data), scales.data(), biases.data(),
                              packed_b_.get());

        bool share_prepacked_weights = (prepacked_weights != nullptr);
        if (share_prepacked_weights) {
          prepacked_weights->buffers_.push_back(std::move(packed_b_));
          prepacked_weights->buffer_sizes_.push_back(packed_b_size);
        }

        is_packed = true;
      }
    }

    return Status::OK();
  }

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_B_SCALE = 2,
    IN_B_ZERO_POINT = 3,
    IN_BIAS = 4
  };

 protected:
  int GetBIdx() const override { return IN_B; }

 private:
  bool can_use_dynamic_quant_mlas_{false};
  bool dynamic_quant_mlas_packed_{false};
};

class MatMulIntegerToFloat final : public MatMulIntegerToFloatBase {
 public:
  MatMulIntegerToFloat(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_SCALE = 2,
    IN_B_SCALE = 3,
    IN_A_ZERO_POINT = 4,
    IN_B_ZERO_POINT = 5,
    IN_BIAS = 6
  };

 protected:
  int GetBIdx() const override { return IN_B; }

 private:
  // a scale and b scale may be switched in fusion stage because of lack of shape information.
  // Fix them up before computation.
  static void FixupScaleTensor(const Tensor*& a_scale_tensor, const Tensor*& b_scale_tensor);
};

Status DynamicQuantizeMatMul::Compute(OpKernelContext* ctx) const {
  // Can this operation be offloaded to a MLAS specific dynamic quantization matmul ?
  if (!can_use_dynamic_quant_mlas_) {
    const Tensor* a = ctx->Input<Tensor>(IN_A);
    const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

    const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
    const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);

    // calculate quantization parameter of a
    const float* a_data = a->Data<float>();
    int64_t num_of_elements = a->Shape().Size();

    float a_scale;
    uint8_t a_zero_point;
    GetQuantizationParameter(a_data, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
    uint8_t* a_data_quant = static_cast<uint8_t*>(allocator->Alloc(SafeInt<size_t>(num_of_elements) * sizeof(uint8_t)));
    BufferUniquePtr a_buffer_quant_holder(a_data_quant, BufferDeleter(std::move(allocator)));

    ParQuantizeLinearStd(a_data, a_data_quant, narrow<size_t>(num_of_elements), a_scale, a_zero_point, ctx->GetOperatorThreadPool());

    bool is_b_scale_supported = IsBQuantParamSupported(b_scale_tensor->Shape(), b ? b->Shape() : b_shape_);
    ORT_RETURN_IF_ERROR(ComputeCommon(
        ctx,
        a_data_quant,
        a->Shape(),
        a_scale,
        a_zero_point,
        false,
        b,
        is_b_scale_supported ? b_scale_tensor : nullptr,
        b_zp_tensor,
        ctx->Input<Tensor>(IN_BIAS)));

    if (!is_b_scale_supported) {
      ScaleOutput(*b_scale_tensor, *ctx->Output<Tensor>(0));
    }
  } else {
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(ctx->Input<Tensor>(IN_A)->Shape(),
                                       b_shape_,  // ctx->Input<Tensor>(IN_B)->Shape(), this is not available now constant data is
                                       // deleted during session init post prepacking
                                       nullptr,
                                       nullptr));

    Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());

    // Bail out early if the output is going to be empty
    if (y->Shape().Size() == 0)
      return Status::OK();

    auto a_data = static_cast<const uint8_t*>(ctx->Input<Tensor>(IN_A)->DataRaw());
    auto* y_data = y->MutableData<float>();

    // batch gemm
    MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS gemm_shape;
    gemm_shape.M = static_cast<size_t>(helper.M());
    gemm_shape.N = static_cast<size_t>(helper.N());
    gemm_shape.K = static_cast<size_t>(helper.K());

    const size_t num_gemms = helper.OutputOffsets().size();
    std::vector<MLAS_GEMM_DYN_QUANT_DATA_PARAMS> gemm_data_vec(num_gemms);

    for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
      auto& params = gemm_data_vec[gemm_idx];
      params.A = reinterpret_cast<const float*>(a_data + helper.LeftOffsets()[gemm_idx]);
      params.lda = gemm_shape.K;
      params.PackedB = packed_b_.get();
      params.ldb = gemm_shape.N;
      params.C = y_data + helper.OutputOffsets()[gemm_idx];
      params.ldc = gemm_shape.N;
    }

    MlasDynamicQGemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, ctx->GetOperatorThreadPool());
    // This evaluates to true if bias data was not provided as constant data for prepacking stage
    if (!dynamic_quant_mlas_packed_) {
      if (ctx->Input<Tensor>(IN_BIAS) != nullptr) {
        const auto biases = std::vector<float>(&ctx->Input<Tensor>(IN_BIAS)->Data<float>()[0],
                                               &ctx->Input<Tensor>(IN_BIAS)->Data<float>()[gemm_shape.N]);

        // deferred adding of bias
        for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
          float* MxN = y_data + helper.OutputOffsets()[gemm_idx];
          for (auto l = gemm_shape.M; l > 0; --l) {
            MlasEltwiseAdd<float>(MxN, biases.data(), MxN, gemm_shape.N);
            MxN += gemm_shape.N;
          }
        }
      }
    }
  }
  return Status::OK();
}

void MatMulIntegerToFloat::FixupScaleTensor(const Tensor*& a_scale_tensor, const Tensor*& b_scale_tensor) {
  const TensorShape a_scale_shape = a_scale_tensor->Shape();
  const TensorShape b_scale_shape = b_scale_tensor->Shape();
  if (!IsScalarOr1ElementVector(a_scale_tensor)) {
    size_t a_scale_rank = a_scale_shape.NumDimensions();
    if (a_scale_rank == 1 || a_scale_shape[a_scale_rank - 1] != 1) {
      std::swap(a_scale_tensor, b_scale_tensor);
    }
  } else if (!IsScalarOr1ElementVector(b_scale_tensor)) {
    size_t b_scale_rank = b_scale_shape.NumDimensions();
    if (b_scale_rank > 1 && b_scale_shape[b_scale_rank - 2] != 1) {
      std::swap(a_scale_tensor, b_scale_tensor);
    }
  }
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* a_scale_tensor = ctx->Input<Tensor>(IN_A_SCALE);
  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  FixupScaleTensor(a_scale_tensor, b_scale_tensor);
  bool is_a_scale_scalar = IsScalarOr1ElementVector(a_scale_tensor);
  bool is_b_scale_supported = IsBQuantParamSupported(b_scale_tensor->Shape(), nullptr != b ? b->Shape() : b_shape_);

  // validate zero point of a
  uint8_t a_zero_point = 0;
  const Tensor* a_zero_point_tensor = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (a_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point_tensor),
                "MatMulIntegerToFloat : input a zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    a_zero_point = *(static_cast<const uint8_t*>(a_zero_point_tensor->DataRaw()));
  }

  const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  ORT_RETURN_IF_ERROR(ComputeCommon(
      ctx,
      static_cast<const uint8_t*>(a->DataRaw()),
      a->Shape(),
      is_a_scale_scalar ? *a_scale_tensor->Data<float>() : 1.f,
      a_zero_point,
      a->IsDataType<int8_t>(),
      b,
      is_b_scale_supported ? b_scale_tensor : nullptr,
      b_zp_tensor,
      ctx->Input<Tensor>(IN_BIAS)));

  if (!is_a_scale_scalar) {
    ScaleOutput(*a_scale_tensor, *ctx->Output<Tensor>(0));
  }
  if (!is_b_scale_supported) {
    ScaleOutput(*b_scale_tensor, *ctx->Output<Tensor>(0));
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeMatMul,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    DynamicQuantizeMatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

}  // namespace contrib
}  // namespace onnxruntime
