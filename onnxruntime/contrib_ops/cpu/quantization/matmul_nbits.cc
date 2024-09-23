// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/matmul_nbits_impl.h"

#include <cstdint>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"

#ifdef ORT_NEURAL_SPEED
#include "contrib_ops/cpu/quantization/neural_speed_gemm.h"
#endif

namespace onnxruntime {
namespace contrib {

namespace {

// MatMulNBits op input indices.
// These should match the inputs names specified in the op schema.
namespace InputIndex {
constexpr size_t A = 0,
                 B = 1,
                 scales = 2,
                 zero_points = 3,
                 g_idx = 4,
                 bias = 5;
};

int64_t GetAccuracyLevel(size_t nbits, size_t block_size, int64_t accuracy_level_attr) {
  const auto accuracy_level = std::clamp(accuracy_level_attr,
                                         static_cast<int64_t>(CompMostAccurate),
                                         static_cast<int64_t>(CompLeastAccurate));

#if defined(ORT_NEURAL_SPEED)

  ORT_UNUSED_PARAMETER(nbits);
  ORT_UNUSED_PARAMETER(block_size);

  // Neural Speed APIs already expect a minimum accuracy level so just use the given value.
  return accuracy_level;

#else  // defined(ORT_NEURAL_SPEED)

  // Find a supported accuracy level that is not less accurate than the one given.
  // CompMostAccurate is always supported with the fallback implementation.
  // Note: A higher numeric accuracy level value means lower accuracy, so the comparison order is reversed.
  int64_t effective_accuracy_level = accuracy_level;
  for (; effective_accuracy_level > CompMostAccurate; --effective_accuracy_level) {
    const auto compute_type = static_cast<MLAS_SQNBIT_GEMM_COMPUTE_TYPE>(effective_accuracy_level);
    if (MlasIsSQNBitGemmAvailable(nbits, block_size, compute_type)) {
      break;
    }
  }

  return effective_accuracy_level;

#endif  // defined(ORT_NEURAL_SPEED)
}
}  // namespace

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

// T1 is the type of the input matrix A, scales and biases.
// Use class level template to facilitate specialization for different types.
template <typename T1>
class MatMulNBits final : public OpKernel {
 public:
  MatMulNBits(const OpKernelInfo& info)
      : OpKernel(info),
        K_{narrow<size_t>(info.GetAttr<int64_t>("K"))},
        N_{narrow<size_t>(info.GetAttr<int64_t>("N"))},
        block_size_{narrow<size_t>(info.GetAttr<int64_t>("block_size"))},
        nbits_{narrow<size_t>(info.GetAttr<int64_t>("bits"))},
        accuracy_level_{GetAccuracyLevel(nbits_, block_size_, info.GetAttr<int64_t>("accuracy_level"))},
        has_g_idx_{info.GetInputCount() > InputIndex::g_idx && info.node().InputDefs()[InputIndex::g_idx]->Exists()},
        has_bias_{info.GetInputCount() > InputIndex::bias && info.node().InputDefs()[InputIndex::bias]->Exists()},
        compute_type_{static_cast<MLAS_SQNBIT_GEMM_COMPUTE_TYPE>(accuracy_level_)} {
    const auto& node = info.node();
    auto input_defs = node.InputDefs();
    const NodeArg* zero_point_arg =
        (info.GetInputCount() > InputIndex::zero_points && input_defs[InputIndex::zero_points]->Exists())
            ? input_defs[3]
            : nullptr;

    if (int32_t type; zero_point_arg && GetType(*zero_point_arg, type)) {
      has_unquantized_zero_point_ = type != ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    }

    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for MatMulNBits op, additional bits support is planned.");
    const Tensor* tensor_zero_point = nullptr;
    has_zp_input_ = info.TryGetConstantInput(InputIndex::zero_points, &tensor_zero_point);
#ifdef ORT_NEURAL_SPEED
    const Tensor* tensor_B = nullptr;
    const Tensor* tensor_scale = nullptr;
    bool B_constant = info.TryGetConstantInput(InputIndex::B, &tensor_B);
    bool scale_constant = info.TryGetConstantInput(InputIndex::scales, &tensor_scale);
    is_asym_ = zero_point_arg != nullptr;
    all_constant_ = B_constant && scale_constant;
    all_constant_ = is_asym_ ? all_constant_ && has_zp_input_ : all_constant_;
#endif
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  const size_t K_;
  const size_t N_;
  const size_t block_size_;
  const size_t nbits_;
  const int64_t accuracy_level_;
  const bool has_g_idx_;
  const bool has_bias_;
  const MLAS_SQNBIT_GEMM_COMPUTE_TYPE compute_type_;
  bool has_unquantized_zero_point_{false};
  const bool column_wise_quant_{true};
  IAllocatorUniquePtr<void> packed_b_{};
  size_t packed_b_size_{0};

  bool has_zp_input_{false};
#if defined(ORT_NEURAL_SPEED)

  bool is_asym_{false};
  bool all_constant_{false};

#endif  // defined(ORT_NEURAL_SPEED)

  // dequantize B first and then compute float gemm
  Status ComputeBUnpacked(const Tensor* a,
                          const Tensor* b,
                          const Tensor* scales,
                          const Tensor* zero_points,
                          const Tensor* reorder_idx,
                          const Tensor* bias,
                          Tensor* y,
                          AllocatorPtr& allocator,
                          concurrency::ThreadPool* thread_pool,
                          const MatMulComputeHelper& helper) const {
    ORT_THROW("ComputeBUnpacked is not supported for T1 type.");
  }

  Status ComputeBPacked(const Tensor* a,
                        const Tensor* scales,
                        const Tensor* zero_points,
                        const Tensor* bias,
                        Tensor* y,
                        AllocatorPtr& allocator,
                        concurrency::ThreadPool* thread_pool,
                        const MatMulComputeHelper& helper) const {
    ORT_THROW("ComputeBPacked is not supported for T1 type.");
  }

  void PackScale(const Tensor& tensor) {
    ORT_THROW("PackScale is not supported for T1 type.");
  }
};

#ifdef MLAS_TARGET_AMD64_IX86
template <>
void MatMulNBits<MLFloat16>::PackScale(const Tensor& tensor) {
  auto sptr = tensor.Data<MLFloat16>();
  std::vector<float> scales_v(static_cast<unsigned int>(tensor.Shape().Size()));
  MlasConvertHalfToFloatBuffer(sptr, &scales_v[0], scales_v.size());
  MlasSQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_, nullptr, packed_b_.get(), &scales_v[0],
                               has_zp_input_, nullptr, nullptr);
}

template <>
void MatMulNBits<float>::PackScale(const Tensor& tensor) {
  auto sptr = tensor.Data<float>();
  MlasSQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_, nullptr, packed_b_.get(), sptr,
                               has_zp_input_, nullptr, nullptr);
}
#endif

template <typename T1>
Status MatMulNBits<T1>::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                                /*out*/ bool& is_packed,
                                /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;
  if (has_g_idx_ || has_unquantized_zero_point_) {
    return Status::OK();
  }
#if defined(ORT_NEURAL_SPEED)

  if (!all_constant_) {
    return Status::OK();
  }

  if (has_bias_) {  // adding bias is not supported
    return Status::OK();
  }

  if (nbits_ != 4) {
    return Status::OK();
  }

  MLAS_THREADPOOL* pool = nullptr;

  auto nbits = static_cast<int>(nbits_);
  if (input_idx == InputIndex::B) {
    packed_b_size_ = NSNBitsGemmPackBSize(N_, K_, block_size_, nbits, is_asym_, compute_type_);
    if (packed_b_size_ == 0) return Status::OK();
    auto qptr = tensor.Data<uint8_t>();
    packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size_, true);
    std::memset(packed_b_.get(), 0, packed_b_size_);
    NSNBitsGemmPackB(packed_b_.get(), qptr, nullptr, nullptr, N_, K_, K_, block_size_, nbits, is_asym_, false,
                     compute_type_, pool);
    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size_);
    }
    is_packed = true;
  }
  if (input_idx == InputIndex::scales && packed_b_ != nullptr) {
    auto sptr = tensor.Data<float>();
    NSNBitsGemmPackB(packed_b_.get(), nullptr, sptr, nullptr, N_, K_, K_, block_size_, nbits, is_asym_, !is_asym_,
                     compute_type_, pool);
    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size_);
    }
    is_packed = true;
  }
  if (input_idx == InputIndex::zero_points && packed_b_ != nullptr) {
    auto zptr = tensor.Data<uint8_t>();
    NSNBitsGemmPackB(packed_b_.get(), nullptr, nullptr, zptr, N_, K_, K_, block_size_, nbits, is_asym_, is_asym_,
                     compute_type_, pool);
    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size_);
    }
    is_packed = true;
  }

#else  // defined(ORT_NEURAL_SPEED)
  ORT_UNUSED_PARAMETER(prepacked_weights);
  if (!MlasIsSQNBitGemmAvailable(nbits_, block_size_, compute_type_)) {
    return Status::OK();
  }
  if (input_idx == InputIndex::B) {
    packed_b_size_ = MlasSQNBitGemmPackQuantBDataSize(N_, K_, nbits_, block_size_, compute_type_);
    if (packed_b_size_ == 0) {
      return Status::OK();
    }
    auto qptr = tensor.DataRaw();
    packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size_, true);
    MlasSQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_, qptr, packed_b_.get(), nullptr, has_zp_input_, nullptr, nullptr);
    is_packed = true;
  } else if (compute_type_ == CompInt8) {
#ifdef MLAS_TARGET_AMD64_IX86
    if (input_idx == InputIndex::scales && packed_b_ != nullptr) {
      PackScale(tensor);
      is_packed = false;
    } else if (input_idx == InputIndex::zero_points && packed_b_ != nullptr) {
      auto zptr = tensor.Data<uint8_t>();
      MlasSQNBitGemmPackQuantBData(N_, K_, nbits_, block_size_, compute_type_, nullptr, packed_b_.get(), nullptr, has_zp_input_, zptr, nullptr);
      is_packed = false;
    }
#endif
  }
#endif  // defined(ORT_NEURAL_SPEED)

  return Status::OK();
}

template <typename T1>
Status MatMulNBits<T1>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                                  /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

#if defined(ORT_NEURAL_SPEED)

  // Pack three tensors into one buffer
  if (input_idx == 1) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }
  if (input_idx == 2) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }
  if (input_idx == 3) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }

#else  // defined(ORT_NEURAL_SPEED)

  if (input_idx == 1) {
    used_shared_buffers = true;
    packed_b_ = std::move(prepacked_buffers[0]);
  }

#endif  // defined(ORT_NEURAL_SPEED)

  return Status::OK();
}

template <>
Status MatMulNBits<float>::ComputeBPacked(const Tensor* a,
                                          const Tensor* scales,
                                          const Tensor* zero_points,
                                          const Tensor* bias,
                                          Tensor* y,
                                          AllocatorPtr& allocator,
                                          concurrency::ThreadPool* thread_pool,
                                          const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<float>();
  const auto* scales_data = scales->Data<float>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* bias_data = bias == nullptr ? nullptr : bias->Data<float>();
  auto* y_data = y->MutableData<float>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);

  IAllocatorUniquePtr<std::byte> workspace{};
  const size_t workspace_size = MlasSQNBitGemmBatchWorkspaceSize(
      M, N, K, batch_count, nbits_, block_size_, compute_type_);
  if (workspace_size > 0) {
    workspace = IAllocator::MakeUniquePtr<std::byte>(allocator, workspace_size);
  }

  InlinedVector<MLAS_SQNBIT_GEMM_DATA_PARAMS> data(batch_count);
  for (size_t i = 0; i < batch_count; ++i) {
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
#ifdef MLAS_TARGET_AMD64_IX86
    if (compute_type_ == CompInt8) {
      data[i].QuantBDataWorkspace = packed_b_.get();
    }
#endif
    data[i].PackedQuantBData = static_cast<std::byte*>(packed_b_.get());
    data[i].QuantBScale = scales_data;
    data[i].QuantBZeroPoint = zero_points_data;
    data[i].Bias = bias_data;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
  }
  MlasSQNBitGemmBatch(M, N, K, batch_count, nbits_, block_size_, compute_type_, data.data(), workspace.get(),
                      thread_pool);
  return Status::OK();
}

template <>
Status MatMulNBits<MLFloat16>::ComputeBPacked(const Tensor* a,
                                              const Tensor* scales,
                                              const Tensor* zero_points,
                                              const Tensor* bias,
                                              Tensor* y,
                                              AllocatorPtr& allocator,
                                              concurrency::ThreadPool* thread_pool,
                                              const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<MLFloat16>();
  const auto* scales_data = scales->Data<MLFloat16>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* bias_data = bias == nullptr ? nullptr : bias->Data<MLFloat16>();
  auto* y_data = y->MutableData<MLFloat16>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);

  IAllocatorUniquePtr<std::byte> workspace{};
  const size_t workspace_size = MlasSQNBitGemmBatchWorkspaceSize(
      M, N, K, batch_count, nbits_, block_size_, compute_type_);
  if (workspace_size > 0) {
    workspace = IAllocator::MakeUniquePtr<std::byte>(allocator, workspace_size);
  }

  auto tmp_a_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, (size_t)(a->Shape().Size()));
  MlasConvertHalfToFloatBuffer(a_data, tmp_a_data_ptr.get(), static_cast<size_t>(a->Shape().Size()));

  auto tmp_scales_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, (size_t)(scales->Shape().Size()));
  MlasConvertHalfToFloatBuffer(scales_data, tmp_scales_data_ptr.get(), static_cast<size_t>(scales->Shape().Size()));

  std::vector<float> bias_data_v;
  if (bias_data != nullptr) {
    bias_data_v.resize(static_cast<size_t>(bias->Shape().Size()));
    MlasConvertHalfToFloatBuffer(bias_data, &bias_data_v[0], bias_data_v.size());
  }

  std::vector<float> C_v(static_cast<size_t>(y->Shape().Size()));

  InlinedVector<MLAS_SQNBIT_GEMM_DATA_PARAMS> data(batch_count);
  for (size_t i = 0; i < batch_count; ++i) {
    data[i].A = tmp_a_data_ptr.get() + helper.LeftOffsets()[i];
    data[i].lda = lda;
#ifdef MLAS_TARGET_AMD64_IX86
    if (compute_type_ == CompInt8) {
      data[i].QuantBDataWorkspace = packed_b_.get();
    }
#endif
    data[i].PackedQuantBData = static_cast<std::byte*>(packed_b_.get());
    data[i].QuantBScale = tmp_scales_data_ptr.get();
    data[i].QuantBZeroPoint = zero_points_data;
    data[i].Bias = bias_data != nullptr ? &bias_data_v[0] : nullptr;
    data[i].C = &C_v[0] + helper.OutputOffsets()[i];
    data[i].ldc = N;
  }
  MlasSQNBitGemmBatch(M, N, K, batch_count, nbits_, block_size_, compute_type_, data.data(), workspace.get(),
                      thread_pool);
  MlasConvertFloatToHalfBuffer(&C_v[0], y_data, C_v.size());
  return Status::OK();
}

template <>
Status MatMulNBits<float>::ComputeBUnpacked(const Tensor* a,
                                            const Tensor* b,
                                            const Tensor* scales,
                                            const Tensor* zero_points,
                                            const Tensor* reorder_idx,
                                            const Tensor* bias,
                                            Tensor* y,
                                            AllocatorPtr& allocator,
                                            concurrency::ThreadPool* thread_pool,
                                            const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<float>();
  const uint8_t* b_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<float>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();
  auto* y_data = y->MutableData<float>();

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);
  const size_t ldb = helper.Ldb(true);

  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);

  if ((reorder_idx_data == nullptr) && (!zero_points || !zero_points->IsDataType<float>())) {
    // dequantize b, only 4b quantization is supported for now
    MlasDequantizeBlockwise<float, 4>(
        tmp_b_data_ptr.get(),                           // dequantized output
        b_data,                                         // quantized input
        scales_data,                                    // quantization scales
        static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
        static_cast<int32_t>(block_size_),              // quantization block size
        column_wise_quant_,                             // columnwise quantization or row-wise
        static_cast<int32_t>(K_),                       // number of rows in quantized input
        static_cast<int32_t>(N_),                       // number of columns in quantized input
        thread_pool);
  } else {
    ORT_ENFORCE(column_wise_quant_, "Row-wise quantization is not supported for now");
    // !!!!!!!!!!!!!! naive implementation, need to be optimized !!!!!!!!!!!!!!
    if (zero_points && zero_points->IsDataType<float>()) {
      DequantizeBlockwise<float, float>(
          tmp_b_data_ptr.get(),                         // dequantized output
          b_data,                                       // quantized input
          scales_data,                                  // quantization scales
          static_cast<const float*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    } else {
      DequantizeBlockwise<float, uint8_t>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_data,                                    // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    }
  }
#if 0  // for debug
  auto tm_b_data_ptr_trans = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  MlasTranspose(tmp_b_data_ptr.get(), tm_b_data_ptr_trans.get(), N_, K_);
#endif

  std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch_count);
  for (size_t i = 0; i < batch_count; i++) {
    data[i].BIsPacked = false;
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
    data[i].B = tmp_b_data_ptr.get() + helper.RightOffsets()[i];
    data[i].ldb = ldb;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = 1.f;
    data[i].beta = 0.0f;
  }

  // if there is a bias input, copy bias values into C and set beta to 1.0f
  if (bias) {
    gsl::span<const float> bias_span = bias->DataAsSpan<float>();
    for (size_t i = 0; i < batch_count; ++i) {
      float* C_row = data[i].C;
      const size_t ldc = data[i].ldc;
      for (size_t m = 0; m < M; ++m) {
        memcpy(C_row, bias_span.data(), bias_span.size_bytes());
        C_row += ldc;
      }

      data[i].beta = 1.0f;
    }
  }

  MlasGemmBatch(CblasNoTrans, CblasTrans,
                M, N, K, data.data(), batch_count, thread_pool);

  return Status::OK();
}

template <>
Status MatMulNBits<MLFloat16>::ComputeBUnpacked(const Tensor* a,
                                                const Tensor* b,
                                                const Tensor* scales,
                                                const Tensor* zero_points,
                                                const Tensor* reorder_idx,
                                                const Tensor* bias,
                                                Tensor* y,
                                                AllocatorPtr& allocator,
                                                concurrency::ThreadPool* thread_pool,
                                                const MatMulComputeHelper& helper) const {
  const auto* a_data = a->Data<MLFloat16>();
  const uint8_t* b_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<MLFloat16>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();
  auto* y_data = y->MutableData<MLFloat16>();

  const float* scales_data_;
  std::vector<float> scales_data_v;
  scales_data_v.resize(static_cast<size_t>(scales->Shape().Size()));
  MlasConvertHalfToFloatBuffer(scales_data, &scales_data_v[0], scales_data_v.size());
  scales_data_ = &scales_data_v[0];

  const size_t batch_count = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(false);
  const size_t ldb = helper.Ldb(true);

  auto tmp_b_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);

  if ((reorder_idx_data == nullptr) && (!zero_points || !zero_points->IsDataType<MLFloat16>())) {
    // dequantize b, only 4b quantization is supported for now
    MlasDequantizeBlockwise<float, 4>(
        tmp_b_data_ptr.get(),                           // dequantized output
        b_data,                                         // quantized input
        scales_data_,                                   // quantization scales
        static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
        static_cast<int32_t>(block_size_),              // quantization block size
        column_wise_quant_,                             // columnwise quantization or row-wise
        static_cast<int32_t>(K_),                       // number of rows in quantized input
        static_cast<int32_t>(N_),                       // number of columns in quantized input
        thread_pool);
  } else {
    ORT_ENFORCE(column_wise_quant_, "Row-wise quantization is not supported for now");
    // !!!!!!!!!!!!!! naive implementation, need to be optimized !!!!!!!!!!!!!!
    if (zero_points && zero_points->IsDataType<MLFloat16>()) {
      DequantizeBlockwise<float, MLFloat16>(
          tmp_b_data_ptr.get(),                             // dequantized output
          b_data,                                           // quantized input
          scales_data_,                                     // quantization scales
          static_cast<const MLFloat16*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    } else {
      DequantizeBlockwise<float, uint8_t>(
          tmp_b_data_ptr.get(),                           // dequantized output
          b_data,                                         // quantized input
          scales_data_,                                   // quantization scales
          static_cast<const uint8_t*>(zero_points_data),  // quantization zero points
          reorder_idx_data,
          static_cast<int32_t>(block_size_),  // quantization block size
          column_wise_quant_,                 // columnwise quantization or row-wise
          static_cast<int32_t>(K_),           // number of rows in quantized input
          static_cast<int32_t>(N_),           // number of columns in quantized input
          thread_pool);
    }
  }
#if 0  // for debug
  auto tm_b_data_ptr_trans = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(K_) * N_);
  MlasTranspose(tmp_b_data_ptr.get(), tm_b_data_ptr_trans.get(), N_, K_);
#endif

  std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch_count);
  auto tmp_a_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, (size_t)(a->Shape().Size()));
  MlasConvertHalfToFloatBuffer(a_data, tmp_a_data_ptr.get(), static_cast<size_t>(a->Shape().Size()));
  auto tmp_c_ptr = IAllocator::MakeUniquePtr<float>(allocator, (size_t)(y->Shape().Size()));
  for (size_t i = 0; i < batch_count; i++) {
    data[i].BIsPacked = false;
    data[i].A = tmp_a_data_ptr.get() + helper.LeftOffsets()[i];
    data[i].lda = lda;
    data[i].B = tmp_b_data_ptr.get() + helper.RightOffsets()[i];
    data[i].ldb = ldb;
    data[i].C = tmp_c_ptr.get() + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].alpha = 1.f;
    data[i].beta = 0.0f;
  }

  // if there is a bias input, copy bias values into C and set beta to 1.0f
  if (bias) {
    auto tmp_bias_data_ptr = IAllocator::MakeUniquePtr<float>(allocator, (size_t)(bias->Shape().Size()));
    MlasConvertHalfToFloatBuffer(bias->Data<MLFloat16>(),
                                 tmp_bias_data_ptr.get(),
                                 static_cast<size_t>(bias->Shape().Size()));
    for (size_t i = 0; i < batch_count; ++i) {
      float* C_row = data[i].C;
      const size_t ldc = data[i].ldc;
      for (size_t m = 0; m < M; ++m) {
        std::copy(tmp_bias_data_ptr.get(), tmp_bias_data_ptr.get() + bias->Shape().Size(), C_row);
        C_row += ldc;
      }
      data[i].beta = 1.0f;
    }
  }

  MlasGemmBatch(CblasNoTrans, CblasTrans,
                M, N, K, data.data(), batch_count, thread_pool);
  MlasConvertFloatToHalfBuffer(tmp_c_ptr.get(), y_data, static_cast<size_t>(y->Shape().Size()));
  return Status::OK();
}

template <typename T1>
Status MatMulNBits<T1>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  const Tensor* a = ctx->Input<Tensor>(InputIndex::A);
  const Tensor* scales = ctx->Input<Tensor>(InputIndex::scales);
  const Tensor* zero_points = ctx->Input<Tensor>(InputIndex::zero_points);
  const Tensor* reorder_idx = ctx->Input<Tensor>(InputIndex::g_idx);
  const Tensor* bias = ctx->Input<Tensor>(InputIndex::bias);

  TensorShape b_shape({static_cast<int64_t>(N_), static_cast<int64_t>(K_)});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));

  // clang-format off
  const bool has_single_b_matrix = std::all_of(
      helper.RightOffsets().begin(),
      helper.RightOffsets().end(),
      [](size_t offset) { return offset == 0; });
  // clang-format on

  if (has_single_b_matrix &&
      packed_b_) {  // Assume that MlasSQNBitGemmBatch() always requires packed B.
                    // If this changes, i.e., if MlasIsSQNBitGemmAvailable() can return true while
                    // MlasSQNBitGemmPackQuantBDataSize() returns 0, we can consider calling MlasSQNBitGemmBatch()
                    // with B directly too.
#if defined(ORT_NEURAL_SPEED)
    const auto* a_data = a->Data<T1>();
    auto* y_data = y->MutableData<T1>();
    const size_t batch_count = helper.OutputOffsets().size();
    const size_t M = static_cast<size_t>(helper.M());
    const size_t N = static_cast<size_t>(helper.N());
    const size_t K = static_cast<size_t>(helper.K());
    const size_t lda = helper.Lda(false);
    InlinedVector<NS_SQNBITS_GEMM_DATA_PACKED_PARAMS> gemm_params(batch_count);
    for (size_t i = 0; i < batch_count; i++) {
      gemm_params[i].A = a_data + helper.LeftOffsets()[i];
      gemm_params[i].lda = lda;
      gemm_params[i].B = packed_b_.get();
      gemm_params[i].C = y_data + helper.OutputOffsets()[i];
      gemm_params[i].ldc = N;
    }
    auto ws_size = NSSQNBitsGemmBatchWorkspaceSize(M, N, K, batch_count, gemm_params.data());
    // workspace for activation process(dynamic quantization and others)
    auto ws_ptr = IAllocator::MakeUniquePtr<int8_t>(allocator, ws_size);
    NSSQNBitsGemmBatchPackedB(M, N, K, batch_count, gemm_params.data(), ws_ptr.get(), thread_pool);
    return Status::OK();
#else   // defined(ORT_NEURAL_SPEED)
    if (MlasIsSQNBitGemmAvailable(nbits_, block_size_, compute_type_)) {
      return ComputeBPacked(a, scales, zero_points, bias, y, allocator, thread_pool, helper);
    }
#endif  // !defined(ORT_NEURAL_SPEED)
  }

  // If B is prepacked, B would have been removed from the context
  const Tensor* b = ctx->Input<Tensor>(InputIndex::B);
  return ComputeBUnpacked(a, b, scales, zero_points, reorder_idx, bias, y, allocator, thread_pool, helper);
}

#define REGISTER_MatMulNBits(T1)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      MatMulNBits,                                                          \
      kMSDomain,                                                            \
      1,                                                                    \
      T1,                                                                   \
      kCpuExecutionProvider,                                                \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())     \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<uint8_t>(),    \
                                 DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),    \
      MatMulNBits<T1>);

REGISTER_MatMulNBits(float);
REGISTER_MatMulNBits(MLFloat16);

}  // namespace contrib
}  // namespace onnxruntime
