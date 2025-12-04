// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <optional>
#include <vector>

#include "core/common/cpuid_info.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"
#include "core/common/safeint.h"
#include "core/quantization/quantization.h"

namespace onnxruntime {

class MatMulIntegerBase : public OpKernel {
 public:
  MatMulIntegerBase(const OpKernelInfo& info) : OpKernel(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override {
    is_packed = false;

    // only pack Matrix B++
    if (input_idx == GetBIdx()) {
#if defined(USE_KLEIDIAI) && !defined(_MSC_VER)
      if (TryKleidiaiDynamicPrePack(tensor, input_idx, alloc, is_packed, prepacked_weights)) {
        return Status::OK();
      }
#endif
      // Only handle the common case of a 2D weight matrix. Additional matrices
      // could be handled by stacking the packed buffers.
      b_shape_ = tensor.Shape();
      if (b_shape_.NumDimensions() != 2) {
        return Status::OK();
      }

      auto a_elem_type = Node().InputDefs()[GetAIdx()]->TypeAsProto()->tensor_type().elem_type();
      bool a_is_signed = ONNX_NAMESPACE::TensorProto_DataType_INT8 == a_elem_type;

      b_is_signed_ = tensor.IsDataType<int8_t>();

      size_t K = static_cast<size_t>(b_shape_[0]);
      size_t N = static_cast<size_t>(b_shape_[1]);

      const auto* b_data = static_cast<const uint8_t*>(tensor.DataRaw());

      std::optional<Tensor> b_trans_buffer;
      if (IsBTransposed()) {
        std::swap(K, N);
        b_data = quantization::TransPoseInputData(b_data, b_trans_buffer, alloc, N, K);
      }
      const size_t packed_b_size = MlasGemmPackBSize(N, K, a_is_signed, b_is_signed_);
      if (packed_b_size == 0) {
        return Status::OK();
      }

      packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size, true);
      // Initialize memory to 0 as there could be some padding associated with pre-packed
      // buffer memory and we don not want it uninitialized and generate different hashes
      // if and when we try to cache this pre-packed buffer for sharing between sessions.
      memset(packed_b_.get(), 0, packed_b_size);
      MlasGemmPackB(N, K, b_data, N, a_is_signed, b_is_signed_, packed_b_.get());

      bool share_prepacked_weights = (prepacked_weights != nullptr);
      if (share_prepacked_weights) {
        prepacked_weights->buffers_.push_back(std::move(packed_b_));
        prepacked_weights->buffer_sizes_.push_back(packed_b_size);
      }

      is_packed = true;
    }
    return Status::OK();
  }

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override {
    used_shared_buffers = false;

    if (input_idx == GetBIdx()) {
      used_shared_buffers = true;
      packed_b_ = std::move(prepacked_buffers[0]);
    }

    return Status::OK();
  }

 protected:
  /**
   * @return input index of Matrix B, the weight tensor
   */
  virtual int GetAIdx() const { return 0; }
  virtual int GetBIdx() const = 0;

  virtual bool IsBTransposed() const {
    return false;
  }

  virtual int GetBScaleIdx() const {
    return -1;
  }

  virtual int GetBZeroPointIdx() const {
    return -1;
  }

  virtual int GetBiasIdx() const {
    return -1;
  }

  virtual bool SupportsKleidiaiDynamicQuant() const {
    return false;
  }

  bool can_use_dynamic_quant_mlas_{false};

#if defined(USE_KLEIDIAI) && !defined(_MSC_VER)
  struct KleidiaiDynamicPackContext {
    const Tensor* scale{nullptr};
    const Tensor* bias{nullptr};
    const uint8_t* b_data{nullptr};
    size_t K{0};
    size_t N{0};
    std::optional<Tensor> transposed_buffer;
  };

  bool TryKleidiaiDynamicPrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                 bool& is_packed,
                                 PrePackedWeights* prepacked_weights) {
    if (!SupportsKleidiaiDynamicQuant() || input_idx != GetBIdx()) {
      return false;
    }

    KleidiaiDynamicPackContext ctx;
    if (!PrepareKleidiaiDynamicPack(tensor, alloc, ctx)) {
      return false;
    }

    return ExecuteKleidiaiDynamicPack(ctx, alloc, is_packed, prepacked_weights);
  }

  bool PrepareKleidiaiDynamicPack(const Tensor& tensor,
                                  AllocatorPtr alloc,
                                  KleidiaiDynamicPackContext& ctx) {
    can_use_dynamic_quant_mlas_ = false;
    dynamic_quant_mlas_bias_data_was_packed_ = false;

    ctx.scale = GetConstantInputTensor(GetBScaleIdx());
    if (ctx.scale == nullptr) {
      return false;
    }

    if (!IsZeroPointSymmetric()) {
      return false;
    }

    if (!AreScalesValid(*ctx.scale)) {
      return false;
    }

    if (!IsBShapeSupportedForDynamicQuant(tensor.Shape())) {
      return false;
    }

    ctx.bias = GetConstantInputTensor(GetBiasIdx());
    if (ctx.bias != nullptr) {
      dynamic_quant_mlas_bias_data_was_packed_ = true;
    }

    ctx.K = static_cast<size_t>(b_shape_[0]);
    ctx.N = static_cast<size_t>(b_shape_[1]);
    ctx.b_data = static_cast<const uint8_t*>(tensor.DataRaw());

    if (IsBTransposed()) {
      std::swap(ctx.K, ctx.N);
      ctx.b_data = quantization::TransPoseInputData(ctx.b_data, ctx.transposed_buffer, alloc, ctx.N, ctx.K);
    }

    can_use_dynamic_quant_mlas_ = true;
    return true;
  }

  bool ExecuteKleidiaiDynamicPack(const KleidiaiDynamicPackContext& ctx,
                                  AllocatorPtr alloc,
                                  bool& is_packed,
                                  PrePackedWeights* prepacked_weights) {
    if (!can_use_dynamic_quant_mlas_) {
      return false;
    }

    is_packed = false;

    const size_t packed_b_size = MlasDynamicQgemmPackBSize(ctx.N, ctx.K);
    if (packed_b_size == 0) {
      can_use_dynamic_quant_mlas_ = false;
      return true;
    }

    packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size, true);
    memset(packed_b_.get(), 0, packed_b_size);

    const auto scales = static_cast<size_t>(ctx.scale->Shape().Size()) == ctx.N
                            ? std::vector<float>(&ctx.scale->Data<float>()[0],
                                                 &ctx.scale->Data<float>()[ctx.N])
                            : std::vector<float>(ctx.N, ctx.scale->Data<float>()[0]);

    const auto biases = ctx.bias != nullptr
                            ? std::vector<float>(&ctx.bias->Data<float>()[0],
                                                 &ctx.bias->Data<float>()[ctx.N])
                            : std::vector<float>(ctx.N, 0.f);

    MlasDynamicQgemmPackB(ctx.N, ctx.K, reinterpret_cast<const int8_t*>(ctx.b_data),
                          scales.data(), biases.data(), packed_b_.get());

    if (prepacked_weights != nullptr) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size);
    }

    is_packed = true;
    return true;
  }

  bool IsZeroPointSymmetric() {
    const Tensor* b_zp_constant_tensor = GetConstantInputTensor(GetBZeroPointIdx());
    if (b_zp_constant_tensor != nullptr) {
      assert(b_zp_constant_tensor->IsDataType<uint8_t>() || b_zp_constant_tensor->IsDataType<int8_t>());
      const auto* zp_bytes = static_cast<const std::byte*>(b_zp_constant_tensor->DataRaw());
      const size_t zp_size_in_bytes = b_zp_constant_tensor->SizeInBytes();
      can_use_dynamic_quant_mlas_ = std::none_of(zp_bytes, zp_bytes + zp_size_in_bytes,
                                                 [](std::byte v) { return v != std::byte{0}; });
      return can_use_dynamic_quant_mlas_;
    }

    const auto input_defs = Info().node().InputDefs();
    const int b_zp_idx = GetBZeroPointIdx();
    const bool b_zp_input_exists = b_zp_idx >= 0 &&
                                   static_cast<size_t>(b_zp_idx) < input_defs.size() &&
                                   input_defs[b_zp_idx]->Exists();
    can_use_dynamic_quant_mlas_ = !b_zp_input_exists;
    return can_use_dynamic_quant_mlas_;
  }

  bool AreScalesValid(const Tensor& b_scale_tensor) {
    const auto bs = b_scale_tensor.DataAsSpan<float>();
    const bool has_invalid =
        std::any_of(bs.begin(), bs.end(),
                    [](float s) { return !std::isfinite(s) || s <= 0.0f; });

    if (has_invalid) {
      can_use_dynamic_quant_mlas_ = false;
    }
    return can_use_dynamic_quant_mlas_;
  }

  bool IsBShapeSupportedForDynamicQuant(const TensorShape& tensor_shape) {
    b_shape_ = tensor_shape;
    if (b_shape_.NumDimensions() < 2) {
      return false;
    }

    for (size_t i = 0; i < (b_shape_.NumDimensions() - 2); ++i) {
      if (b_shape_[i] != 1) {
        return false;
      }
    }
    return true;
  }

  const Tensor* GetConstantInputTensor(int input_idx) const {
    if (input_idx < 0) {
      return nullptr;
    }
    const OrtValue* ort_value = nullptr;
    if (!Info().TryGetConstantInput(input_idx, &ort_value)) {
      return nullptr;
    }

    return &ort_value->Get<Tensor>();
  }

  bool dynamic_quant_mlas_bias_data_was_packed_{false};
#endif

  // Check if quantization parameter of B is supported.
  // It should be in one of the formats below:
  // 1. Scalar
  // 2. 1D tensor with size equal to 1 or last dimension of B_shape if B_shape is a 2D tensor
  // 3. Equal to B_shape except that the second to last is 1
  bool IsBQuantParamSupported(const TensorShape& B_quant_param_shape, const TensorShape& B_shape) const {
    int64_t B_quant_param_rank = B_quant_param_shape.NumDimensions();
    int64_t B_shape_rank = B_shape.NumDimensions();
    if (B_quant_param_rank == 0 ||                                       // scalar
        (B_quant_param_rank == 1 && B_quant_param_shape.Size() == 1)) {  // 1D tensor with size 1
      return true;
    }

    if (B_quant_param_rank == 1 &&
        B_shape_rank == 2 &&
        B_quant_param_shape[0] == B_shape[1]) {
      return true;
    }

    if (B_quant_param_rank != B_shape_rank ||
        B_quant_param_rank <= 1 ||
        B_quant_param_shape[SafeInt<size_t>(B_quant_param_rank) - 2] != 1) {
      return false;
    }

    for (int64_t rank = 0; rank < B_quant_param_rank; rank++) {
      if (rank != B_quant_param_rank - 2 &&
          B_quant_param_shape[onnxruntime::narrow<size_t>(rank)] != B_shape[onnxruntime::narrow<size_t>(rank)]) {
        return false;
      }
    }

    return true;
  }

  bool b_is_signed_{true};
  TensorShape b_shape_;
  IAllocatorUniquePtr<void> packed_b_;
};

}  // namespace onnxruntime
