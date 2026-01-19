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

    // only pack Matrix B
    if (input_idx == GetBIdx()) {
#if defined(USE_KLEIDIAI)
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

#if defined(USE_KLEIDIAI)
  struct KleidiaiDynamicPackContext {
    const Tensor* scale{nullptr};
    const Tensor* bias{nullptr};
    const uint8_t* b_data{nullptr};
    size_t K{0};
    size_t N{0};
    std::optional<Tensor> transposed_buffer;
  };
  /*
    Helper method to pre-pack Matrix B using Arm® KleidiAI™ packing if eligible.

    Returns false if KleidiAI dynamic quantization is not supported or the index of the input tensor is not input B's index.
    If these checks pass, prepares a dynamic quantization pack context and calls PrepareKleidiaiDynamicPack for further policies.
    If those policies also satisfy, it calls the helper to execute the pre-packing in KleidiAI context.
    Returns true if pre-packing was performed and false otherwise.
  */
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
  /*
      Helper method to determine if Arm® KleidiAI™ dynamic quantization pre-packing policies are satisfied.

      Checks for the presence of the constant input tensor B, symmetry on the zero point and validity of the scales.
      Also checks if the shape of the tensor B is supported by KleidiAI and if bias tensor is also a constant input.
      Makes B transposition if necessary.
      Sets can_use_dynamic_quant_mlas_ flag accordingly and returns true if all policies are satisfied.
  */
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

    ctx.K = static_cast<size_t>(b_shape_[0]);
    ctx.N = static_cast<size_t>(b_shape_[1]);
    ctx.b_data = static_cast<const uint8_t*>(tensor.DataRaw());

    if (IsBTransposed()) {
      std::swap(ctx.K, ctx.N);
      ctx.b_data = quantization::TransPoseInputData(ctx.b_data, ctx.transposed_buffer, alloc, ctx.N, ctx.K);
    }

    // KleidiAI dynamic-qgemm packing is not expected to handle degenerate shapes.
    // If K==0 there is nothing to reduce over, and the RHS packer may dereference invalid memory.
    if (ctx.K == 0 || ctx.N == 0) {
      return false;
    }

    if (ctx.bias != nullptr) {
      dynamic_quant_mlas_bias_data_was_packed_ = true;
    }

    can_use_dynamic_quant_mlas_ = true;
    return true;
  }
  /*
    Helper method to execute Arm® KleidiAI™ dynamic quantization pre-packing.

    If can_use_dynamic_quant_mlas_ flag was true from previous policy controls then it checks the packed
    RHS matrix size in bytes and allocates the packed buffer. If the size is 0 returns false.
    It then assigns the scale and bias data accordingly and calls the packing function.
    It caches this pre-packed buffer as Mlas does.
  */
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
      return false;
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
  /*
  Helper for checking the zero points tensor of the input. Arm® KleidiAI™ supports symmetric zero points.

  This helper method checks if the zero point tensor, if it's present in the inputs with its index, it checks the data type either uint8_t or int8_t.
  It also checks if all the zero point values are zeros. If not, sets the can_use_dynamic_quant_mlas_ flag to false.
  If zero point tensor is not present, it sets the flag true as symmetric zero point is assumed.
  Returns the flag.
  */
  bool IsZeroPointSymmetric() {
    const Tensor* b_zp_constant_tensor = GetConstantInputTensor(GetBZeroPointIdx());
    if (b_zp_constant_tensor != nullptr) {
      assert(b_zp_constant_tensor->IsDataType<uint8_t>() || b_zp_constant_tensor->IsDataType<int8_t>());
      const auto* zp_bytes = static_cast<const std::byte*>(b_zp_constant_tensor->DataRaw());
      const size_t zp_size_in_bytes = b_zp_constant_tensor->SizeInBytes();
      return std::none_of(zp_bytes, zp_bytes + zp_size_in_bytes,
                          [](std::byte v) { return v != std::byte{0}; });
    }

    const auto input_defs = Info().node().InputDefs();
    const int b_zp_idx = GetBZeroPointIdx();
    const bool b_zp_input_exists = b_zp_idx >= 0 &&
                                   static_cast<size_t>(b_zp_idx) < input_defs.size() &&
                                   input_defs[b_zp_idx]->Exists();
    return !b_zp_input_exists;
  }
  /*
  Helper method to check the validity of the scales tensor for Arm® KleidiAI™ dynamic quantization.
  Scales are invalid and can_use_dynamic_quant_mlas_ flag is false returns if the float scales are non-finite or non-positive.
  Otherwise can_use_dynamic_quant_mlas_ flag returned true.
  */
  bool AreScalesValid(const Tensor& b_scale_tensor) {
    const auto bs = b_scale_tensor.DataAsSpan<float>();
    const bool has_invalid =
        std::any_of(bs.begin(), bs.end(),
                    [](float s) { return !std::isfinite(s) || s <= 0.0f; });

    return !has_invalid;
  }
  /*
    Helper to promote a 1D tensor to 2D, for Arm® KleidiAI™ dynamic quantization, if necessary. Returns false if the tensor rank is 0.
  */
  bool PromoteBShapeIfNeeded() {
    if (b_shape_.NumDimensions() == 0) {
      return false;  // rank-0 tensor is not supported
    }

    if (b_shape_.NumDimensions() == 1) {
      TensorShapeVector expanded{1, b_shape_[0]};
      b_shape_ = TensorShape(expanded);
    }

    return true;
  }
  /*
    Helper method to check the shape policy of the tensor B is passes for Arm® KleidiAI™ dynamic quantization.
    The shape should be at least 2D and all the dimensions except the last two should be 1. 1D tensor is promoted to 2D.
  */
  bool IsBShapeSupportedForDynamicQuant(const TensorShape& tensor_shape) {
    b_shape_ = tensor_shape;
    if (!PromoteBShapeIfNeeded()) {
      return false;
    }

    for (size_t i = 0; i < (b_shape_.NumDimensions() - 2); ++i) {
      if (b_shape_[i] != 1) {
        return false;
      }
    }
    b_shape_ = tensor_shape;
    return true;
  }
  /*
    Checks against the constant initialized tensor index and returns the constant tensor if present.
    Returns nullptr if index is invalid or the tensor is not held by the kernel instance.
  */
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
