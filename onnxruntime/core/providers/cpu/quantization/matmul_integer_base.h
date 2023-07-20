// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
    if (input_idx != GetBIdx()) {
      return Status::OK();
    }

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

    if (b_is_signed_ && TrySymmetricPrePack((const int8_t*)b_data, K, N, a_is_signed, alloc, prepacked_weights)) {
      is_packed = true;
      return Status::OK();
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
  virtual int GetAZeroPointIdx() const { return -1; }
  virtual int GetBIdx() const = 0;
  virtual int GetBZeroPointIdx() const = 0;

  virtual bool IsBTransposed() const {
    return false;
  }

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

  bool TrySymmetricPrePack(const int8_t* b_data, size_t K, size_t N, bool a_is_signed, AllocatorPtr& alloc, PrePackedWeights* prepacked_weights) {
    // We use symmetric qgemm when right hand side is signed int with zeropoint == 0
    const Tensor* W_zero_point = nullptr;
    if (!Info().TryGetConstantInput(GetBZeroPointIdx(), &W_zero_point)) {
      return false;
    }

    const size_t W_zero_point_size = static_cast<size_t>(W_zero_point->Shape().Size());
    const auto* W_zero_point_data = W_zero_point->Data<int8_t>();
    if (!std::all_of(W_zero_point_data, W_zero_point_data + W_zero_point_size, [](int8_t v) { return v == 0; })) {
      // Symmetric means weight zero point must be zero
      return false;
    }

    auto packed_b_size = MlasSymmQgemmPackBSize(N, K, a_is_signed);
    if (packed_b_size == 0) {
      // not supported on this platform
      return false;
    }

    const Tensor* X_zero_point = nullptr;
    auto x_zp_idx = GetAZeroPointIdx();
    if (x_zp_idx != -1 && Info().TryGetConstantInput(x_zp_idx, &X_zero_point) && IsScalarOr1ElementVector(X_zero_point)) {
      x_zero_point_ = a_is_signed ? *(X_zero_point->Data<int8_t>()) : *(X_zero_point->Data<uint8_t>());
    }

    packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size, true);
    // Initialize memory to 0 as there could be some padding associated with pre-packed
    // buffer memory and we don not want it uninitialized and generate different hashes
    // if and when we try to cache this pre-packed buffer for sharing between sessions.
    memset(packed_b_.get(), 0, packed_b_size);
    MlasSymmQgemmPackB(N, K, b_data, N, a_is_signed, x_zero_point_, packed_b_.get());

    bool share_prepacked_weights = (prepacked_weights != nullptr);
    if (share_prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_b_));
      prepacked_weights->buffer_sizes_.push_back(packed_b_size);
    }
    b_is_symmetrically_packed_ = true;

    return true;
  }


  bool b_is_signed_{true};
  bool b_is_symmetrically_packed_{false};
  TensorShape b_shape_;
  int32_t x_zero_point_{-1};
  IAllocatorUniquePtr<void> packed_b_;
};

}  // namespace onnxruntime
