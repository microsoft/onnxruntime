// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/framework/prepacked_weights.h"
#include "core/mlas/inc/mlas_t8.h"
#include "core/providers/common.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime::contrib {

class MatMulTernary final : public OpKernel {
 public:
  explicit MatMulTernary(const OpKernelInfo& info) : OpKernel(info) {
    const int64_t k = info.GetAttr<int64_t>("K"), n = info.GetAttr<int64_t>("N");
    ORT_ENFORCE(k > 0 && k % 128 == 0 && n > 0, "MatMulTernary requires K > 0 divisible by 128 and N > 0");
    ORT_ENFORCE(info.GetAttr<int64_t>("format_version") == 1 &&
                    info.GetAttr<std::string>("layout") == "unit5x128-planes-v1",
                "Unsupported MatMulTernary T8 format version or layout");
    k_ = narrow<size_t>(k);
    n_ = narrow<size_t>(n);
    const auto packing = info.GetConfigOptions().GetConfigOrDefault(kOrtSessionOptionsConfigTernaryPrepack, "current");
    ORT_ENFORCE(packing == "current" || packing == "tiled" || packing == "masks",
                "session.ternary.prepack must be current, tiled, or masks");
    use_packed_ = packing != "current";
    layout_ = packing == "masks" ? MLAS_T8_PACKED_LAYOUT::Masks : MLAS_T8_PACKED_LAYOUT::Tiled;
    const auto row_tile = info.GetConfigOptions().GetConfigOrDefault(kOrtSessionOptionsConfigTernaryRowTile, "1");
    ORT_ENFORCE(row_tile == "1" || row_tile == "2", "session.ternary.row_tile must be 1 or 2");
    row_tile_ = row_tile == "2" ? 2 : 1;
    ORT_ENFORCE(use_packed_ || row_tile_ == 1, "current T8 kernel requires row_tile 1");
    const Tensor* planes = nullptr;
    if (info.TryGetConstantInput(1, &planes) && planes != nullptr) {
      ORT_THROW_IF_ERROR(ValidatePlanes(*planes));
      constant_planes_ = true;
    }
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr allocator,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override {
    is_packed = false;
    if (input_idx != 1 || !use_packed_) return Status::OK();
    ORT_RETURN_IF_ERROR(ValidatePlanes(tensor));
    packed_ = IAllocator::MakeUniquePtr<uint32_t>(allocator, PackedWords(), true);
    MlasT8Pack(n_, k_, tensor.Data<uint32_t>(), packed_.get(), layout_);
    constant_planes_ = true;
    is_packed = true;
    // The packed layout depends on session.ternary.prepack, which the shared container cannot key on
    // yet, so keep the buffer owned by this kernel instead of publishing it for cross-session sharing.
    if (prepacked_weights != nullptr) prepacked_weights->has_kernel_owned_packed_weights_ = true;
    return Status::OK();
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* a = context->Input<Tensor>(0);
    const Tensor* planes = packed_ ? nullptr : context->Input<Tensor>(1);
    const Tensor* scales = context->Input<Tensor>(2);
    const size_t units = n_ / 5 + (n_ % 5 != 0);
    ORT_RETURN_IF_NOT(a->Shape().NumDimensions() >= 1 && a->Shape().GetDims().back() == static_cast<int64_t>(k_),
                      "MatMulTernary A must have rank >= 1 and last dimension K");
    ORT_RETURN_IF_NOT(scales->Shape() == TensorShape({narrow<int64_t>(units), narrow<int64_t>(k_ / 128), 5}),
                      "MatMulTernary scales must have shape [ceil(N/5), K/128, 5]");
    if (!constant_planes_) ORT_RETURN_IF_ERROR(ValidatePlanes(*planes));
    TensorShapeVector shape(a->Shape().GetDims().begin(), a->Shape().GetDims().end());
    shape.back() = narrow<int64_t>(n_);
    Tensor* output = context->Output(0, TensorShape(shape));
    const size_t m = narrow<size_t>(a->Shape().Size() / narrow<int64_t>(k_));
    if (m == 0) return Status::OK();
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    auto workspace = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(m) * k_, true);
    if (use_packed_) {
      IAllocatorUniquePtr<uint32_t> dynamic_packed;
      const uint32_t* packed = packed_.get();
      if (packed == nullptr) {
        dynamic_packed = IAllocator::MakeUniquePtr<uint32_t>(allocator, PackedWords(), true);
        MlasT8Pack(n_, k_, planes->Data<uint32_t>(), dynamic_packed.get(), layout_);
        packed = dynamic_packed.get();
      }
      MlasT8GemmPacked(m, n_, k_, a->Data<float>(), packed, scales->Data<MLFloat16>(),
                       output->MutableData<float>(), workspace.get(), context->GetOperatorThreadPool(), layout_, row_tile_);
    } else {
      MlasT8Gemm(m, n_, k_, a->Data<float>(), planes->Data<uint32_t>(), scales->Data<MLFloat16>(),
                 output->MutableData<float>(), workspace.get(), context->GetOperatorThreadPool());
    }
    return Status::OK();
  }

 private:
  size_t PackedWords() const {
    return SafeInt<size_t>(n_ / 5 + (n_ % 5 != 0)) * (k_ / 128) *
           (layout_ == MLAS_T8_PACKED_LAYOUT::Tiled ? 32 : 40);
  }

  Status ValidatePlanes(const Tensor& planes) const {
    const size_t units = n_ / 5 + (n_ % 5 != 0);
    ORT_RETURN_IF_NOT(planes.Shape() == TensorShape({narrow<int64_t>(units), 8, narrow<int64_t>(k_ / 32)}),
                      "MatMulTernary B must have shape [ceil(N/5), 8, K/32]");
    ORT_RETURN_IF_NOT(MlasT8Validate(n_, k_, planes.Data<uint32_t>()),
                      "MatMulTernary B contains an invalid T8 byte (243..255)");
    return Status::OK();
  }

  size_t k_{}, n_{};
  bool constant_planes_{false};
  bool use_packed_{false};
  size_t row_tile_{1};
  MLAS_T8_PACKED_LAYOUT layout_{MLAS_T8_PACKED_LAYOUT::Tiled};
  IAllocatorUniquePtr<uint32_t> packed_{};
};

ONNX_OPERATOR_KERNEL_EX(
    MatMulTernary, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TPlanes", DataTypeImpl::GetTensorType<uint32_t>())
        .TypeConstraint("TScale", DataTypeImpl::GetTensorType<MLFloat16>()),
    MatMulTernary);

}  // namespace onnxruntime::contrib
