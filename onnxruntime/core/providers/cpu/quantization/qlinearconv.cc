// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/cpuid_info.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

using ConvPadVector = ConvAttributes::ConvPadVector;

template <typename ActType>
class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    channels_last_ = (info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(0)) != 0);
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,
    IN_W = 3,
    IN_W_SCALE = 4,
    IN_W_ZERO_POINT = 5,
    IN_Y_SCALE = 6,
    IN_Y_ZERO_POINT = 7,
    IN_BIAS = 8
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  inline static bool IsValidQuantParam(const Tensor* quant_param, int64_t N) {
    const auto& shape = quant_param->Shape();
    return (shape.NumDimensions() == 0 || (shape.NumDimensions() == 1 && (shape[0] == 1 || shape[0] == N)));
  }

  static void ComputeOffset(OpKernelContext* context,
                            int64_t M,
                            ActType& X_zero_point_value,
                            ActType& Y_zero_point_value,
                            uint8_t& W_zero_point_value) {
    const Tensor* X_zero_point = context->Input<Tensor>(InputTensors::IN_X_ZERO_POINT);
    const Tensor* W_zero_point = context->Input<Tensor>(InputTensors::IN_W_ZERO_POINT);
    const Tensor* Y_zero_point = context->Input<Tensor>(InputTensors::IN_Y_ZERO_POINT);
    ORT_ENFORCE(IsScalarOr1ElementVector(X_zero_point),
                "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
    ORT_ENFORCE(IsScalarOr1ElementVector(Y_zero_point),
                "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");
    ORT_ENFORCE(IsValidQuantParam(W_zero_point, M),
                "QLinearConv : filter zero point shape invalid");

    X_zero_point_value = *(X_zero_point->Data<ActType>());
    Y_zero_point_value = *(Y_zero_point->Data<ActType>());

    const int64_t W_zero_point_size = W_zero_point->Shape().Size();
    const auto* W_zero_point_data = static_cast<const uint8_t*>(W_zero_point->DataRaw());
    W_zero_point_value = W_zero_point_data[0];
    for (int64_t i = 1; i < W_zero_point_size; i++) {
      ORT_ENFORCE(W_zero_point_data[i] == W_zero_point_value,
                  "QLinearConv : zero point of per-channel filter must be same. "
                  "This happens by design if the quantization is symmetric.");
    }
  }

  static std::vector<float> ComputeOutputScale(OpKernelContext* context,
                                               int64_t M) {
    const Tensor* X_scale = context->Input<Tensor>(InputTensors::IN_X_SCALE);
    const Tensor* W_scale = context->Input<Tensor>(InputTensors::IN_W_SCALE);
    const Tensor* Y_scale = context->Input<Tensor>(InputTensors::IN_Y_SCALE);
    ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
                "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
    ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
                "QLinearConv : result scale must be a scalar or 1D tensor of size 1");
    ORT_ENFORCE(IsValidQuantParam(W_scale, M),
                "QLinearConv : filter scale shape invalid");

    auto X_scale_value = *(X_scale->Data<float>());
    auto Y_scale_value = *(Y_scale->Data<float>());

    std::vector<float> output_scales;
    const int64_t W_scale_size = W_scale->Shape().Size();
    const auto* W_scale_data = W_scale->Data<float>();
    output_scales.resize(static_cast<size_t>(W_scale_size));
    for (int64_t i = 0; i < W_scale_size; i++) {
      output_scales[onnxruntime::narrow<size_t>(i)] = (X_scale_value * W_scale_data[i] / Y_scale_value);
    }

    return output_scales;
  }

  /**
   * @brief Computes the partition stride of the activation tensor.
   *
   *        Current threaded job partiton is limited in that we can't
   *        partition the filter tensor (a TODO item). So we can only
   *        horizontally partition the activation tensor into thin
   *        slices. This function decides the thickness of that slice,
   *        which is also number of output pixels each job produces.
   *
   * @param degree_of_parallelism  Configured thread parallelism for this run
   * @param output_image_size      Number of pixels in the output image
   * @param group_output_channels  Number of filters in this group.
   * @param kernel_dim             Dimension of a filter
   * @param comp_kernel_stride     Best stride to fully utilize hand tuned computing kernel.
   * @return
   */
  static int32_t ComputeOutputStride(int32_t degree_of_parallelism,
                                     int64_t output_image_size,
                                     int64_t group_output_channels,
                                     int64_t kernel_dim,
                                     int64_t comp_kernel_stride) {
    //
    // The idea is to simply partition the activation tensor using the computation kernel stride, to ensure
    // the hand crafted kernel code has maximum throughput in almost all the jobs. Most of the below logic,
    // however, is to take care of corner cases where we have either too few or too many partitions.
    //
    constexpr double MIN_COMPLEXITY = static_cast<double>(64 * 1024);

    const int64_t weights = group_output_channels * kernel_dim;
    const int32_t min_stride = static_cast<int32_t>(std::ceil(MIN_COMPLEXITY / static_cast<double>(weights)));

    int32_t output_stride = static_cast<int32_t>(comp_kernel_stride);

    if (output_stride < min_stride) {
      output_stride = (min_stride + output_stride - 1) / output_stride * output_stride;
    }

    const auto task_count = (output_image_size + output_stride - 1) / output_stride;
#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
    const auto large_jobs = degree_of_parallelism << 6;
#else
    const auto large_jobs = degree_of_parallelism * 5;
#endif
    if (task_count > large_jobs) {
      // too many tasks, need a bigger stride
      output_stride = static_cast<int32_t>(((output_image_size + large_jobs - 1) / large_jobs + comp_kernel_stride - 1) / comp_kernel_stride * comp_kernel_stride);
    }

    // We need a better partiton when we have a big filter tensor and very small activation tensor
    // TODO!! we should partition the weight tensor instead
    constexpr int64_t BIG_WEIGHT = 1024 * 1024;
    if (weights >= BIG_WEIGHT && task_count < (degree_of_parallelism / 8)) {
      int32_t s1 = static_cast<int32_t>((output_image_size + degree_of_parallelism - 1) / degree_of_parallelism);
      output_stride = std::max(s1, min_stride);
    }

    return output_stride;
  }

  bool TryConvSymPrepack(const uint8_t* Wdata,
                         AllocatorPtr alloc,
                         size_t output_channels,
                         size_t group_count,
                         size_t group_input_channels,
                         size_t group_output_channels,
                         size_t kernel_size) {
    const Tensor* X_zero_point = nullptr;
    const Tensor* W_zero_point = nullptr;

    // We need activation and weight zero points for symmetric packing
    if (!Info().TryGetConstantInput(InputTensors::IN_X_ZERO_POINT, &X_zero_point) || !IsScalarOr1ElementVector(X_zero_point) ||
        !Info().TryGetConstantInput(InputTensors::IN_W_ZERO_POINT, &W_zero_point) || !IsValidQuantParam(W_zero_point, static_cast<int64_t>(output_channels))) {
      return false;
    }

    auto X_zero_point_value = *(X_zero_point->Data<ActType>());
    const size_t W_zero_point_size = static_cast<size_t>(W_zero_point->Shape().Size());
    const auto* W_zero_point_data = W_zero_point->Data<int8_t>();
    if (!std::all_of(W_zero_point_data, W_zero_point_data + W_zero_point_size, [](int8_t v) { return v == 0; })) {
      // Symmetric means weight zero point must be zero
      return false;
    }

    // Try indirect conv packing
    size_t packed_size = MlasConvSymPackWSize(group_count, group_input_channels, group_output_channels, kernel_size, std::is_signed<ActType>::value);
    if (packed_size != 0) {
      const Tensor* B = nullptr;
      Info().TryGetConstantInput(8, &B);
      const auto* Bdata = B != nullptr ? B->Data<int32_t>() : nullptr;

      column_sums_.resize(output_channels);
      const int8_t* sdata = (const int8_t*)Wdata;
      int32_t X_zero_point_fixup = MlasConvSymFixupInputZeroPoint(X_zero_point_value, std::is_signed<ActType>::value);
      for (size_t oc = 0; oc < output_channels; oc++) {
        int32_t sum = 0;
        for (size_t ks = 0; ks < kernel_size * group_input_channels; ks++) {
          sum += *sdata++;
        }
        column_sums_[oc] = (Bdata != nullptr ? Bdata[oc] : 0) - sum * X_zero_point_fixup;
      }

      auto* packed_W = static_cast<uint8_t*>(alloc->Alloc(packed_size));
      packed_W_buffer_ = BufferUniquePtr(packed_W, BufferDeleter(alloc));

      MlasConvSymPackW(group_count,
                       group_input_channels,
                       group_output_channels,
                       kernel_size,
                       reinterpret_cast<const int8_t*>(Wdata),
                       reinterpret_cast<int8_t*>(packed_W),
                       packed_size,
                       std::is_signed<ActType>::value);

      is_symmetric_conv_ = true;
      is_W_packed_ = true;
      return true;
    }

    // Try symmetric GEMM packing
    // Don't pack the filter buffer if the MlasConvDepthwise path is used.
    if (group_input_channels != 1 || group_output_channels != 1) {
      const size_t kernel_dim = group_input_channels * kernel_size;
      packed_W_size_ = MlasSymmQgemmPackBSize(group_output_channels,
                                              kernel_dim,
                                              std::is_same<ActType, int8_t>::value);

      if (packed_W_size_ != 0) {
        size_t packed_W_data_size = SafeInt<size_t>(group_count) * packed_W_size_;
        auto* packed_W = static_cast<uint8_t*>(alloc->Alloc(packed_W_data_size));

        memset(packed_W, 0, packed_W_data_size);

        packed_W_buffer_ = BufferUniquePtr(packed_W, BufferDeleter(alloc));

        // Allocate a temporary buffer to hold the reordered oihw->hwio filter for
        // a single group.
        //
        // Note: The size of this buffer is less than or equal to the size of the original
        // weight tensor, so the allocation size is guaranteed to fit inside size_t.
        auto* group_reordered_W = static_cast<int8_t*>(alloc->Alloc(group_output_channels * group_input_channels * kernel_size));
        BufferUniquePtr group_reordered_W_buffer(group_reordered_W, BufferDeleter(alloc));

        const size_t W_offset = group_output_channels * kernel_dim;

        for (int64_t group_id = 0; group_id < conv_attrs_.group; ++group_id) {
          ReorderFilter(Wdata, (uint8_t*)(group_reordered_W), group_output_channels, group_input_channels, kernel_size);
          MlasSymmQgemmPackB(group_output_channels,
                             kernel_dim,
                             group_reordered_W,
                             group_output_channels,
                             std::is_same<ActType, int8_t>::value,
                             X_zero_point_value,
                             packed_W);
          packed_W += packed_W_size_;
          Wdata += W_offset;
        }
        is_symmetric_gemm_ = true;
        is_W_packed_ = true;
        return true;
      }
    }

    return false;
  }

  // Reorder filter storage format from MCK1..Kn to K1...KnCM
  static void ReorderFilter(const uint8_t* input,
                            uint8_t* output,
                            size_t output_channels,
                            size_t input_channels,
                            size_t kernel_size) {
    for (size_t k = 0; k < kernel_size; k++) {
      for (size_t ic = 0; ic < input_channels; ic++) {
        for (size_t oc = 0; oc < output_channels; oc++) {
          size_t index = (oc * input_channels * kernel_size) + (ic * kernel_size) + k;
          *output++ = input[index];
        }
      }
    }
  }

  ConvAttributes conv_attrs_;
  TensorShape W_shape_;
  IAllocatorUniquePtr<void> packed_W_buffer_;
  size_t packed_W_size_{0};
  IAllocatorUniquePtr<void> reordered_W_buffer_;
  bool is_W_signed_{false};
  bool is_W_packed_{false};
  bool is_symmetric_conv_{false};
  bool is_symmetric_gemm_{false};
  bool channels_last_{false};
  std::vector<int32_t> column_sums_;
};

// uint8_t kernel supports weight being either uint8_t or int8_t
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<uint8_t>);

// int8_t kernel only supports weight being int8_t
#define REGISTER_QLINEARCONV_INT8_KERNEL(domain, version)                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      QLinearConv,                                                       \
      domain,                                                            \
      version,                                                           \
      int8_t,                                                            \
      kCpuExecutionProvider,                                             \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())   \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())   \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()), \
      QLinearConv<int8_t>);

REGISTER_QLINEARCONV_INT8_KERNEL(kOnnxDomain, 10);

#ifndef DISABLE_CONTRIB_OPS

namespace contrib {

// Register an alternate version of this kernel that supports the channels_last
// attribute in order to consume and produce NHWC tensors.
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<uint8_t>);

REGISTER_QLINEARCONV_INT8_KERNEL(kMSDomain, 1);

}  // namespace contrib

#endif

template <typename ActType>
Status QLinearConv<ActType>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                     /*out*/ bool& is_packed,
                                     /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  // Support packing the weight matrix.
  if (input_idx != InputTensors::IN_W) {
    return Status::OK();
  }

  is_W_signed_ = tensor.IsDataType<int8_t>();

  const auto& shape = tensor.Shape().GetDims();
  size_t rank = shape.size();
  if (rank <= 2) {
    return Status::OK();
  }

  const int64_t M = shape[0];
  const int64_t C = shape[1];

  // Verify that the total number of output channels is a multiple of the group count.
  if (M % conv_attrs_.group != 0) {
    return Status::OK();
  }

  // Note: The tensor has already been allocated with this tensor shape, so all
  // shape indices are guaranteed to fit inside size_t.
  const size_t output_channels = static_cast<size_t>(M);
  const size_t group_input_channels = static_cast<size_t>(C);
  const size_t kernel_size =
      static_cast<size_t>(std::accumulate(shape.data() + 2, shape.data() + rank, 1LL, std::multiplies<int64_t>()));

  const auto* Wdata = static_cast<const uint8_t*>(tensor.DataRaw());
  W_shape_ = shape;

  const size_t group_count = static_cast<size_t>(conv_attrs_.group);
  const size_t group_output_channels = output_channels / group_count;
  const size_t kernel_dim = group_input_channels * kernel_size;

  bool share_prepacked_weights = (prepacked_weights != nullptr);

  // Determine if the symmetric weight convolution path can be used. The weights must be
  // signed and all weight zero points must be zero.
  if (is_W_signed_ && TryConvSymPrepack(Wdata,
                                        alloc,
                                        output_channels,
                                        group_count,
                                        group_input_channels,
                                        group_output_channels,
                                        kernel_size)) {
    is_packed = true;
    return Status::OK();
  }

  // Don't pack the filter buffer if the MlasConvDepthwise path is used.
  if (group_input_channels != 1 || group_output_channels != 1) {
    packed_W_size_ = MlasGemmPackBSize(group_output_channels,
                                       kernel_dim,
                                       std::is_same<ActType, int8_t>::value,
                                       is_W_signed_);
    if (packed_W_size_ != 0) {
      size_t packed_W_data_size = SafeInt<size_t>(group_count) * packed_W_size_;
      packed_W_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, packed_W_data_size, true);
      auto* packed_W = static_cast<uint8_t*>(packed_W_buffer_.get());

      // Initialize memory to 0 as there could be some padding associated with pre-packed
      // buffer memory and we don not want it uninitialized and generate different hashes
      // if and when we try to cache this pre-packed buffer for sharing between sessions.
      memset(packed_W, 0, packed_W_data_size);

      // Allocate a temporary buffer to hold the reordered oihw->hwio filter for
      // a single group.
      //
      // Note: The size of this buffer is less than or equal to the size of the original
      // weight tensor, so the allocation size is guaranteed to fit inside size_t.
      auto group_reordered_W_buffer = IAllocator::MakeUniquePtr<void>(alloc, group_output_channels * group_input_channels * kernel_size, true);
      auto* group_reordered_W = static_cast<uint8_t*>(group_reordered_W_buffer.get());

      const size_t W_offset = group_output_channels * kernel_dim;

      for (int64_t group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        ReorderFilter(Wdata, group_reordered_W, group_output_channels, group_input_channels, kernel_size);
        MlasGemmPackB(group_output_channels,
                      kernel_dim,
                      group_reordered_W,
                      group_output_channels,
                      std::is_same<ActType, int8_t>::value,
                      is_W_signed_,
                      packed_W);
        packed_W += packed_W_size_;
        Wdata += W_offset;
      }

      if (share_prepacked_weights) {
        prepacked_weights->buffers_.push_back(std::move(packed_W_buffer_));
        prepacked_weights->buffer_sizes_.push_back(packed_W_data_size);
      }

      is_W_packed_ = true;
      is_packed = true;
      return Status::OK();
    }
  }

  if (share_prepacked_weights) {
    prepacked_weights->buffers_.push_back(nullptr);  // packed_W_buffer_ is nullptr
    prepacked_weights->buffer_sizes_.push_back(0);
  }

  size_t reordered_w_data_size = SafeInt<size_t>(sizeof(uint8_t)) * output_channels * group_input_channels * kernel_size;
  reordered_W_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, reordered_w_data_size, true);
  uint8_t* reordered_W = static_cast<uint8_t*>(reordered_W_buffer_.get());

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(reordered_W, 0, reordered_w_data_size);

  ReorderFilter(Wdata, reordered_W, output_channels, group_input_channels, kernel_size);

  if (share_prepacked_weights) {
    prepacked_weights->buffers_.push_back(std::move(reordered_W_buffer_));
    prepacked_weights->buffer_sizes_.push_back(reordered_w_data_size);
  }

  is_W_packed_ = true;
  is_packed = true;
  return Status::OK();
}

template <typename ActType>
Status QLinearConv<ActType>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                       int input_idx,
                                                       /*out*/ bool& used_shared_buffers) {
  if (input_idx != 3) {
    return Status::OK();
  }

  used_shared_buffers = true;

  if (prepacked_buffers.size() == 1) {  // This means that only packed_W_ exists
    packed_W_buffer_ = std::move(prepacked_buffers[0]);
  } else if (prepacked_buffers.size() == 2) {  // This means that only reordered_W_ exists
    // Enforce that the first "placeholder" buffer is nullptr
    ORT_ENFORCE(prepacked_buffers[0].get() == nullptr);
    reordered_W_buffer_ = std::move(prepacked_buffers[1]);
  }

  return Status::OK();
}

template <typename ActType>
Status QLinearConv<ActType>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(InputTensors::IN_X);
  const Tensor* W = is_W_packed_ ? nullptr : context->Input<Tensor>(InputTensors::IN_W);
  const auto& W_shape = W ? W->Shape() : W_shape_;
  const bool is_W_signed = (W != nullptr) ? W->IsDataType<int8_t>() : is_W_signed_;

  const int64_t N = X->Shape()[0];
  const int64_t M = W_shape[0];

  ActType X_zero_point_value;
  ActType Y_zero_point_value;
  uint8_t W_zero_point_value;
  ComputeOffset(context, M, X_zero_point_value, Y_zero_point_value, W_zero_point_value);
  std::vector<float> output_scales = ComputeOutputScale(context, M);

  const Tensor* B = context->Input<Tensor>(InputTensors::IN_BIAS);

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W_shape, channels_last_));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));

  const size_t kernel_rank = kernel_shape.size();

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
  }

  const int64_t C = X->Shape()[channels_last_ ? 1 + kernel_rank : 1];
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;
  const size_t spatial_dim_end = spatial_dim_start + kernel_rank;

  TensorShapeVector Y_dims({N});
  if (!channels_last_) {
    Y_dims.push_back(M);
  }
  TensorShape input_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  if (channels_last_) {
    Y_dims.push_back(M);
  }
  Tensor* Y = context->Output(OutputTensors::OUT_Y, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(spatial_dim_start, spatial_dim_end);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  // Handle the case of a dynamic weight filter.
  BufferUniquePtr reordered_W_buffer;
  uint8_t* reordered_W = nullptr;
  if (!packed_W_buffer_) {
    if (W == nullptr) {
      // Weight was constant and reordered.
      reordered_W = static_cast<uint8_t*>(reordered_W_buffer_.get());
    } else {
      // Weight tensor was not constant or prepacking is disabled.
      reordered_W = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * W_shape.Size()));
      reordered_W_buffer = BufferUniquePtr(reordered_W, BufferDeleter(alloc));
      ReorderFilter(
          static_cast<const uint8_t*>(W->DataRaw()),
          reordered_W,
          static_cast<size_t>(M),
          static_cast<size_t>(W_shape[1]),
          static_cast<size_t>(kernel_size));
    }
  }

  int64_t group_count = conv_attrs_.group;
  int64_t group_input_channels = W_shape[1];
  int64_t group_output_channels = M / group_count;

  // Test for depthwise convolution.
  const bool is_depthwise_conv = ((is_symmetric_conv_ || reordered_W != nullptr) && group_input_channels == 1 && group_output_channels == 1);
  if (is_depthwise_conv) {
    // Update the input and output channels to the number of groups in order to
    // reuse as much of the below standard convolution path.
    group_input_channels = group_count;
    group_output_channels = group_count;
    group_count = 1;
  }

  const int64_t X_offset = C * input_image_size;
  const int64_t Y_offset = M * output_image_size;
  const int64_t kernel_dim = group_input_channels * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  // Use an intermediate int32_t buffer for the GEMM computation before
  // requantizing to the output type.
  //
  // This buffer is not needed for the symmetric convolution path as requantization
  // is fused with the GEMM compuation.
  BufferUniquePtr gemm_output_buffer;
  if (!is_symmetric_conv_) {
    auto* gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) * Y_offset);
    gemm_output_buffer = BufferUniquePtr(gemm_output_data, BufferDeleter(alloc));
  }

  const auto* Xdata = X->Data<ActType>();
  const auto* Bdata = B != nullptr ? B->Data<int32_t>() : nullptr;
  auto* Ydata = Y->MutableData<ActType>();

  BufferUniquePtr transpose_input_buffer;
  BufferUniquePtr transpose_output_buffer;

  // Allocate temporary buffers for transposing to channels last format.
  if (!channels_last_) {
    auto* transpose_input = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * (X_offset + MLAS_SYMM_QGEMM_BUF_OVERRUN));
    transpose_input_buffer = BufferUniquePtr(transpose_input, BufferDeleter(alloc));
    auto* transpose_output = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * Y_offset);
    transpose_output_buffer = BufferUniquePtr(transpose_output, BufferDeleter(alloc));
  }

  BufferUniquePtr col_buffer;
  BufferUniquePtr indirection_buffer;
  size_t ind_buf_length = 0;
  std::vector<ActType> padding_data;

  bool use_indirection_buffer = false;
  if (is_depthwise_conv) {
    use_indirection_buffer = true;
  } else if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    if (is_symmetric_conv_) {
      use_indirection_buffer = true;
    } else {
      // Pointwise convolutions can use the original input tensor in place,
      // otherwise a temporary buffer is required for the im2col transform.
      int64_t group_col_buffer_size = (kernel_rank > 2) ? group_count * col_buffer_size : col_buffer_size;
      group_col_buffer_size += MLAS_SYMM_QGEMM_BUF_OVERRUN;
      auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * group_col_buffer_size);
      col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
      memset(col_data, 0, SafeInt<size_t>(sizeof(ActType)) * group_col_buffer_size);
    }
  }

  bool parallel_batch = is_symmetric_conv_ && channels_last_;

  if (use_indirection_buffer) {
    // Allocate indirection buffer pointers and prepare a padding vector for
    // the im2col transform.
    ind_buf_length = SafeInt<size_t>(sizeof(const ActType*)) * kernel_size * output_image_size;
    if (parallel_batch)
      ind_buf_length *= SafeInt<size_t>(N);  // ind buffer per each image in the batch
    auto* indirection_data = alloc->Alloc(ind_buf_length);
    indirection_buffer = BufferUniquePtr(indirection_data, BufferDeleter(alloc));
    padding_data.resize(static_cast<size_t>(C), X_zero_point_value);
  }

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  /*************************************
   * Thread partition idea: we are essentially partition a GEMM A[M,K] x B[K,N].
   * Here B contains the conv filters, which are usually not big, so we assume
   * it can be in cache entirely. Then we simply partition A horizontally into
   * thin slices along M dimension. This would ensure that the slice of A fits
   * into the cache and reduce the chance of kernel waiting for memory.
   *
   * The thickness of A slice should be multiple of kernel stride M. Since
   * we have to choose from many different kernels, the logic of finding
   * the stride M is hacky.
   */

  // The following convoluted branches must match the kernel selection logic
  // in conv_worker.
  int64_t compute_stride;
  if (is_symmetric_conv_) {
    if (is_depthwise_conv) {
      compute_stride = MlasConvSymDepthwiseGetKernelOutputCnt(std::is_signed<ActType>::value);
    } else {
      compute_stride = MlasConvSymGetKernelOutputCount(std::is_signed<ActType>::value);
    }
  } else if (is_depthwise_conv) {
    compute_stride = MlasConvDepthwiseGetKernelOutputCnt();
  } else {
    if (is_symmetric_gemm_) {
      compute_stride = MlasSymmQgemmGetKernelOutputCnt();
    } else {
      compute_stride = MlasQgemmGetKernelOutputCnt(std::is_signed<ActType>::value, is_W_signed);
    }
  }

  const int32_t degree_of_par = concurrency::ThreadPool::DegreeOfParallelism(thread_pool);
  const int32_t stride_m = ComputeOutputStride(degree_of_par, output_image_size, group_output_channels, kernel_dim, compute_stride);
  const int64_t task_count = (output_image_size + stride_m - 1) / stride_m;

  if (parallel_batch)  // process all batch images in the same parallel section
  {
    auto conv_worker = [&](ptrdiff_t batch) {
      int64_t image_id = batch / task_count;
      int64_t output_start = (batch % task_count) * stride_m;
      int64_t output_count = std::min((int64_t)stride_m, output_image_size - output_start);

      auto* worker_input_image = Xdata + X_offset * image_id;

      ActType const** worker_indirection_buffer = nullptr;
      if (indirection_buffer) {
        size_t offset = SafeInt<size_t>(image_id * output_image_size + output_start) * kernel_size;
        assert(offset < ind_buf_length);
        worker_indirection_buffer = static_cast<ActType const**>(indirection_buffer.get()) + offset;

        math::Im2col<ActType, StorageOrder::NHWC>()(
            worker_input_image,
            C,
            input_shape.GetDims().data(),
            output_shape.GetDims().data(),
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<ptrdiff_t>(kernel_rank),
            output_start,
            output_count,
            worker_indirection_buffer,
            padding_data.data());
      }

      auto* worker_output = Ydata + Y_offset * image_id + output_start * M;

      MLAS_CONV_SYM_PARAMS conv_params = {};
      if (worker_indirection_buffer) {
        conv_params.InputIndirection = reinterpret_cast<void const**>(worker_indirection_buffer);
      } else {
        conv_params.InputDirect = worker_input_image + output_start * C;
      }
      conv_params.Filter = packed_W_buffer_.get();
      conv_params.Output = worker_output;
      conv_params.InputChannels = static_cast<size_t>(C);
      conv_params.OutputChannels = static_cast<size_t>(M);
      conv_params.OutputCount = static_cast<size_t>(output_count);
      conv_params.KernelSize = static_cast<size_t>(kernel_size);
      conv_params.Bias = column_sums_.data();
      conv_params.Scale = output_scales.data();
      conv_params.PerChannelScale = output_scales.size() > 1;
      conv_params.OutputZeroPoint = Y_zero_point_value;
      conv_params.InputIsSigned = std::is_signed<ActType>::value;

      if (is_depthwise_conv) {
        MlasConvSymDepthwise(conv_params);
      } else {
        MlasConvSym(conv_params);
      }
    };

    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, onnxruntime::narrow<ptrdiff_t>(task_count * N), conv_worker);

    return Status::OK();
  }

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    const auto* input_data = Xdata;
    auto* output_data = Ydata;

    if (!channels_last_) {
      // Transpose the input from channels first (NCHW) to channels last (NHWC).
      MlasTranspose(
          Xdata,
          static_cast<ActType*>(transpose_input_buffer.get()),
          static_cast<size_t>(C),
          static_cast<size_t>(input_image_size));
      input_data = static_cast<ActType*>(transpose_input_buffer.get());
      output_data = static_cast<ActType*>(transpose_output_buffer.get());
    }

    // Threaded implementation of ND convolution is not yet supported, so
    // prepare all im2col transformations here.
    if (col_buffer && kernel_rank > 2) {
      for (int64_t group_id = 0; group_id < group_count; ++group_id) {
        math::Im2col<ActType, StorageOrder::NHWC>()(
            input_data + group_id * group_input_channels,
            group_input_channels,
            C,
            input_shape.GetDims().data(),
            output_shape.GetDims().data(),
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<ptrdiff_t>(kernel_rank),
            static_cast<ActType*>(col_buffer.get()) + group_id * col_buffer_size,
            X_zero_point_value);
      }
    }

    auto conv_worker = [&](ptrdiff_t batch) {
      int64_t output_start = (int64_t)batch * (int64_t)stride_m;
      int64_t output_count = std::min((int64_t)stride_m, output_image_size - output_start);

      ActType const** worker_indirection_buffer = nullptr;
      if (indirection_buffer) {
        worker_indirection_buffer = static_cast<ActType const**>(indirection_buffer.get()) + output_start * kernel_size;
        math::Im2col<ActType, StorageOrder::NHWC>()(
            input_data,
            C,
            input_shape.GetDims().data(),
            output_shape.GetDims().data(),
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<ptrdiff_t>(kernel_rank),
            output_start,
            output_count,
            worker_indirection_buffer,
            padding_data.data());
      }

      auto* worker_output = output_data + output_start * M;

      if (is_symmetric_conv_) {
        MLAS_CONV_SYM_PARAMS conv_params = {};
        if (worker_indirection_buffer) {
          conv_params.InputIndirection = reinterpret_cast<void const**>(worker_indirection_buffer);
        } else {
          conv_params.InputDirect = input_data + output_start * C;
        }
        conv_params.Filter = packed_W_buffer_.get();
        conv_params.Output = worker_output;
        conv_params.InputChannels = static_cast<size_t>(C);
        conv_params.OutputChannels = static_cast<size_t>(M);
        conv_params.OutputCount = static_cast<size_t>(output_count);
        conv_params.KernelSize = static_cast<size_t>(kernel_size);
        conv_params.Bias = column_sums_.data();
        conv_params.Scale = output_scales.data();
        conv_params.PerChannelScale = output_scales.size() > 1;
        conv_params.OutputZeroPoint = Y_zero_point_value;
        conv_params.InputIsSigned = std::is_signed<ActType>::value;

        if (is_depthwise_conv) {
          MlasConvSymDepthwise(conv_params);
        } else {
          MlasConvSym(conv_params);
        }
        return;
      }

      auto* worker_gemm_output = static_cast<int32_t*>(gemm_output_buffer.get()) + output_start * M;

      if (is_depthwise_conv) {
        MlasConvDepthwise(
            reinterpret_cast<const void* const*>(worker_indirection_buffer),
            X_zero_point_value,
            std::is_signed<ActType>::value,
            reinterpret_cast<const void* const*>(reordered_W),
            W_zero_point_value,
            is_W_signed,
            worker_gemm_output,
            static_cast<size_t>(M),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      } else {
        for (int64_t group_id = 0; group_id < group_count; ++group_id) {
          // Prepare the im2col transformation or use the input buffer directly for
          // pointwise convolutions.
          const auto* group_input_data = input_data + group_id * group_input_channels;
          const uint8_t* AData;
          size_t lda;
          if (col_buffer) {
            auto* worker_col_buffer = static_cast<ActType*>(col_buffer.get()) + output_start * kernel_dim;
            if (kernel_rank == 2) {
              math::Im2col<ActType, StorageOrder::NHWC>()(
                  group_input_data,
                  group_input_channels,
                  C,
                  input_shape[0],
                  input_shape[1],
                  kernel_shape[0],
                  kernel_shape[1],
                  dilations[0],
                  dilations[1],
                  pads[0],
                  pads[1],
                  strides[0],
                  strides[1],
                  output_shape[1],
                  output_start,
                  output_count,
                  worker_col_buffer,
                  X_zero_point_value);
            } else if (kernel_rank == 1) {
              math::Im2col<ActType, StorageOrder::NHWC>()(
                  group_input_data,
                  group_input_channels,
                  C,
                  1,
                  input_shape[0],
                  1,
                  kernel_shape[0],
                  1,
                  dilations[0],
                  0,
                  pads[0],
                  1,
                  strides[0],
                  output_shape[0],
                  output_start,
                  output_count,
                  worker_col_buffer,
                  X_zero_point_value);
            } else {
              // Use the im2col buffer prepared outside the thread, indexed by group.
              worker_col_buffer += group_id * col_buffer_size;
            }
            AData = reinterpret_cast<const uint8_t*>(worker_col_buffer);
            lda = static_cast<size_t>(kernel_dim);
          } else {
            AData = reinterpret_cast<const uint8_t*>(group_input_data + output_start * C);
            lda = static_cast<size_t>(C);
          }

          MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
          gemm_shape.M = static_cast<size_t>(output_count);
          gemm_shape.N = static_cast<size_t>(group_output_channels);
          gemm_shape.K = static_cast<size_t>(kernel_dim);
          gemm_shape.AIsSigned = std::is_signed<ActType>::value;
          gemm_shape.BIsSigned = is_W_signed;

          if (is_symmetric_gemm_) {
            MLAS_SYMM_QGEMM_DATA_PARAMS symm_gemm;
            symm_gemm.A = AData;
            symm_gemm.lda = lda;
            symm_gemm.C = worker_gemm_output + group_id * group_output_channels;
            symm_gemm.ldc = static_cast<size_t>(M);
            symm_gemm.B = static_cast<const int8_t*>(packed_W_buffer_.get()) + group_id * packed_W_size_,
            MlasSymmQgemmBatch(gemm_shape, &symm_gemm, 1, nullptr);
          } else {
            MLAS_GEMM_QUANT_DATA_PARAMS gemm_params;
            gemm_params.ZeroPointA = static_cast<uint8_t>(X_zero_point_value);
            gemm_params.A = AData;
            gemm_params.lda = lda;
            if (packed_W_buffer_) {
              gemm_params.B = static_cast<const int8_t*>(packed_W_buffer_.get()) + group_id * packed_W_size_,
              gemm_params.BIsPacked = true;
            } else {
              gemm_params.B = reordered_W + group_id * group_output_channels,
              gemm_params.ldb = static_cast<size_t>(M);
            }
            gemm_params.ZeroPointB = &W_zero_point_value;
            gemm_params.C = worker_gemm_output + group_id * group_output_channels;
            gemm_params.ldc = static_cast<size_t>(M);

            MlasGemm(gemm_shape, gemm_params, nullptr);
          }
        }
      }

      MlasRequantizeOutput(
          worker_gemm_output,
          static_cast<size_t>(M),
          worker_output,
          static_cast<size_t>(M),
          Bdata,
          output_scales.data(),
          output_scales.size() > 1,
          Y_zero_point_value,
          0,
          0,
          static_cast<size_t>(output_count),
          static_cast<size_t>(M));
    };

    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, onnxruntime::narrow<ptrdiff_t>(task_count), conv_worker);

    if (!channels_last_) {
      // Transpose the output from channels last (NHWC) to channels first (NCHW).
      MlasTranspose(
          output_data,
          Ydata,
          static_cast<size_t>(output_image_size),
          static_cast<size_t>(M));
    }

    Xdata += X_offset;
    Ydata += Y_offset;
  }

  return Status::OK();
}

}  // namespace onnxruntime
