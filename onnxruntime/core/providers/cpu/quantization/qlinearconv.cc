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

    X_zero_point_value = *(X_zero_point->template Data<ActType>());
    Y_zero_point_value = *(Y_zero_point->template Data<ActType>());

    const int64_t W_zero_point_size = W_zero_point->Shape().Size();
    const auto* W_zero_point_data = static_cast<const uint8_t*>(W_zero_point->DataRaw());
    W_zero_point_value = W_zero_point_data[0];
    for (int64_t i = 1; i < W_zero_point_size; i++) {
      ORT_ENFORCE(W_zero_point_data[i] == W_zero_point_value,
                  "QLinearConv : zero point of per-channel filter must be same");
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

    auto X_scale_value = *(X_scale->template Data<float>());
    auto Y_scale_value = *(Y_scale->template Data<float>());

    std::vector<float> output_scales;
    const int64_t W_scale_size = W_scale->Shape().Size();
    const auto* W_scale_data = W_scale->template Data<float>();
    output_scales.resize(static_cast<size_t>(W_scale_size));
    for (int64_t i = 0; i < W_scale_size; i++) {
      output_scales[i] = (X_scale_value * W_scale_data[i] / Y_scale_value);
    }

    return output_scales;
  }

  static int32_t ComputeThreadCount(int64_t output_image_size, int64_t group_output_channels, int64_t kernel_dim) {
    // Replicate the logic from MlasGemmU8X8Schedule to control the number of
    // worker threads used for the convolution.
    int32_t maximum_thread_count;
    if (CPUIDInfo::GetCPUIDInfo().IsHybrid()) {
      maximum_thread_count = 64;
    } else {
      maximum_thread_count = 16;
    }
    constexpr double thread_complexity = static_cast<double>(64 * 1024);

    const double complexity = static_cast<double>(output_image_size) *
                              static_cast<double>(group_output_channels) *
                              static_cast<double>(kernel_dim);

    int32_t thread_count = maximum_thread_count;
    if (complexity < thread_complexity * maximum_thread_count) {
      thread_count = static_cast<int32_t>(complexity / thread_complexity) + 1;
    }
    if (thread_count > output_image_size) {
      // Ensure that every thread produces at least one output.
      thread_count = static_cast<int32_t>(output_image_size);
    }

    return thread_count;
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
    if (Info().TryGetConstantInput(InputTensors::IN_X_ZERO_POINT, &X_zero_point) && IsScalarOr1ElementVector(X_zero_point) &&
        Info().TryGetConstantInput(InputTensors::IN_W_ZERO_POINT, &W_zero_point) && IsValidQuantParam(W_zero_point, static_cast<int64_t>(output_channels))) {
      auto X_zero_point_value = *(X_zero_point->template Data<ActType>());
      const size_t W_zero_point_size = static_cast<size_t>(W_zero_point->Shape().Size());
      const auto* W_zero_point_data = W_zero_point->Data<int8_t>();
      if (std::all_of(W_zero_point_data, W_zero_point_data + W_zero_point_size, [](int8_t v) { return v == 0; })) {
        size_t packed_size = MlasConvSymPackWSize(group_count, group_input_channels, group_output_channels, kernel_size, std::is_signed<ActType>::value);
        if (packed_size != 0) {
          const Tensor* B = nullptr;
          Info().TryGetConstantInput(8, &B);
          const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;

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
  BufferUniquePtr packed_W_buffer_;
  size_t packed_W_size_{0};
  BufferUniquePtr reordered_W_buffer_;
  bool is_W_signed_{false};
  bool is_W_packed_{false};
  bool is_symmetric_conv_{false};
  bool channels_last_{false};
  std::vector<int32_t> column_sums_;
};

ONNX_CPU_OPERATOR_KERNEL(
    QLinearConv,
    10,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<uint8_t>);

#define REGISTER_QLINEARCONV_TYPED_KERNEL(domain, version, act_type, weight_type) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                  \
      QLinearConv,                                                                \
      domain,                                                                     \
      version,                                                                    \
      act_type##_##weight_type,                                                   \
      kCpuExecutionProvider,                                                      \
      KernelDefBuilder()                                                          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<act_type>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<weight_type>())       \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<act_type>())          \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),          \
      QLinearConv<act_type>);

REGISTER_QLINEARCONV_TYPED_KERNEL(kOnnxDomain, 10, int8_t, int8_t);

#ifndef DISABLE_CONTRIB_OPS

namespace contrib {

// Register an alternate version of this kernel that supports the channels_last
// attribute in order to consume and produce NHWC tensors.
ONNX_OPERATOR_KERNEL_EX(
    QLinearConv,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<uint8_t>);

REGISTER_QLINEARCONV_TYPED_KERNEL(kMSDomain, 1, int8_t, int8_t);

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
  if (group_input_channels != 1 && group_output_channels != 1) {
    packed_W_size_ = MlasGemmPackBSize(group_output_channels,
                                       kernel_dim,
                                       std::is_same<ActType, int8_t>::value,
                                       is_W_signed_);
    if (packed_W_size_ != 0) {
      size_t packed_W_data_size = SafeInt<size_t>(group_count) * packed_W_size_;
      auto* packed_W = static_cast<uint8_t*>(alloc->Alloc(packed_W_data_size));

      // Initialize memory to 0 as there could be some padding associated with pre-packed
      // buffer memory and we don not want it uninitialized and generate different hashes
      // if and when we try to cache this pre-packed buffer for sharing between sessions.
      memset(packed_W, 0, packed_W_data_size);

      packed_W_buffer_ = BufferUniquePtr(packed_W, BufferDeleter(alloc));

      // Allocate a temporary buffer to hold the reordered oihw->hwio filter for
      // a single group.
      //
      // Note: The size of this buffer is less than or equal to the size of the original
      // weight tensor, so the allocation size is guaranteed to fit inside size_t.
      auto* group_reordered_W = static_cast<uint8_t*>(alloc->Alloc(group_output_channels * group_input_channels * kernel_size));
      BufferUniquePtr group_reordered_W_buffer(group_reordered_W, BufferDeleter(alloc));

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
  auto* reordered_W = static_cast<uint8_t*>(alloc->Alloc(reordered_w_data_size));

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(reordered_W, 0, reordered_w_data_size);

  reordered_W_buffer_ = BufferUniquePtr(reordered_W, BufferDeleter(alloc));

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

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));

  const size_t kernel_rank = kernel_shape.size();

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
  }

  const int64_t C = X->Shape()[channels_last_ ? 1 + kernel_rank : 1];
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;
  const size_t spatial_dim_end = spatial_dim_start + kernel_rank;

  std::vector<int64_t> Y_dims({N});
  if (!channels_last_) {
    Y_dims.push_back(M);
  }
  TensorShape input_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
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

  const auto* Xdata = X->template Data<ActType>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<ActType>();

  BufferUniquePtr transpose_input_buffer;
  BufferUniquePtr transpose_output_buffer;

  // Allocate temporary buffers for transposing to channels last format.
  if (!channels_last_) {
    auto* transpose_input = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * X_offset);
    transpose_input_buffer = BufferUniquePtr(transpose_input, BufferDeleter(alloc));
    auto* transpose_output = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * Y_offset);
    transpose_output_buffer = BufferUniquePtr(transpose_output, BufferDeleter(alloc));
  }

  BufferUniquePtr col_buffer;
  BufferUniquePtr indirection_buffer;
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
      auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(ActType)) * group_col_buffer_size);
      col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
    }
  }
  if (use_indirection_buffer) {
    // Allocate indirection buffer pointers and prepare a padding vector for
    // the im2col transform.
    auto* indirection_data = alloc->Alloc(SafeInt<size_t>(sizeof(const ActType*)) * kernel_size * output_image_size);
    indirection_buffer = BufferUniquePtr(indirection_data, BufferDeleter(alloc));
    padding_data.resize(static_cast<size_t>(C), X_zero_point_value);
  }

  int32_t thread_count = ComputeThreadCount(output_image_size, group_output_channels, kernel_dim);
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  thread_count = std::min(thread_count, concurrency::ThreadPool::DegreeOfParallelism(thread_pool));

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
            static_cast<int64_t>(kernel_rank),
            static_cast<ActType*>(col_buffer.get()) + group_id * col_buffer_size,
            X_zero_point_value);
      }
    }

    auto conv_worker = [&](ptrdiff_t batch) {
      auto work = concurrency::ThreadPool::PartitionWork(batch, thread_count, static_cast<ptrdiff_t>(output_image_size));
      int64_t output_start = static_cast<int64_t>(work.start);
      int64_t output_count = static_cast<int64_t>(work.end) - work.start;

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
          MLAS_GEMM_QUANT_DATA_PARAMS gemm_params;
          gemm_params.ZeroPointA = static_cast<uint8_t>(X_zero_point_value);
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

          // Prepare the im2col transformation or use the input buffer directly for
          // pointwise convolutions.
          const auto* group_input_data = input_data + group_id * group_input_channels;
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
            gemm_params.A = reinterpret_cast<const uint8_t*>(worker_col_buffer);
            gemm_params.lda = static_cast<size_t>(kernel_dim);
          } else {
            gemm_params.A = reinterpret_cast<const uint8_t*>(group_input_data + output_start * C);
            gemm_params.lda = static_cast<size_t>(C);
          }

          MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
          gemm_shape.M = static_cast<size_t>(output_count);
          gemm_shape.N = static_cast<size_t>(group_output_channels);
          gemm_shape.K = static_cast<size_t>(kernel_dim);
          gemm_shape.AIsSigned = std::is_signed<ActType>::value;
          gemm_shape.BIsSigned = is_W_signed;

          MlasGemm(gemm_shape, gemm_params, nullptr);
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

    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, thread_count, conv_worker);

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
