// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/nn/conv.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
// Struct which encapsulates parameters for MKLDNN Conv2d primitive.
struct Conv2dParams {
  mkldnn::memory::dims& src_dims;
  mkldnn::memory::dims& filter_dims;
  mkldnn::memory::dims& bias_dims;
  mkldnn::memory::dims& dst_dims;
  mkldnn::memory::dims& strides;
  mkldnn::memory::dims& dilations;
  mkldnn::memory::dims& padding_left;
  mkldnn::memory::dims& padding_right;

  Conv2dParams(mkldnn::memory::dims& src_dims, mkldnn::memory::dims& filter_dims,
               mkldnn::memory::dims& bias_dims, mkldnn::memory::dims& dst_dims,
               mkldnn::memory::dims& strides, mkldnn::memory::dims& dilations,
               mkldnn::memory::dims& padding_left, mkldnn::memory::dims& padding_right)
      : src_dims(src_dims),
        filter_dims(filter_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        strides(strides),
        dilations(dilations),
        padding_left(padding_left),
        padding_right(padding_right) {}

  // Used as the key for Conv2d Primitive Reuse Pool.
  std::string ToString() const {
    std::string key;
    key.reserve(128);
    key.append("conv2d_");
    AddDimsToKey(key, src_dims);
    AddDimsToKey(key, filter_dims);
    AddDimsToKey(key, bias_dims);
    AddDimsToKey(key, dst_dims);
    AddDimsToKey(key, strides);
    AddDimsToKey(key, dilations);
    AddDimsToKey(key, padding_left);
    AddDimsToKey(key, padding_right);
    return key;
  }
};

template <typename T>
class Conv2dPrimitive : public PrimitiveBase {
 public:
  explicit Conv2dPrimitive(const Conv2dParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.conv_fwd == nullptr) {
      Initialize(params);
    }
  }

  ~Conv2dPrimitive() = default;

  void Compute(const T* src_data, const T* filter_data,
               const T* dst_data, const T* bias_data = nullptr) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(filter_data)));
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(
          static_cast<void*>(const_cast<T*>(bias_data)));
    }
    context_.dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(dst_data)));
    context_.stream->submit(context_.net);

    context_.src_mem->set_data_handle(nullptr);
    context_.filter_mem->set_data_handle(nullptr);
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(nullptr);
    }
    context_.dst_mem->set_data_handle(nullptr);
    return;
  }

  mkldnn::memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }

  mkldnn::memory::format GetFilterMemoryFormat() const { return context_.filter_fmt; }

  mkldnn::memory::format GetDstMemoryFormat() const { return context_.dst_fmt; }

  size_t GetSrcSize() const { return context_.src_size; }

  size_t GetFilterSize() const { return context_.filter_size; }

  size_t GetDstSize() const { return context_.dst_size; }

  mkldnn::convolution_forward::primitive_desc* GetPrimitiveDesc() const {
    return context_.conv_fwd_pd.get();
  }

 private:
  struct Conv2dContext {
    mkldnn::memory::format src_fmt;
    mkldnn::memory::format filter_fmt;
    mkldnn::memory::format dst_fmt;

    size_t src_size;
    size_t filter_size;
    size_t dst_size;

    std::unique_ptr<mkldnn::memory> src_mem;
    std::unique_ptr<mkldnn::memory> filter_mem;
    std::unique_ptr<mkldnn::memory> bias_mem;
    std::unique_ptr<mkldnn::memory> dst_mem;

    std::unique_ptr<mkldnn::convolution_forward::desc> fwd_desc;

    std::unique_ptr<mkldnn::memory::desc> src_md;
    std::unique_ptr<mkldnn::memory::desc> filter_md;
    std::unique_ptr<mkldnn::memory::desc> bias_md;
    std::unique_ptr<mkldnn::memory::desc> dst_md;

    std::unique_ptr<mkldnn::convolution_forward::primitive_desc> conv_fwd_pd;
    std::unique_ptr<mkldnn::primitive> conv_fwd;

    std::unique_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;

    Conv2dContext()
        : src_fmt(mkldnn::memory::format::any),
          filter_fmt(mkldnn::memory::format::any),
          dst_fmt(mkldnn::memory::format::any),
          src_size(0),
          filter_size(0),
          dst_size(0),
          src_mem(nullptr),
          filter_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          src_md(nullptr),
          filter_md(nullptr),
          bias_md(nullptr),
          conv_fwd_pd(nullptr),
          conv_fwd(nullptr),
          stream(nullptr) {}
  };

  void Initialize(const Conv2dParams& params) {
    // Set the memory descriptors to format::any to allow MKLDNN to decide what the optimal memory layout should be
    // for the computation given the input params.
    context_.src_md.reset(new mkldnn::memory::desc(
        {params.src_dims}, MklDnnType<T>(), mkldnn::memory::format::any));
    context_.filter_md.reset(new mkldnn::memory::desc(
        {params.filter_dims}, MklDnnType<T>(), mkldnn::memory::format::any));
    context_.dst_md.reset(new mkldnn::memory::desc(
        {params.dst_dims}, MklDnnType<T>(), mkldnn::memory::format::any));
    if (!params.bias_dims.empty())
      context_.bias_md.reset(new mkldnn::memory::desc(
          {params.bias_dims}, MklDnnType<T>(), mkldnn::memory::format::any));

    if (!params.bias_dims.empty()) {
      context_.fwd_desc.reset(new mkldnn::convolution_forward::desc(
          mkldnn::prop_kind::forward, mkldnn::convolution_direct, *context_.src_md,
          *context_.filter_md, *context_.bias_md, *context_.dst_md,
          params.strides, params.dilations, params.padding_left,
          params.padding_right, mkldnn::padding_kind::zero));
    } else {
      context_.fwd_desc.reset(new mkldnn::convolution_forward::desc(
          mkldnn::prop_kind::forward, mkldnn::convolution_direct, *context_.src_md,
          *context_.filter_md, *context_.dst_md, params.strides,
          params.dilations, params.padding_left,
          params.padding_right, mkldnn::padding_kind::zero));
    }

    context_.conv_fwd_pd.reset(new mkldnn::convolution_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    context_.src_fmt = static_cast<mkldnn::memory::format>(
        context_.conv_fwd_pd.get()->src_primitive_desc().desc().data.format);

    context_.filter_fmt = static_cast<mkldnn::memory::format>(
        context_.conv_fwd_pd.get()->weights_primitive_desc().desc().data.format);

    context_.dst_fmt = static_cast<mkldnn::memory::format>(
        context_.conv_fwd_pd.get()->dst_primitive_desc().desc().data.format);

    context_.src_size = context_.conv_fwd_pd.get()->src_primitive_desc().get_size();

    context_.filter_size = context_.conv_fwd_pd.get()->weights_primitive_desc().get_size();

    context_.dst_size = context_.conv_fwd_pd.get()->dst_primitive_desc().get_size();

    context_.src_mem.reset(
        new mkldnn::memory(context_.conv_fwd_pd.get()->src_primitive_desc(), nullptr));
    context_.filter_mem.reset(
        new mkldnn::memory(context_.conv_fwd_pd.get()->weights_primitive_desc(), nullptr));
    context_.dst_mem.reset(
        new mkldnn::memory(context_.conv_fwd_pd.get()->dst_primitive_desc(), nullptr));

    if (!params.bias_dims.empty()) {
      context_.bias_mem.reset(
          new mkldnn::memory(context_.conv_fwd_pd.get()->bias_primitive_desc(), nullptr));

      context_.conv_fwd.reset(new mkldnn::convolution_forward(
          *context_.conv_fwd_pd, *context_.src_mem, *context_.filter_mem,
          *context_.bias_mem, *context_.dst_mem));
    } else {
      context_.conv_fwd.reset(
          new mkldnn::convolution_forward(*context_.conv_fwd_pd, *context_.src_mem,
                                          *context_.filter_mem, *context_.dst_mem));
    }

    context_.net.push_back(*context_.conv_fwd);
  }

  Conv2dContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN Conv2d primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T>
class Conv2dPrimitivePool : public PrimitivePool<T> {
 public:
  static Conv2dPrimitive<T>* Get(const Conv2dParams& params) {
    Conv2dPrimitive<T>* primitive = dynamic_cast<Conv2dPrimitive<T>*>(
        Conv2dPrimitivePool<T>::GetInstance().GetPrimitive(params.ToString()));

    if (primitive == nullptr) {
      auto conv2d_primitive = std::make_unique<Conv2dPrimitive<T>>(params);
      primitive = conv2d_primitive.get();
      Conv2dPrimitivePool<T>::GetInstance().SetPrimitive(params.ToString(), std::move(conv2d_primitive));
    }
    return primitive;
  }

 private:
  Conv2dPrimitivePool() = default;
  ~Conv2dPrimitivePool() = default;

  static Conv2dPrimitivePool& GetInstance() {
    static Conv2dPrimitivePool pool;
    return pool;
  }
};
}  // namespace

template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;
  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];
  const int group_mkl = static_cast<int>(onnxruntime::ConvBase::group_);

  ONNXRUNTIME_RETURN_IF_ERROR(onnxruntime::ConvBase::ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape = onnxruntime::ConvBase::ComputeKernelShape(W->Shape());

  // TODO: Support more than 2d kernels
  if (kernel_shape.size() != 2) {
    // Fall Back to CPU implementation.
    return onnxruntime::Conv<T>::Compute(context);
  }

  if (kernel_shape.size() + 2 != W->Shape().NumDimensions()) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                                   " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                   " W: ", W->Shape().ToString().c_str());
  }

  for (size_t i = 0; i < kernel_shape.size(); ++i) {
    if (kernel_shape[i] != W->Shape()[i + 2]) {
      return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                                     " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                     " W: ", W->Shape().ToString().c_str());
    }
  }

  std::vector<int64_t> pads(onnxruntime::ConvBase::pads_);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(onnxruntime::ConvBase::dilations_);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(onnxruntime::ConvBase::strides_);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ONNXRUNTIME_RETURN_IF_ERROR(onnxruntime::ConvBase::InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  mkldnn::memory::dims src_dims_mkl(X->Shape().GetDims().begin(), X->Shape().GetDims().end());
  mkldnn::memory::dims filter_dims_mkl;
  if (group_mkl == 1) {
    filter_dims_mkl.assign(W->Shape().GetDims().begin(), W->Shape().GetDims().end());
  } else {
    filter_dims_mkl.assign({
        group_mkl,
        static_cast<int>(W->Shape()[0] / group_mkl),
        static_cast<int>(W->Shape()[1]),
        static_cast<int>(W->Shape()[2]),
        static_cast<int>(W->Shape()[3]),
    });
  }
  mkldnn::memory::dims strides_mkl(strides.begin(), strides.end());
  mkldnn::memory::dims dilations_mkl(dilations.begin(), dilations.end());
  // mkldnn dilations start from 0 so we need to subtract 1 from each dim.
  dilations_mkl[0] -= 1;
  dilations_mkl[1] -= 1;
  mkldnn::memory::dims padding_left_mkl(pads.begin(), pads.begin() + 2);
  mkldnn::memory::dims padding_right_mkl(pads.begin() + 2, pads.end());
  mkldnn::memory::dims dst_dims_mkl(Y_dims.begin(), Y_dims.end());
  mkldnn::memory::dims bias_dims_mkl;
  if (B != nullptr) {
    bias_dims_mkl.assign(B->Shape().GetDims().begin(), B->Shape().GetDims().end());
  }

  AllocatorPtr alloc;
  ONNXRUNTIME_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  IAllocatorUniquePtr<void> src_reorder_buffer;
  IAllocatorUniquePtr<void> filter_reorder_buffer;
  IAllocatorUniquePtr<void> dst_reorder_buffer;

  const T* src_data = X->template Data<T>();
  const T* filter_data = W->template Data<T>();
  T* dst_data = Y->template MutableData<T>();
  const T* bias_data = nullptr;
  if (B != nullptr) {
    bias_data = B->template Data<T>();
  }

  try {
    Conv2dParams conv2d_params(src_dims_mkl, filter_dims_mkl, bias_dims_mkl,
                               dst_dims_mkl, strides_mkl, dilations_mkl,
                               padding_left_mkl, padding_right_mkl);
    Conv2dPrimitive<T>* conv2d_primitive = Conv2dPrimitivePool<T>::Get(conv2d_params);
    auto conv_fwd_pd = conv2d_primitive->GetPrimitiveDesc();

    mkldnn::engine& cpu_engine = GetEngine();

    // Per ONNX spec,
    // X (src) is NCHW, W (filter) is OIHW/GOIHW, and Y (dst) is NCHW
    auto src_md = mkldnn::memory::desc(src_dims_mkl, MklDnnType<T>(), mkldnn::memory::format::nchw);
    auto filter_format = group_mkl == 1 ? mkldnn::memory::format::oihw : mkldnn::memory::format::goihw;
    auto dst_md = mkldnn::memory::desc(dst_dims_mkl, MklDnnType<T>(), mkldnn::memory::format::nchw);

    // Reorder src memory layout if necessary.
    if (src_md.data.format != conv2d_primitive->GetSrcMemoryFormat()) {
      auto pd = mkldnn::memory::primitive_desc(src_md, cpu_engine);
      mkldnn::memory src = mkldnn::memory(pd, (void*)src_data);
      // allocate the size queried from memory primitive desc. it may not match tensor logical size due to
      // mkldnn using padding to allow use of blocked format.
      src_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc, conv2d_primitive->GetSrcSize());
      mkldnn::memory dst = mkldnn::memory(conv_fwd_pd->src_primitive_desc(), src_reorder_buffer.get());
      MemoryReorderParams params(src, dst);
      DoReorder<T>(params);
      src_data = static_cast<T*>(dst.get_data_handle());
    }

    // Reorder filter memory layout if necessary.
    if (filter_format != conv2d_primitive->GetFilterMemoryFormat()) {
      auto pd = mkldnn::memory::primitive_desc(mkldnn::memory::desc(filter_dims_mkl,
                                                                    MklDnnType<T>(),
                                                                    filter_format),
                                               cpu_engine);
      mkldnn::memory src = mkldnn::memory(pd, (void*)filter_data);
      // allocate the size queried from memory primitive desc. it may not match tensor logical size due to
      // mkldnn using padding to allow use of blocked format.
      filter_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc, conv2d_primitive->GetFilterSize());
      mkldnn::memory dst = mkldnn::memory(conv_fwd_pd->weights_primitive_desc(), filter_reorder_buffer.get());
      MemoryReorderParams params(src, dst);
      DoReorder<T>(params);
      filter_data = static_cast<T*>(dst.get_data_handle());
    }

    // Allocate dst buffer if reorder is necessary
    if (dst_md.data.format != conv2d_primitive->GetDstMemoryFormat()) {
      // allocate the size queried from memory primitive desc. it may not match tensor logical size due to
      // mkldnn using padding to allow use of blocked format.
      dst_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc, conv2d_primitive->GetDstSize());
      dst_data = static_cast<T*>(dst_reorder_buffer.get());
    }

    conv2d_primitive->Compute(src_data, filter_data, dst_data, bias_data);

    // Reorder dst memory layout if necessary
    if (dst_md.data.format != conv2d_primitive->GetDstMemoryFormat()) {
      mkldnn::memory src = mkldnn::memory(conv_fwd_pd->dst_primitive_desc(), (void*)dst_data);
      auto pd = mkldnn::memory::primitive_desc(dst_md, cpu_engine);
      mkldnn::memory dst = mkldnn::memory(pd, Y->template MutableData<T>());
      MemoryReorderParams params(src, dst);
      DoReorder<T>(params);
    }

  } catch (mkldnn::error& e) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Status: ", e.status, ", message: ", e.message.c_str());
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    kMklDnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace mkl_dnn
}  // namespace onnxruntime
