// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/nn/batch_norm.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"
#include "core/providers/mkldnn/memcpy_s.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
// Struct which encapsulates parameters for MKLDNN BatchNorm primitive.
struct BatchNormParams {
  const mkldnn::memory::dims& src_dims;
  const mkldnn::memory::dims& scale_dims;
  const mkldnn::memory::dims& b_dims;
  const mkldnn::memory::dims& mean_dims;
  const mkldnn::memory::dims& var_dims;
  const mkldnn::memory::dims& dst_dims;
  const float epsilon;

  BatchNormParams(const mkldnn::memory::dims& src_dims_mkl,
    const mkldnn::memory::dims& scale_dims_mkl,
    const mkldnn::memory::dims& b_dims_mkl, const mkldnn::memory::dims& mean_dims_mkl,
    const mkldnn::memory::dims& var_dims_mkl, const mkldnn::memory::dims& dst_dims_mkl,
    float eps)
    : src_dims(src_dims_mkl),
    scale_dims(scale_dims_mkl),
    b_dims(b_dims_mkl),
    mean_dims(mean_dims_mkl),
    var_dims(var_dims_mkl),
    dst_dims(dst_dims_mkl),
    epsilon(eps) {}

  // Used as the key for BatchNorm Primitive Reuse Pool.
  std::string ToString() const {
    std::string key;
    key.reserve(128);
    key.append("BatchNorm_");
    AddDimsToKey(key, src_dims);
    AddDimsToKey(key, scale_dims);
    AddDimsToKey(key, b_dims);
    AddDimsToKey(key, mean_dims);
    AddDimsToKey(key, var_dims);
    AddDimsToKey(key, dst_dims);
    return key;
  }
};

template <typename T>
class BatchNormPrimitive final : public PrimitiveBase {
 public:
  explicit BatchNormPrimitive(const BatchNormParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.batchnorm_fwd == nullptr) {
      Initialize(params);
    }
  }

  ~BatchNormPrimitive() = default;

  void Compute(const T* src_data, const T* scale_data, const T* b_data, 
    const T* mean_data, const T* var_data, T* dst_data, 
    int scale_dims_channels) {
    context_.src_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(src_data)));
    context_.mean_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(mean_data)));
    context_.var_mem->set_data_handle(
      static_cast<void*>(const_cast<T*>(var_data)));
    context_.dst_mem->set_data_handle(
      static_cast<void*>(dst_data));

    T* scale_shift_buf = static_cast<T*>(context_.scale_shift_mem->get_data_handle());

    size_t src_bytes = sizeof(T) * scale_dims_channels;
    size_t dst_bytes = sizeof(T) * scale_dims_channels;

    MEMCPY_S(scale_shift_buf, scale_data, src_bytes, dst_bytes);
    MEMCPY_S(&scale_shift_buf[scale_dims_channels], b_data, src_bytes, dst_bytes);
    context_.stream->submit(context_.net);
    return;
  }

  std::unique_ptr<mkldnn::convolution_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.conv_fwd_pd;
  }

 private:
  struct BatchNormContext {
    std::unique_ptr<mkldnn::memory> src_mem;
    std::unique_ptr<mkldnn::memory> scale_shift_mem;
    std::unique_ptr<mkldnn::memory> mean_mem;
    std::unique_ptr<mkldnn::memory> var_mem;
    std::unique_ptr<mkldnn::memory> dst_mem;

    std::unique_ptr<mkldnn::memory::desc> src_md;
    std::unique_ptr<mkldnn::memory::desc> scale_shift_md;
    std::unique_ptr<mkldnn::memory::desc> mean_md;
    std::unique_ptr<mkldnn::memory::desc> var_md;
    std::unique_ptr<mkldnn::memory::desc> dst_md;

    std::unique_ptr<mkldnn::batch_normalization_forward::desc> batchnorm_fwd;
    std::unique_ptr<mkldnn::batch_normalization_forward::primitive_desc> 
      batchnorm_fwd_pd;

    std::unique_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;
  };

  void Initialize(const BatchNormParams& params) {
    mkldnn::memory::format fmt = mkldnn::memory::format::any;
    switch (params.src_dims.size()) {
    case 1: { fmt = mkldnn::memory::format::x; break; }
    case 2: { fmt = mkldnn::memory::format::nc; break; }
    case 3: { fmt = mkldnn::memory::format::ntc; break; }
    case 4: { fmt = mkldnn::memory::format::nchw; break; }
    case 5: { fmt = mkldnn::memory::format::ncdhw; break; }
    default: {  fmt = mkldnn::memory::format::any; break; }
    }
    context_.src_md.reset(new mkldnn::memory::desc(
      { params.src_dims }, MklDnnType<T>(), fmt));

    context_.scale_shift_md.reset(new mkldnn::memory::desc(
      { 2, params.scale_dims[0] }, MklDnnType<T>(), mkldnn::memory::format::nc));

    context_.mean_md.reset(new mkldnn::memory::desc(
      { params.mean_dims }, MklDnnType<T>(), mkldnn::memory::format::x));
    context_.var_md.reset(new mkldnn::memory::desc(
      { params.var_dims }, MklDnnType<T>(), mkldnn::memory::format::x));
    context_.dst_md.reset(new mkldnn::memory::desc(
      { params.dst_dims }, MklDnnType<T>(), fmt));

    context_.src_mem.reset(
      new mkldnn::memory({ *context_.src_md, cpu_engine_ }, nullptr));
   
   // scale_shift_mem will allocate 2*C*sizeof(float) buffer
   //
    context_.scale_shift_mem.reset(
      new mkldnn::memory({ *context_.scale_shift_md, cpu_engine_ }));

    context_.mean_mem.reset(
      new mkldnn::memory({ *context_.mean_md, cpu_engine_ }, nullptr));
    context_.var_mem.reset(
      new mkldnn::memory({ *context_.var_md, cpu_engine_ }, nullptr));

    context_.batchnorm_fwd.reset(new mkldnn::batch_normalization_forward::desc(
      mkldnn::prop_kind::forward_inference, *context_.src_md, params.epsilon,
      mkldnn::batch_normalization_flag::use_scale_shift |
      mkldnn::batch_normalization_flag::use_global_stats));

   context_.batchnorm_fwd_pd.reset(
     new mkldnn::batch_normalization_forward::primitive_desc(
       *context_.batchnorm_fwd, cpu_engine_));

   context_.dst_mem.reset(
     new mkldnn::memory(
       context_.batchnorm_fwd_pd->dst_primitive_desc(), nullptr));

   auto bn = mkldnn::batch_normalization_forward(
     *context_.batchnorm_fwd_pd,
     (const mkldnn::primitive::at)*context_.src_mem,
     (const mkldnn::primitive::at)*context_.mean_mem,
     (const mkldnn::primitive::at)*context_.var_mem,
     (const mkldnn::memory)*context_.scale_shift_mem,
     (const mkldnn::memory) *context_.dst_mem);
   
   context_.net.push_back(bn);
  }

  BatchNormContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN BatchNorm primitives which are 
// expensive to instantiate. To address thread safety, the primitives are
// stored in a map on thread local storage.
template <typename T>
class BatchNormPrimitivePool : public PrimitivePool<T> {
 public:
  static BatchNormPrimitive<T>* Get(const BatchNormParams& params) {
    BatchNormPrimitive<T>* primitive = 
      dynamic_cast<BatchNormPrimitive<T>*>(
        BatchNormPrimitivePool<T>::GetInstance().GetPrimitive(params.ToString()));

    if (primitive == nullptr) {
      auto BatchNorm_primitive = onnxruntime::make_unique<BatchNormPrimitive<T>>(params);
      primitive = BatchNorm_primitive.get();
      BatchNormPrimitivePool<T>::GetInstance().SetPrimitive(
        params.ToString(), std::move(BatchNorm_primitive));
    }
    return primitive;
  }

 private:
  BatchNormPrimitivePool() = default;
  ~BatchNormPrimitivePool() = default;

  static BatchNormPrimitivePool& GetInstance() {
    static BatchNormPrimitivePool pool;
    return pool;
  }
};
} // namespace

template <typename T>
Status BatchNorm<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  
  int num_dimensions = static_cast<int>(X->Shape().NumDimensions());
  if (num_dimensions == 3) {
    // Fall back CPU implementation
    return onnxruntime::BatchNorm<T>::Compute(context);
  }

  const T* src_data = X->template Data<T>();

  const Tensor* scale = context->Input<Tensor>(1);
  const T* scale_data = scale->template Data<T>();

  const Tensor* B = context->Input<Tensor>(2);
  const T* b_data = B->template Data<T>();

  const Tensor* mean = context->Input<Tensor>(3);
  const T* mean_data = mean->template Data<T>();

  const Tensor* var = context->Input<Tensor>(4);
  const T* var_data = var->template Data<T>();

  Tensor* Y = context->Output(0, X->Shape());
  T* dst_data = Y->template MutableData<T>();

  ORT_RETURN_IF_ERROR(
    BatchNormHelper::ValidateInputs(X, scale, B, mean, var));

  mkldnn::memory::dims src_dims_mkl(
    X->Shape().GetDims().begin(), X->Shape().GetDims().end());
  mkldnn::memory::dims scale_dims_mkl(
    scale->Shape().GetDims().begin(), scale->Shape().GetDims().end());
  mkldnn::memory::dims b_dims_mkl(
    B->Shape().GetDims().begin(), B->Shape().GetDims().end());
  mkldnn::memory::dims mean_dims_mkl(
    mean->Shape().GetDims().begin(), mean->Shape().GetDims().end());
  mkldnn::memory::dims var_dims_mkl(
    var->Shape().GetDims().begin(), var->Shape().GetDims().end());

  mkldnn::memory::dims dst_dims_mkl(
    Y->Shape().GetDims().begin(), Y->Shape().GetDims().end());

  try {
    BatchNormParams batchNorm_params(src_dims_mkl, scale_dims_mkl, 
      b_dims_mkl, mean_dims_mkl, var_dims_mkl, dst_dims_mkl, 
      onnxruntime::BatchNorm<T>::epsilon_);
    BatchNormPrimitive<T>* batchNorm_primitive = 
      BatchNormPrimitivePool<T>::Get(batchNorm_params);
    ORT_RETURN_IF_NOT(batchNorm_primitive != nullptr);
    batchNorm_primitive->Compute(src_data, scale_data, b_data, 
      mean_data, var_data, dst_data, scale_dims_mkl[0]);

  } catch (const mkldnn::error& e) {
    return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, "Status: ", e.status, ", message: ", e.message.c_str());
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    BatchNormalization,
    kOnnxDomain,
    7,
    kMklDnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BatchNorm<float>);

}  // namespace mkl_dnn
}  // namespace onnxruntime
