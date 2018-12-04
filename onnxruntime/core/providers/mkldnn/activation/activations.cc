// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/activation/activations.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
// Struct which encapsulates parameters for MKLDNN Pool primitive.
struct ReluParams {
  mkldnn::memory::dims& src_dims;
  mkldnn::memory::dims& dst_dims;
  size_t num_dimensions;

  ReluParams(mkldnn::memory::dims& src_dims, mkldnn::memory::dims& dst_dims, 
		size_t dimensions = 0)
      : src_dims(src_dims),
        dst_dims(dst_dims),
        num_dimensions(dimensions) {}

  // Used as the key for Pool Primitive Reuse Pool.
  std::string ToString() const {
    std::string key;
    key.reserve(64);
    key.append("Relu_");
    AddDimsToKey(key, src_dims);
    AddDimsToKey(key, dst_dims);
    return key;
  }
};

template <typename T>
class ReluPrimitive final : public PrimitiveBase {
 public:
  explicit ReluPrimitive(const ReluParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.relu_fwd == nullptr) {
      Initialize(params);
    }
  }

  ~ReluPrimitive() = default;

  void Compute(const T* src_data, const T* dst_data) {
	  context_.src_mem->set_data_handle(
		  static_cast<void*>(const_cast<T*>(src_data)));
	  context_.dst_mem->set_data_handle(
		  static_cast<void*>(const_cast<T*>(dst_data)));
	  context_.stream->submit(context_.net);

	  context_.src_mem->set_data_handle(nullptr);
	  context_.dst_mem->set_data_handle(nullptr);
	  return;
  }

  std::unique_ptr<mkldnn::memory::desc> 
	GetDstMemoryDesc() const { return context_.dst_md; }

  std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> 
	GetPrimitiveDesc() const {
		return context_.relu_fwd_pd;
  }

 private:
  struct ReluContext {
    mkldnn::memory::format src_fmt;
    mkldnn::memory::format dst_fmt;

    std::unique_ptr<mkldnn::memory> src_mem;
    std::unique_ptr<mkldnn::memory> dst_mem;

    std::unique_ptr<mkldnn::eltwise_forward::desc> fwd_desc;
    std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> relu_fwd_pd;
    std::unique_ptr<mkldnn::primitive> relu_fwd;

    std::unique_ptr<mkldnn::memory::desc> src_md;
    std::unique_ptr<mkldnn::memory::desc> dst_md;

    std::unique_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;

    ReluContext()
    : src_fmt(mkldnn::memory::format::any),
      dst_fmt(mkldnn::memory::format::any),
      src_mem(nullptr),
      dst_mem(nullptr),
      fwd_desc(nullptr),
      relu_fwd_pd(nullptr),
      relu_fwd(nullptr),
      src_md(nullptr),
      dst_md(nullptr),
      stream(nullptr) {}
  };

  void Initialize(const ReluParams& params) {
    
    mkldnn::memory::format fmt = mkldnn::memory::format::any;
    switch (params.num_dimensions) {
    case 1: { fmt = mkldnn::memory::format::x; break; }
    case 2: { fmt = mkldnn::memory::format::nc; break; }
    case 3: { fmt = mkldnn::memory::format::ntc; break; }
    case 4: { fmt = mkldnn::memory::format::nchw; break; }
    case 5: { fmt = mkldnn::memory::format::ncdhw; break; }
    default: {  fmt = mkldnn::memory::format::any; break; }
    }

    context_.src_md.reset(new mkldnn::memory::desc(
      { params.src_dims }, MklDnnType<T>(), fmt));
    context_.dst_md.reset(new mkldnn::memory::desc(
      { params.dst_dims }, MklDnnType<T>(), fmt));

    context_.fwd_desc.reset(new mkldnn::eltwise_forward::desc(
      mkldnn::prop_kind::forward_inference, mkldnn::algorithm::eltwise_relu, 
      *context_.src_md, 0, 0));

    context_.relu_fwd_pd.reset(
      new mkldnn::eltwise_forward::primitive_desc(*context_.fwd_desc,
      cpu_engine_));

    context_.src_fmt = static_cast<mkldnn::memory::format>(
      context_.relu_fwd_pd.get()->dst_primitive_desc().desc().data.format);

    context_.dst_fmt = static_cast<mkldnn::memory::format>(
      context_.relu_fwd_pd.get()->dst_primitive_desc().desc().data.format);

    context_.src_mem.reset(
     new mkldnn::memory(context_.relu_fwd_pd.get()->dst_primitive_desc(), 
			  nullptr));
    context_.dst_mem.reset(
      new mkldnn::memory(context_.relu_fwd_pd.get()->dst_primitive_desc(), 
			nullptr));
    context_.relu_fwd.reset(
      new mkldnn::eltwise_forward(*context_.relu_fwd_pd, *context_.src_mem, 
			*context_.dst_mem));
    context_.net.push_back(*context_.relu_fwd);
  }

  ReluContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN Relu primitives which are expensive 
// to instantiate. To address thread safety, the primitives are stored in a map 
// on thread local storage.
template <typename T>
class ReluPrimitivePool : public PrimitivePool<T> {
 public:
  static ReluPrimitive<T>* Get(const ReluParams& params) {
    ReluPrimitive<T>* primitive = dynamic_cast<ReluPrimitive<T>*>(
        ReluPrimitivePool<T>::GetInstance().GetPrimitive(params.ToString()));

    if (primitive == nullptr) {
      auto relu_primitive = std::make_unique<ReluPrimitive<T>>(params);
      primitive = relu_primitive.get();
      ReluPrimitivePool<T>::GetInstance().SetPrimitive(params.ToString(), 
		std::move(relu_primitive));
    }
    return primitive;
  }

 private:
  ReluPrimitivePool() = default;
  ~ReluPrimitivePool() = default;

  static ReluPrimitivePool& GetInstance() {
    static ReluPrimitivePool pool;
    return pool;
  }
};
}	// namespace

template <typename T>
Status Relu<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());
  
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  if (X->Shape().NumDimensions() > 5 ) {
    return onnxruntime::Relu<T>::Compute(context);
  }

  const TensorShape& y_shape = Y->Shape();
  auto& y_dims = y_shape.GetDims();

  const T* src_data = X->template Data<T>();
  T* dst_data = Y->template MutableData<T>();

  mkldnn::memory::dims src_dims_mkl(x_dims.begin(), x_dims.end());
  mkldnn::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());

  try {
    ReluParams pool_params(src_dims_mkl, dst_dims_mkl, x_shape.NumDimensions());
    ReluPrimitive<T>* relulPrimitive = ReluPrimitivePool<T>::Get(pool_params);

    relulPrimitive->Compute(src_data, dst_data);
  } catch (const mkldnn::error& e) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Status: ", e.status, 
		", message: ", e.message.c_str());
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,
    kMklDnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Relu<float>);

}  // namespace mkl_dnn
}  // namespace onnxruntime
