// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/nn/lrn.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"

namespace onnxruntime {
namespace mkl_dnn {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LRN,
    kOnnxDomain,
    1,
    float,
    kMklDnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LRN<float>);

namespace {
// Struct which encapsulates parameters for MKLDNN LRN primitive.
struct LRNParams {
  const mkldnn::memory::dims& dims_;
  float alpha_;
  float beta_;
  float bias_;
  int size_;

  LRNParams(const mkldnn::memory::dims& dims, float alpha, float beta, float bias, int size)
      : dims_(dims), alpha_(alpha), beta_(beta), bias_(bias), size_(size) {}

  // Used as the key for LRN Primitive Reuse LRN.
  std::string ToString() const {
    std::string key;
    key.reserve(128);
    key.append("lrn");
    AddDimsToKey(key, dims_);
    key.append('#' + std::to_string(alpha_) + '#');
    key.append('#' + std::to_string(beta_) + '#');
    key.append('#' + std::to_string(bias_) + '#');
    key.append('#' + std::to_string(size_) + '#');

    return key;
  }
};

template <typename T>
class LRNPrimitive : public PrimitiveBase {
 public:
  explicit LRNPrimitive(const LRNParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.lrn_fwd == nullptr) {
      Initialize(params);
    }
  }

  ~LRNPrimitive() = default;

  void Compute(const T* src_data, T* dst_data) {
    context_.src_mem->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
    context_.stream->submit(context_.net);

    context_.src_mem->set_data_handle(nullptr);
    context_.dst_mem->set_data_handle(nullptr);
    return;
  }

  mkldnn::memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }

  size_t GetSrcSize() const { return context_.src_size; }

  size_t GetDstSize() const { return context_.dst_size; }

  mkldnn::lrn_forward::primitive_desc* GetPrimitiveDesc() const {
    return context_.fwd_primitive_desc.get();
  }

 private:
  struct LRNContext {
    mkldnn::memory::format src_fmt;
    std::unique_ptr<mkldnn::memory::desc> src_md;

    size_t src_size;
    size_t dst_size;

    std::unique_ptr<mkldnn::memory> src_mem;
    std::unique_ptr<mkldnn::memory> dst_mem;

    std::unique_ptr<mkldnn::lrn_forward::desc> fwd_desc;
    std::unique_ptr<mkldnn::lrn_forward::primitive_desc> fwd_primitive_desc;
    std::unique_ptr<mkldnn::primitive> lrn_fwd;

    std::unique_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;

    LRNContext()
        : src_fmt(mkldnn::memory::format::any),
          src_md(nullptr),
          src_size(0),
          dst_size(0),
          src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_primitive_desc(nullptr),
          lrn_fwd(nullptr),
          stream(nullptr) {}
  };

  void Initialize(const LRNParams& params) {
    context_.src_md.reset(new mkldnn::memory::desc({params.dims_}, MklDnnType<T>(), mkldnn::memory::format::nchw));

    mkldnn::algorithm algo = mkldnn::algorithm::lrn_across_channels;
    context_.fwd_desc.reset(new mkldnn::lrn_forward::desc(
        mkldnn::prop_kind::forward_scoring, algo, *context_.src_md,
        params.size_, params.alpha_, params.beta_, params.bias_));

    context_.fwd_primitive_desc.reset(new mkldnn::lrn_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    context_.src_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_primitive_desc.get()->src_primitive_desc().desc().data.format);

    context_.src_size = context_.fwd_primitive_desc.get()->src_primitive_desc().get_size();
    context_.dst_size = context_.fwd_primitive_desc.get()->dst_primitive_desc().get_size();

    context_.src_mem.reset(new mkldnn::memory(context_.fwd_primitive_desc.get()->src_primitive_desc(), nullptr));
    context_.dst_mem.reset(new mkldnn::memory(context_.fwd_primitive_desc.get()->dst_primitive_desc(), nullptr));
    context_.lrn_fwd.reset(
        new mkldnn::lrn_forward(*context_.fwd_primitive_desc, *context_.src_mem, *context_.dst_mem));
    context_.net.push_back(*context_.lrn_fwd);
  }

  LRNContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN Pool primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T>
class LRNPrimitivePool : public PrimitivePool<T> {
 public:
  static LRNPrimitive<T>* Get(const LRNParams& params) {
    LRNPrimitive<T>* primitive = dynamic_cast<LRNPrimitive<T>*>(
        LRNPrimitivePool<T>::GetInstance().GetPrimitive(params.ToString()));
    if (primitive == nullptr) {
      auto pool_primitive = onnxruntime::make_unique<LRNPrimitive<T>>(params);
      primitive = pool_primitive.get();
      LRNPrimitivePool<T>::GetInstance().SetPrimitive(params.ToString(), std::move(pool_primitive));
    }
    return primitive;
  }

 private:
  LRNPrimitivePool() = default;
  ~LRNPrimitivePool() = default;

  static LRNPrimitivePool& GetInstance() {
    static LRNPrimitivePool pool;
    return pool;
  }
};
}  // namespace

template <typename T>
Status LRN<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const T* src_data = X->template Data<T>();

  const TensorShape& x_shape = X->Shape();
  if (x_shape.NumDimensions() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Support NCHW image only.");
  }

  const auto& x_dims = x_shape.GetDims();
  mkldnn::memory::dims dims_mkl(x_dims.begin(), x_dims.end());

  Tensor* Y = context->Output(0, TensorShape(x_dims));
  T* dst_data = Y->template MutableData<T>();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  IAllocatorUniquePtr<void> src_reorder_buffer;
  IAllocatorUniquePtr<void> dst_reorder_buffer;

  try {
    LRNParams lrn_params(dims_mkl, this->alpha_, this->beta_, this->bias_, this->size_);
    LRNPrimitive<T>* lrn_primitive = LRNPrimitivePool<T>::Get(lrn_params);
    auto fwd_primitive_desc = lrn_primitive->GetPrimitiveDesc();

    mkldnn::engine& cpu_engine = GetEngine();
    mkldnn::memory::format mem_format = dims_mkl.size() == 5 ? mkldnn::memory::format::ncdhw : mkldnn::memory::format::nchw;
    // Per ONNX spec, X (src) is NCHW and Y (dst) is NCHW
    auto src_md = mkldnn::memory::desc(dims_mkl, MklDnnType<T>(), mem_format);
    auto dst_md = mkldnn::memory::desc(dims_mkl, MklDnnType<T>(), mem_format);

    // Reorder src memory layout if necessary.
    if (src_md.data.format != lrn_primitive->GetSrcMemoryFormat()) {
      auto pd = mkldnn::memory::primitive_desc(src_md, cpu_engine);
      mkldnn::memory src = mkldnn::memory(pd, (void*)src_data);
      // allocate the size queried from memory primitive desc. it may not match tensor logical size due to
      // mkldnn using padding to allow use of blocked format.
      src_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc, lrn_primitive->GetSrcSize());
      mkldnn::memory dst = mkldnn::memory(fwd_primitive_desc->src_primitive_desc(), src_reorder_buffer.get());
      MemoryReorderParams params(src, dst);
      DoReorder<T>(params);
      src_data = static_cast<T*>(dst.get_data_handle());
    }

    // Allocate dst buffer if reorder is necessary
    if (src_md.data.format != lrn_primitive->GetSrcMemoryFormat()) {
      // allocate the size queried from memory primitive desc. it may not match tensor logical size due to
      // mkldnn using padding to allow use of blocked format.
      dst_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc, lrn_primitive->GetDstSize());
      dst_data = static_cast<T*>(dst_reorder_buffer.get());
    }

    lrn_primitive->Compute(src_data, dst_data);

    // Reorder dst memory layout if necessary
    if (src_md.data.format != lrn_primitive->GetSrcMemoryFormat()) {
      mkldnn::memory src = mkldnn::memory(fwd_primitive_desc->dst_primitive_desc(), (void*)dst_data);
      auto pd = mkldnn::memory::primitive_desc(dst_md, cpu_engine);
      mkldnn::memory dst = mkldnn::memory(pd, Y->template MutableData<T>());
      MemoryReorderParams params(src, dst);
      DoReorder<T>(params);
    }
  } catch (const mkldnn::error& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Status: ", e.status, ", message: ", e.message.c_str());
  }

  return Status::OK();
}

}  // namespace mkl_dnn
}  // namespace onnxruntime
