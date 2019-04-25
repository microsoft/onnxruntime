// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/common/cpuid_info.h"
#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/nn/pool.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"

namespace onnxruntime {
namespace mkl_dnn {

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version, end_version)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      op_name,                                                                          \
      kOnnxDomain,                                                                      \
      since_version,                                                                    \
      end_version,                                                                      \
      data_type,                                                                        \
      kMklDnnExecutionProvider,                                                         \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type>);

POOLING_KERNEL(AveragePool, float, AveragePool, 7, 8)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1, 8)
POOLING_KERNEL(MaxPool, float, MaxPool<1>, 1, 7)
POOLING_KERNEL(MaxPool, float, MaxPool<8>, 8, 8)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1, 8)

namespace {
// Struct which encapsulates parameters for MKLDNN Pool primitive.
struct PoolParams {
  std::string op_name;
  // Pooling primitive needs version as part of key because there
  // are multiple versions of mkldnn pool kernels.
  std::string version;
  mkldnn::memory::dims& src_dims;
  mkldnn::memory::dims& dst_dims;
  mkldnn::memory::dims& kernel;
  mkldnn::memory::dims& strides;
  mkldnn::memory::dims& padding_left;
  mkldnn::memory::dims& padding_right;
  bool count_include_pad;

  PoolParams(const std::string& op_name, const std::string& version,
             mkldnn::memory::dims& src_dims, mkldnn::memory::dims& dst_dims,
             mkldnn::memory::dims& kernel, mkldnn::memory::dims& strides,
             mkldnn::memory::dims& padding_left, mkldnn::memory::dims& padding_right,
             bool count_include_pad)
      : op_name(op_name),
        version(version),
        src_dims(src_dims),
        dst_dims(dst_dims),
        kernel(kernel),
        strides(strides),
        padding_left(padding_left),
        padding_right(padding_right),
        count_include_pad(count_include_pad) {}

  // Used as the key for Pool Primitive Reuse Pool.
  std::string ToString() const {
    std::string key;
    key.reserve(128);
    key.append(op_name);
    key.append(version);
    AddDimsToKey(key, src_dims);
    AddDimsToKey(key, dst_dims);
    AddDimsToKey(key, kernel);
    AddDimsToKey(key, strides);
    AddDimsToKey(key, padding_left);
    AddDimsToKey(key, padding_right);
    key.append(count_include_pad ? "true" : "false");
    return key;
  }
};

template <typename T, typename PoolType>
class PoolPrimitive : public PrimitiveBase {
 public:
  explicit PoolPrimitive(const PoolParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.pool_fwd == nullptr) {
      Initialize(params);
    }
  }

  ~PoolPrimitive() = default;

  void Compute(const T* src_data, T* dst_data) {
    context_.src_mem->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
    context_.stream->submit(context_.net);

    context_.src_mem->set_data_handle(nullptr);
    context_.dst_mem->set_data_handle(nullptr);
    return;
  }

  mkldnn::memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }
  mkldnn::memory::format GetDstMemoryFormat() const { return context_.dst_fmt; }

  size_t GetSrcSize() const { return context_.src_size; }
  size_t GetDstSize() const { return context_.dst_size; }

  // std::unique_ptr<mkldnn::memory::desc> GetDstMemoryDesc() const { return context_.dst_md; }
  mkldnn::pooling_forward::primitive_desc* GetPrimitiveDesc() const {
    return context_.fwd_primitive_desc.get();
  }

 private:
  struct PoolContext {
    mkldnn::memory::format src_fmt;
    mkldnn::memory::format dst_fmt;

    size_t src_size;
    size_t dst_size;

    std::unique_ptr<mkldnn::memory> src_mem;
    std::unique_ptr<mkldnn::memory> dst_mem;

    std::unique_ptr<mkldnn::pooling_forward::desc> fwd_desc;

    std::unique_ptr<mkldnn::memory::desc> src_md;
    std::unique_ptr<mkldnn::memory::desc> dst_md;

    std::unique_ptr<mkldnn::pooling_forward::primitive_desc> fwd_primitive_desc;

    std::unique_ptr<mkldnn::primitive> pool_fwd;

    std::unique_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;

    PoolContext()
        : src_fmt(mkldnn::memory::format::any),
          dst_fmt(mkldnn::memory::format::any),
          src_size(0),
          dst_size(0),
          src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          src_md(nullptr),
          fwd_primitive_desc(nullptr),
          pool_fwd(nullptr),
          stream(nullptr) {}
  };

  void Initialize(const PoolParams& params) {
    bool is_2D = params.src_dims.size() == 4 ? true : false;
    mkldnn::memory::format fmt = mkldnn::memory::format::any;
    if (CPUIDInfo::GetCPUIDInfo().HasAVX512f()) {
      fmt = is_2D ? mkldnn::memory::format::nChw16c : mkldnn::memory::format::nCdhw16c;
    } else if (CPUIDInfo::GetCPUIDInfo().HasAVX2() && (params.src_dims[1] % 8 == 0)) {
      fmt = is_2D ? mkldnn::memory::format::nChw8c : mkldnn::memory::format::ncdhw;
    } else {
      fmt = is_2D ? mkldnn::memory::format::nchw : mkldnn::memory::format::ncdhw;
    }
    context_.src_md.reset(new mkldnn::memory::desc(
        {params.src_dims}, MklDnnType<T>(), fmt));
    context_.dst_md.reset(new mkldnn::memory::desc(
        {params.dst_dims}, MklDnnType<T>(), mkldnn::memory::format::any));

    mkldnn::algorithm algo = mkldnn::algorithm::pooling_max;
    if (PoolType::type == onnxruntime::PoolType::kAveragePool) {
      algo = mkldnn::algorithm::pooling_avg_exclude_padding;
      if (params.count_include_pad) {
        algo = mkldnn::algorithm::pooling_avg_include_padding;
      }
    }
    context_.fwd_desc.reset(new mkldnn::pooling_forward::desc(
        mkldnn::prop_kind::forward_inference, algo,
        *context_.src_md, *context_.dst_md,
        params.strides, params.kernel,
        params.padding_left, params.padding_right,
        mkldnn::padding_kind::zero));

    context_.fwd_primitive_desc.reset(new mkldnn::pooling_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    context_.src_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_primitive_desc.get()->src_primitive_desc().desc().data.format);

    context_.dst_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_primitive_desc.get()->dst_primitive_desc().desc().data.format);

    context_.src_size = context_.fwd_primitive_desc.get()->src_primitive_desc().get_size();
    context_.dst_size = context_.fwd_primitive_desc.get()->dst_primitive_desc().get_size();

    context_.src_mem.reset(
        new mkldnn::memory(context_.fwd_primitive_desc.get()->src_primitive_desc(), nullptr));
    context_.dst_mem.reset(
        new mkldnn::memory(context_.fwd_primitive_desc.get()->dst_primitive_desc(), nullptr));
    context_.pool_fwd.reset(
        new mkldnn::pooling_forward(*context_.fwd_primitive_desc, *context_.src_mem, *context_.dst_mem));
    context_.net.push_back(*context_.pool_fwd);
  }

  PoolContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN Pool primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T, typename PoolType>
class PoolPrimitivePool : public PrimitivePool<T> {
 public:
  static PoolPrimitive<T, PoolType>* Get(const PoolParams& params) {
    PoolPrimitive<T, PoolType>* primitive = dynamic_cast<PoolPrimitive<T, PoolType>*>(
        PoolPrimitivePool<T, PoolType>::GetInstance().GetPrimitive(params.ToString()));
    if (primitive == nullptr) {
      auto pool_primitive = std::make_unique<PoolPrimitive<T, PoolType>>(params);
      primitive = pool_primitive.get();
      PoolPrimitivePool<T, PoolType>::GetInstance().SetPrimitive(params.ToString(), std::move(pool_primitive));
    }
    return primitive;
  }

 private:
  PoolPrimitivePool() = default;
  ~PoolPrimitivePool() = default;

  static PoolPrimitivePool& GetInstance() {
    static PoolPrimitivePool pool;
    return pool;
  }
};
}  // namespace

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  if (x_shape.NumDimensions() == 3) {
    // Fall Back to CPU implementation.
    return onnxruntime::Pool<T, PoolType>::Compute(context);
  }

  std::vector<int64_t> kernel_shape = this->kernel_shape_;
  std::vector<int64_t> pads = this->pads_;
  std::vector<int64_t> strides = this->strides_;

  if (this->global_pooling_) {
    kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    pads.assign(kernel_shape.size() * 2, 0);
    strides.assign(kernel_shape.size(), 1);
  }

  std::vector<int64_t> y_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads, this->dilations_, this->ceil_mode_);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  size_t num_outputs = OpKernel::Node().OutputDefs().size();
  if (num_outputs == 2) {
    Tensor* I = context->Output(1, TensorShape(y_dims));
    if (nullptr != I) {
      return onnxruntime::Pool<T, PoolType>::Compute(context);
    }
  }

  const T* src_data = X->template Data<T>();
  T* dst_data = Y->template MutableData<T>();

  mkldnn::memory::dims src_dims_mkl(x_dims.begin(), x_dims.end());
  mkldnn::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());
  mkldnn::memory::dims kernel_mkl(kernel_shape.begin(), kernel_shape.end());
  mkldnn::memory::dims strides_mkl(strides.begin(), strides.end());
  mkldnn::memory::dims padding_left_mkl(pads.begin(), pads.begin() + (pads.size() / 2));
  mkldnn::memory::dims padding_right_mkl(pads.begin() + (pads.size() / 2), pads.end());

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  IAllocatorUniquePtr<void> src_reorder_buffer;
  IAllocatorUniquePtr<void> dst_reorder_buffer;

  try {
    PoolParams pool_params(this->op_name_, this->opset_version_,
                           src_dims_mkl, dst_dims_mkl,
                           kernel_mkl, strides_mkl,
                           padding_left_mkl, padding_right_mkl,
                           this->count_include_pad_);
    PoolPrimitive<T, PoolType>* pool_primitive = PoolPrimitivePool<T, PoolType>::Get(pool_params);
    auto fwd_primitive_desc = pool_primitive->GetPrimitiveDesc();

    mkldnn::engine& cpu_engine = GetEngine();
    mkldnn::memory::format mem_format = src_dims_mkl.size() == 5 ? mkldnn::memory::format::ncdhw : mkldnn::memory::format::nchw;
    // Per ONNX spec, X (src) is NCHW and Y (dst) is NCHW
    auto src_md = mkldnn::memory::desc(src_dims_mkl, MklDnnType<T>(), mem_format);
    auto dst_md = mkldnn::memory::desc(dst_dims_mkl, MklDnnType<T>(), mem_format);

    // Reorder src memory layout if necessary.
    if (src_md.data.format != pool_primitive->GetSrcMemoryFormat()) {
      auto pd = mkldnn::memory::primitive_desc(src_md, cpu_engine);
      mkldnn::memory src = mkldnn::memory(pd, (void*)src_data);
      // allocate the size queried from memory primitive desc. it may not match tensor logical size due to
      // mkldnn using padding to allow use of blocked format.
      src_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc, pool_primitive->GetSrcSize());
      mkldnn::memory dst = mkldnn::memory(fwd_primitive_desc->src_primitive_desc(), src_reorder_buffer.get());
      MemoryReorderParams params(src, dst);
      DoReorder<T>(params);
      src_data = static_cast<T*>(dst.get_data_handle());
    }

    // Allocate dst buffer if reorder is necessary
    if (dst_md.data.format != pool_primitive->GetDstMemoryFormat()) {
      // allocate the size queried from memory primitive desc. it may not match tensor logical size due to
      // mkldnn using padding to allow use of blocked format.
      dst_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc, pool_primitive->GetDstSize());
      dst_data = static_cast<T*>(dst_reorder_buffer.get());
    }

    pool_primitive->Compute(src_data, dst_data);

    // Reorder dst memory layout if necessary
    if (dst_md.data.format != pool_primitive->GetDstMemoryFormat()) {
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
