#pragma once
#include "onnxruntime_cxx_api.h"
#include <numeric>

namespace Ort {
namespace Custom2 {

class Tensor {
 public:
  Tensor(OrtKernelContext* ctx) : ctx_(ctx) {}

 protected:
  struct KernelContext ctx_;
};

template <typename T>
struct Span {
  const T* data_ = {};
  size_t size_ = {};
  void Assign(const T* data, size_t size) {
    data_ = data;
    size_ = size;
  }
  size_t size() const { return size_; }
  T operator[](size_t indice) const {
    return data_[indice];
  }
};

template <typename T>
class TensorT : public Tensor {
 public:
  using TT = typename std::remove_reference<T>::type;
  TensorT(OrtKernelContext* ctx, size_t indice, bool is_input) : Tensor(ctx), indice_(indice), is_input_(is_input) {
    if (is_input) {
      const_value_ = ctx_.GetInput(indice);
      auto type_shape_info = const_value_.GetTensorTypeAndShapeInfo();
      shape_ = type_shape_info.GetShape();
    }
  }
  std::vector<int64_t> Shape() const {
    return shape_;
  }
  int64_t NumberOfElement() const {
    if (shape_.empty()) {
      return 0;
    } else {
      return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<int64_t>());
    }
  }
  const TT* Data() const {
    return reinterpret_cast<const TT*>(const_value_.GetTensorRawData());
  }
  TT* Allocate(const std::vector<int64_t>& shape) {
    if (!data_) {
      shape_ = shape;
      data_ = ctx_.GetOutput(indice_, shape).GetTensorMutableData<TT>();
    }
    return data_;
  }
  static TT GetT() { return (TT)0; }

  const Span<T>& AsSpan() {
    // assert shape_ is 1-d
    span_.Assign(Data(), shape_[0]);
    return span_;
  }

  const T& AsScalar() {
    // assert shape_ is {1}
    return *Data();
  }

 private:
  size_t indice_;
  bool is_input_;
  ConstValue const_value_;  // for input
  TT* data_{};              // for output
  std::vector<int64_t> shape_;
  Span<T> span_;
};

using TensorPtr = std::unique_ptr<Custom2::Tensor>;

template <typename CustomType, typename... Args>
struct OrtCustomOpT2 : public OrtCustomOp {
  using InitFn = CustomType* (*)(const OrtKernelInfo*);
  using ComputeFn = void (*)(Args...);
  using ExitFn = void (*)(CustomType*);
  using MyType = OrtCustomOpT2<CustomType, Args...>;

  OrtCustomOpT2(const char* op_name,
                const char* execution_provider,
                InitFn init_fn,
                ComputeFn compute_fn,
                ExitFn exit_fn) : op_name_(op_name),
                                  execution_provider_(execution_provider),
                                  init_fn_(init_fn),
                                  compute_fn_(compute_fn),
                                  exit_fn_(exit_fn) {
    ParseArgs<Args...>();

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto self = reinterpret_cast<MyType*>(op_kernel);
      auto t = self->CreateInputTuple<0, 0, Args...>(context);
      std::apply([self](Args const&... t_args) { self->compute_fn_(t_args...); }, t);
    };

    OrtCustomOp::version = ORT_API_VERSION;

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi*, const OrtKernelInfo* info) {
      auto self = const_cast<MyType*>(reinterpret_cast<const MyType*>(this_));
      if (self->init_fn_) {
        self->custom_handle_ = self->init_fn_(info);
      }
      return (void*)this_;
    };

    OrtCustomOp::GetName = [](const OrtCustomOp* this_) { return static_cast<const MyType*>(this_)->op_name_.c_str(); };
    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) { return ((MyType*)this_)->execution_provider_.c_str(); };
    OrtCustomOp::GetInputMemoryType = [](const OrtCustomOp*, size_t) { return OrtMemTypeDefault; };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->input_types_.size();
    };

    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t indice) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->input_types_[indice];
    };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->output_types_.size();
    };

    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t indice) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->output_types_[indice];
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      auto self = reinterpret_cast<MyType*>(op_kernel);
      if (self->exit_fn_) {
        self->exit_fn_(self->custom_handle_);
      }
    };

    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; };
    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; };
    OrtCustomOp::GetVariadicInputMinArity = [](const OrtCustomOp*) { return 0; };
    OrtCustomOp::GetVariadicInputHomogeneity = [](const OrtCustomOp*) { return 0; };
    OrtCustomOp::GetVariadicOutputMinArity = [](const OrtCustomOp*) { return 0; };
    OrtCustomOp::GetVariadicOutputHomogeneity = [](const OrtCustomOp*) { return 0; };
  }

  /////////////////////////////  create input tuple ///////////////////////////////

  template <size_t ith_input, size_t ith_output, typename... Ts>
  typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateInputTuple(OrtKernelContext* context) {
    return std::make_tuple();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, CustomType*>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    std::tuple<T> current = std::tuple<T>{custom_handle_};
    auto next = CreateInputTuple<ith_input, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, OrtKernelContext*>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    std::tuple<T> current = std::tuple<OrtKernelContext*>{context};
    auto next = CreateInputTuple<ith_input, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  // tensor inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::TensorT<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  // span inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::Span<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<float>*>(tensors_.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::Span<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int32_t>*>(tensors_.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  //scalar inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, float>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<float>*>(tensors_.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, int32_t>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int32_t>*>(tensors_.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  // tensor outputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Custom2::TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Custom2::TensorT<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(context);
    return std::tuple_cat(current, next);
  }

  /////////////////////////////  parse args ///////////////////////////////

  template <typename... Ts>
  typename std::enable_if<0 == sizeof...(Ts)>::type
  ParseArgs() {
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, CustomType*>::value>::type
  ParseArgs() {
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, OrtKernelContext*>::value>::type
  ParseArgs() {
    ParseArgs<Ts...>();
  }

  // tensor inputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<float>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<int32_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  // span inputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<float>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<int32_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  //scalar inputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, float>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, int32_t>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  // outputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<float>&>::value>::type
  ParseArgs() {
    output_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<int32_t>&>::value>::type
  ParseArgs() {
    output_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  ////////////////////////////// members //////////////////////////////

  const std::string op_name_;
  const std::string execution_provider_;

  const InitFn init_fn_;
  const ComputeFn compute_fn_;
  const ExitFn exit_fn_;

  CustomType* custom_handle_ = {};

  std::vector<TensorPtr> tensors_;
  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<ONNXTensorElementDataType> output_types_;
};  // class OrtCustomOpLite

template <typename... Args>
OrtCustomOp* CreateCustomOpT2(const char* op_name,
                              const char* execution_provider,
                              void (*custom_compute_fn)(Args...)) {
  using OrtCustomOpTPtr = OrtCustomOpT2<void, Args...>;
  return std::make_unique<OrtCustomOpTPtr>(op_name, execution_provider, nullptr, custom_compute_fn, nullptr).release();
}

template <typename T, typename... Args>
OrtCustomOp* CreateCustomOpT2(const char* op_name,
                              const char* execution_provider,
                              T* (*custom_init_fn)(const OrtKernelInfo*),
                              void (*custom_compute_fn)(Args...),
                              void (*custom_exit_fn)(T*)) {
  using OrtCustomOpTPtr = OrtCustomOpT2<T, Args...>;
  return std::make_unique<OrtCustomOpTPtr>(op_name, execution_provider, custom_init_fn, custom_compute_fn, custom_exit_fn).release();
}

}  // namespace Custom2
}  // namespace Ort