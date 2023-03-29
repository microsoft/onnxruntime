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

// std::enable_if<std::is_same<T, std::string>::value>::type
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
      return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<int64_t>());
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

template <>
class TensorT<std::string> : public Tensor {
 public:
  using strings = std::vector<std::string>;

  TensorT(OrtKernelContext* ctx, size_t indice, bool is_input) : Tensor(ctx), indice_(indice), is_input_(is_input) {
    if (is_input) {
      auto const_value = ctx_.GetInput(indice);
      auto type_shape_info = const_value.GetTensorTypeAndShapeInfo();
      shape_ = type_shape_info.GetShape();
      auto num_chars = const_value.GetStringTensorDataLength();
      //todo - too much copies here ...
      std::vector<char> chars(num_chars + 1, '\0');
      auto num_strings = NumberOfElement();
      std::vector<size_t> offsets(NumberOfElement());
      const_value.GetStringTensorContent(static_cast<void*>(chars.data()), num_chars, offsets.data(), offsets.size());
      auto upper_bound = static_cast<int64_t>(num_strings) - 1;
      input_strings_.resize(num_strings);
      for (int64_t i = upper_bound; i >= 0; --i) {
        if (i < upper_bound) {
          chars[offsets[i + 1]] = '\0';
        }
        input_strings_[i] = chars.data() + offsets[i];
      }
    }
  }
  std::vector<int64_t> Shape() const {
    return shape_;
  }
  int64_t NumberOfElement() const {
    if (shape_.empty()) {
      return 0;
    } else {
      return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<int64_t>());
    }
  }
  const strings& Data() const {
    return input_strings_;
  }
  void SetStringOutput(size_t output_indice, const strings& ss, const std::vector<int64_t>& dims) {
    std::vector<const char*> raw;
    for (const auto& s: ss) {
      raw.push_back(s.data());
    }
    auto output = ctx_.GetOutput(0, dims.data(), dims.size());
    // note - there will be copy ...
    output.FillStringTensor(raw.data(), raw.size());
  }
  const std::string& AsScalar() {
    // assert shape_ is {1}
    return input_strings_[0];
  }
 private:
  size_t indice_;
  bool is_input_;
  std::vector<std::string> input_strings_; // for input
  // TT* data_{};              // for output
  std::vector<int64_t> shape_;
};

using TensorPtr = std::unique_ptr<Custom2::Tensor>;

//////////////////////////// OrtCustomOpBase ////////////////////////////////

struct OrtCustomOpBase : public OrtCustomOp {
  // CreateInputTuple
  template <size_t ith_input, size_t ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    return std::make_tuple();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, OrtKernelContext*>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    std::tuple<T> current = std::tuple<OrtKernelContext*>{context};
    auto next = CreateInputTuple<ith_input, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  // tensor inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Custom2::TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Custom2::TensorT<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Custom2::TensorT<std::string>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<std::string>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  // span inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Custom2::Span<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<float>*>(tensors.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Custom2::Span<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int32_t>*>(tensors.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  //scalar inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, float>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<float>*>(tensors.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, int32_t>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int32_t>*>(tensors.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const std::string&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<std::string>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<std::string>*>(tensors.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  // tensor outputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Custom2::TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<float>>(context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Custom2::TensorT<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<int32_t>>(context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Custom2::TensorT<std::string>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(OrtKernelContext* context, std::vector<TensorPtr>& tensors) {
    tensors.push_back(std::make_unique<Custom2::TensorT<std::string>>(context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(context, tensors);
    return std::tuple_cat(current, next);
  }

  // ParseArgs ...
  template <typename... Ts>
  static typename std::enable_if<0 == sizeof...(Ts)>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>&, std::vector<ONNXTensorElementDataType>&) {
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, OrtKernelContext*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    ParseArgs<Ts...>(input_types, output_types);
  }

  // tensor inputs
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<float>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<int32_t>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<std::string>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    ParseArgs<Ts...>(input_types, output_types);
  }

  // span inputs
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<float>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<int32_t>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>(input_types, output_types);
  }

  //scalar inputs
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, float>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, int32_t>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const std::string&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    ParseArgs<Ts...>(input_types, output_types);
  }

  // outputs
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<float>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    output_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<int32_t>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    output_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<std::string>&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    output_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    ParseArgs<Ts...>(input_types, output_types);
  }

  OrtCustomOpBase(const char* op_name,
                  const char* execution_provider) : op_name_(op_name),
                                                    execution_provider_(execution_provider) {
    OrtCustomOp::version = ORT_API_VERSION;

    OrtCustomOp::GetName = [](const OrtCustomOp* this_) { return static_cast<const OrtCustomOpBase*>(this_)->op_name_.c_str(); };
    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) { return ((OrtCustomOpBase*)this_)->execution_provider_.c_str(); };
    OrtCustomOp::GetInputMemoryType = [](const OrtCustomOp*, size_t) { return OrtMemTypeDefault; };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) {
      auto self = reinterpret_cast<const OrtCustomOpBase*>(this_);
      return self->input_types_.size();
    };

    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t indice) {
      auto self = reinterpret_cast<const OrtCustomOpBase*>(this_);
      return self->input_types_[indice];
    };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) {
      auto self = reinterpret_cast<const OrtCustomOpBase*>(this_);
      return self->output_types_.size();
    };

    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t indice) {
      auto self = reinterpret_cast<const OrtCustomOpBase*>(this_);
      return self->output_types_[indice];
    };

    OrtCustomOp::KernelDestroy = [](void* /*op_kernel*/) {};

    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; };
    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; };
    OrtCustomOp::GetVariadicInputMinArity = [](const OrtCustomOp*) { return 0; };
    OrtCustomOp::GetVariadicInputHomogeneity = [](const OrtCustomOp*) { return 0; };
    OrtCustomOp::GetVariadicOutputMinArity = [](const OrtCustomOp*) { return 0; };
    OrtCustomOp::GetVariadicOutputHomogeneity = [](const OrtCustomOp*) { return 0; };
  }

  const std::string op_name_;
  const std::string execution_provider_;

  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<ONNXTensorElementDataType> output_types_;
};

//////////////////////////// OrtCustomOpT1 ////////////////////////////////

template <typename... Args>
struct OrtCustomOpT1 : public OrtCustomOpBase {
  using ComputeFn = void (*)(Args...);
  using MyType = OrtCustomOpT1<Args...>;

  OrtCustomOpT1(const char* op_name,
                const char* execution_provider,
                ComputeFn compute_fn) : OrtCustomOpBase(op_name, execution_provider),
                                        compute_fn_(compute_fn) {
    ParseArgs<Args...>(input_types_, output_types_);

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto compute_fn = reinterpret_cast<ComputeFn>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateInputTuple<0, 0, Args...>(context, tensors);
      std::apply([compute_fn](Args const&... t_args) { compute_fn(t_args...); }, t);
    };

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi*, const OrtKernelInfo*) {
      return reinterpret_cast<void*>(static_cast<const MyType*>(this_)->compute_fn_);
    };
  }

  ComputeFn compute_fn_;
};  // struct OrtCustomOpT1

/////////////////////////// OrtCustomOpT2 ///////////////////////////

template <typename CustomOp>
struct OrtCustomOpT2 : public OrtCustomOpBase {
  template <typename... Args>
  using CustomComputeFn = void (CustomOp::*)(Args...);
  using MyType = OrtCustomOpT2<CustomOp>;

  OrtCustomOpT2(const char* op_name,
                const char* execution_provider) : OrtCustomOpBase(op_name,
                                                                  execution_provider) {
    init(&CustomOp::Compute);
  }

  template <typename... Args>
  void init(CustomComputeFn<Args...> custom_compute_fn) {
    ParseArgs<Args...>(input_types_, output_types_);

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto custom_op = reinterpret_cast<CustomOp*>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateInputTuple<0, 0, Args...>(context, tensors);
      std::apply([custom_op](Args const&... t_args) { custom_op->Compute(t_args...); }, t);
    };

    OrtCustomOp::CreateKernel = [](const OrtCustomOp*, const OrtApi* ort_api, const OrtKernelInfo* info) {
      return reinterpret_cast<void*>(new CustomOp(ort_api, info));
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      if (op_kernel) {
        delete reinterpret_cast<CustomOp*>(op_kernel);
      }
    };
  }
};  // struct OrtCustomOpT2

/////////////////////////// CreateCustomOp ////////////////////////////

template <typename... Args>
OrtCustomOp* CreateCustomOp(const char* op_name,
                            const char* execution_provider,
                            void (*custom_compute_fn)(Args...)) {
  using OrtCustomOpTPtr = OrtCustomOpT1<Args...>;
  return std::make_unique<OrtCustomOpTPtr>(op_name, execution_provider, custom_compute_fn).release();
}

template <typename CustomOp>
OrtCustomOp* CreateCustomOp(const char* op_name,
                            const char* execution_provider) {
  using OrtCustomOpTPtr = OrtCustomOpT2<CustomOp>;
  return std::make_unique<OrtCustomOpTPtr>(op_name, execution_provider).release();
}

}  // namespace Custom2
}  // namespace Ort
