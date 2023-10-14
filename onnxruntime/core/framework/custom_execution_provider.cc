#include "core/framework/custom_execution_provider.h"

namespace onnxruntime {

namespace lite {

template <size_t, size_t, typename... Ts>
typename std::enable_if<sizeof...(Ts) == 0>::type
SetBuilder(KernelDefBuilder&) {}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
typename std::enable_if<std::is_same<T, const TensorT<float>&>::value>::type
SetBuilder(KernelDefBuilder& builder) {
  builder.TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  SetBuilder<ith_input + 1, ith_output, Ts...>(builder);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
typename std::enable_if<std::is_same<T, const TensorT1<float>&>::value>::type
SetBuilder(KernelDefBuilder& builder) {
  builder.TypeConstraint("T1", DataTypeImpl::GetTensorType<float>());
  SetBuilder<ith_input + 1, ith_output, Ts...>(builder);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
typename std::enable_if<std::is_same<T, const TensorV<float>&>::value>::type
SetBuilder(KernelDefBuilder& builder) {
  builder.TypeConstraint("V", DataTypeImpl::GetTensorType<float>());
  SetBuilder<ith_input + 1, ith_output, Ts...>(builder);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
typename std::enable_if<std::is_same<T, Aliased<float>&>::value>::type
SetBuilder(KernelDefBuilder& builder) {
  builder.Alias(Aliased<float>::InputIndice(), 0);
  SetBuilder<ith_input, ith_output + 1, Ts...>(builder);
}

/////////////////////////////////////////////////////////////////////////////////////////////

inline onnxruntime::Status Identity(const TensorT<float>& /*X*/, Aliased<float>& /*Y*/) {
  return onnxruntime::Status::OK();
}

template <typename... Args>
onnxruntime::KernelDefBuilder RegisterKernel(const char* ep,
                                             const char* domain,
                                             const char* op,
                                             int since_ver,
                                             int end_ver,
                                             onnxruntime::Status (*compute_fn)(Args...)) {
  KernelDefBuilder builder;
  builder.Provider(ep).SetDomain(domain).SetName(op).SinceVersion(since_ver, end_ver);
  SetBuilder<0, 0, Args...>(builder);

  KernelCreateFn kernel_create_fn = [compute_fn](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> onnxruntime::Status {
    using LiteKernelFnType = LiteKernelFn<Args...>;
    out = std::make_unique<LiteKernelFnType>(info, compute_fn);
    return onnxruntime::Status::OK();
  };
  return KernelDefBuilder{};
}

onnxruntime::KernelDefBuilder identity_builder = RegisterKernel("MYEP", "ms.onnx", "identity", 12, 17, Identity);

}

}