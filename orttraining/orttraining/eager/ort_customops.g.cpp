// AUTO-GENERATED CODE! - DO NOT EDIT!
// $ python /home/lw/code/onnxruntime/orttraining/orttraining/eager/opgen/opgen.py --output_file /home/lw/code/onnxruntime/orttraining/orttraining/eager/ort_customops.g.cpp.working --ops_module /home/lw/code/onnxruntime/orttraining/orttraining/eager/opgen/opgen/custom_ops.py --header_file /home/lw/code/onnxruntime/orttraining/orttraining/eager/opgen/CustomOpDeclarations.h --custom_ops

#include "python/onnxruntime_pybind_state_common.h"

#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>

#include <core/providers/dml/OperatorAuthorHelper/Attributes.h>

#include "ort_tensor.h"
#include "ort_aten.h"
#include "ort_log.h"

namespace torch_ort {
namespace eager {

using namespace at;
using NodeAttributes = onnxruntime::NodeAttributes;

Tensor gemm(
  const Tensor& A, 
  const Tensor& B, 
  const Tensor& C, 
  double alpha, 
  double beta, 
  int64_t transA, 
  int64_t transB) {
  ORT_LOG_FN(A, B, C, alpha, beta, transA, transB);
  
  auto& invoker = GetORTInvoker(A.device());
  
  auto ort_input_0_A = create_ort_value(invoker, A);
  auto ort_input_0_B = create_ort_value(invoker, B);
  auto ort_input_0_C = create_ort_value(invoker, C);
  
  NodeAttributes attrs_0(4);
  attrs_0["alpha"] = create_ort_attribute(
    "alpha", alpha, at::ScalarType::Float);
  attrs_0["beta"] = create_ort_attribute(
    "beta", beta, at::ScalarType::Float);
  attrs_0["transA"] = create_ort_attribute(
    "transA", transA, at::ScalarType::Int);
  attrs_0["transB"] = create_ort_attribute(
    "transB", transB, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_Gemm(1);
  
  auto status = invoker.Invoke("Gemm", {
    std::move(ort_input_0_A),
    std::move(ort_input_0_B),
    std::move(ort_input_0_C),
  }, ort_outputs_0_Gemm, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = A.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Gemm[0]),
    tensor_options);
}

// batchnorm_inplace(Tensor(a!) X, Tensor scale, Tensor b, Tensor(b!) input_mean, Tensor(c!) input_var, float epsilon, float momentum) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor&, Tensor&, Tensor&> batchnorm_inplace(
  Tensor& X, 
  const Tensor& scale, 
  const Tensor& B, 
  Tensor& input_mean, 
  Tensor& input_var, 
  const double epsilon, 
  const double momentum) {
  ORT_LOG_FN(X, scale, B, input_mean, input_var, epsilon, momentum);
  
  auto& invoker = GetORTInvoker(X.device());
  
  auto ort_input_0_X = create_ort_value(invoker, X);
  auto ort_input_0_scale = create_ort_value(invoker, scale);
  auto ort_input_0_B = create_ort_value(invoker, B);
  auto ort_input_0_input_mean = create_ort_value(invoker, input_mean);
  auto ort_input_0_input_var = create_ort_value(invoker, input_var);
  
  NodeAttributes attrs_0(3);
  attrs_0["epsilon"] = create_ort_attribute(
    "epsilon", epsilon, at::ScalarType::Float);
  attrs_0["momentum"] = create_ort_attribute(
    "momentum", momentum, at::ScalarType::Float);
  attrs_0["training_mode"] = create_ort_attribute(
    "training_mode", 1, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_BatchNormalization(3);
  ort_outputs_0_BatchNormalization[0] = ort_input_0_X;
  ort_outputs_0_BatchNormalization[1] = ort_input_0_input_mean;
  ort_outputs_0_BatchNormalization[2] = ort_input_0_input_var;
  
  auto status = invoker.Invoke("BatchNormalization", {
    std::move(ort_input_0_X),
    std::move(ort_input_0_scale),
    std::move(ort_input_0_B),
    std::move(ort_input_0_input_mean),
    std::move(ort_input_0_input_var),
  }, ort_outputs_0_BatchNormalization, &attrs_0);
  CHECK_STATUS(status);
  
  return std::tuple<Tensor&,Tensor&,Tensor&>(X, input_mean, input_var);
}

// my_cat(Tensor[] tensors, int dim=0) -> Tensor
Tensor my_cat(
  TensorList tensors, 
  int64_t dim) {
  ORT_LOG_FN(tensors, dim);
  
  assert(tensors.size()>0);
  auto& invoker = GetORTInvoker(tensors[0].device());
  
  auto ort_input_0_tensors = create_ort_value(invoker, tensors);
  
  NodeAttributes attrs_0(1);
  attrs_0["axis"] = create_ort_attribute(
    "axis", dim, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_Concat(1);
  
  auto status = invoker.Invoke("Concat", {
    std::move(ort_input_0_tensors),
  }, ort_outputs_0_Concat, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = tensors[0].options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Concat[0]),
    tensor_options);
}

TORCH_LIBRARY(ort, m) {
  m.def("gemm", &gemm);
  m.def("batchnorm_inplace", &batchnorm_inplace);
  m.def("my_cat", &my_cat);
}

} // namespace eager
} // namespace torch_ort
