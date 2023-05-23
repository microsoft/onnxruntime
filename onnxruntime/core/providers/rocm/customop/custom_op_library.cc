#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_lite_custom_op.h"

#include <vector>
#include <cmath>
#include <mutex>

#include "core/common/common.h"


static const char* c_OpDomain = "com.custom";

// lite custom op as a function
template<typename T>
void FuseMatMulGeluMatMul(
    OrtKernelContext*,
    const Ort::Custom::Tensor<T>& data,
    const Ort::Custom::Tensor<T>& weight1,
    const Ort::Custom::Tensor<T>& bias1,
    const Ort::Custom::Tensor<T>& weight2,
    const Ort::Custom::Tensor<T>& bias2,
    Ort::Custom::Tensor<T>& output) {
  const auto &shape = data.Shape();
  output.allocate(shape);
  LOGS_DEFAULT(VERBOSE) << "Here run into FuseMatMulGeluMatMul";
}

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  using LiteOp = Ort::Custom::OrtLiteCustomOp;
  static const std::unique_ptr<LiteOp> c_FuseMatGeluMat_rocm_f32{Ort::Custom::CreateLiteCustomOp("fuse_matmul_gelu_matmul", "ROCMExecutionProvider", FuseMatMulGeluMatMul<float>)};
  static const std::unique_ptr<LiteOp> c_FuseMatGeluMat_rocm_f16{Ort::Custom::CreateLiteCustomOp("fuse_matmul_gelu_matmul", "ROCMExecutionProvider", FuseMatMulGeluMatMul<half>)};
  static const std::unique_ptr<LiteOp> c_FuseMatGeluMat_cpu_f32{Ort::Custom::CreateLiteCustomOp("fuse_matmul_gelu_matmul", "CPUExecutionProvider", FuseMatMulGeluMatMul<float>)};
  static const std::unique_ptr<LiteOp> c_FuseMatGeluMat_cpu_f16{Ort::Custom::CreateLiteCustomOp("fuse_matmul_gelu_matmul", "CPUExecutionProvider", FuseMatMulGeluMatMul<half>)};

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(c_FuseMatGeluMat_rocm_f32.get());
    domain.Add(c_FuseMatGeluMat_rocm_f16.get());
    domain.Add(c_FuseMatGeluMat_cpu_f32.get());
    domain.Add(c_FuseMatGeluMat_cpu_f16.get());

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
