#include "kernel_ep.h"
//#include "core/session/onnxruntime_lite_custom_op.h"
#include <memory>
#include <vector>
namespace onnxruntime {

struct MyRelu : OrtCustomOp {
    MyRelu() {
        OrtCustomOp::version = ORT_API_VERSION;
        OrtCustomOp::GetName = [](const struct OrtCustomOp* op) { return "Relu"; };
        OrtCustomOp::GetExecutionProviderType = [](const struct OrtCustomOp* op) { return "KernelEp"; };
        OrtCustomOp::CreateKernelV2 = [](const struct OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info, void** kernel) -> OrtStatusPtr {
            return nullptr;
        };
        OrtCustomOp::KernelComputeV2 = [](void* op_kernel, OrtKernelContext* context) -> OrtStatusPtr {
            const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
            const OrtValue* input = nullptr;
            api->KernelContext_GetInput(context, 0, &input);
            std::vector<int64_t> dim(1,4);
            OrtValue* output = nullptr;
            api->KernelContext_GetOutput(context, 0, dim.data(), dim.size(), &output);

            float* input_raw = nullptr, *output_raw = nullptr;
            api->GetTensorMutableData(const_cast<OrtValue*>(input), reinterpret_cast<void**>(&input_raw));
            api->GetTensorMutableData(output, reinterpret_cast<void**>(&output_raw));

            for (int i = 0; i < 4; i++) {
                output_raw[i] = input_raw[i];
                if (input_raw[i] < 0) output_raw[i] = 0;

                output_raw[i] = 2.0;
            }
            return nullptr;
        };
        OrtCustomOp::GetInputTypeCount = [](const struct OrtCustomOp* op) -> size_t { return 1; };
        OrtCustomOp::GetOutputTypeCount = [](const struct OrtCustomOp* op) -> size_t { return 1; };
        OrtCustomOp::GetInputMemoryType = [](const struct OrtCustomOp* op, size_t index) { return OrtMemType::OrtMemTypeDefault; };
        OrtCustomOp::GetInputType = [](const struct OrtCustomOp* op, size_t index) { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
        OrtCustomOp::GetOutputType = [](const struct OrtCustomOp* op, size_t index) { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
        OrtCustomOp::GetStartVersion = [](const struct OrtCustomOp* op) { return 14; };
    }
};

//void MyRelu(const Ort::Custom::Tensor<float>& X, Ort::Custom::Tensor<float>& Y) {
//  const auto& shape = X.Shape();
//  auto X_raw = X.Data();
//  auto Y_raw = Y.Allocate(shape);
//  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
//  for (int64_t i = 0; i < total; i++) {
//    Y_raw[i] = X_raw[i] > 0 ? X_raw[i] : 0;
//  }
//  std::cout<<"In MyRelu()\n";
//}

KernelEp::KernelEp(const char* ep_type, const KernelEpInfo& ep_info) : info(ep_info) {
    type = ep_type;
    OrtExecutionProvider::RegisterKernels = [](OrtKernelRegistry* kernel_registry) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        //Ort::Custom::OrtLiteCustomOp* op = Ort::Custom::CreateLiteCustomOp("Relu", "kernel_ep", MyRelu);

        OrtTypeConstraints* type_constraints = nullptr;
        api->CreateOrtTypeConstraints(&type_constraints);
        api->AddTypeConstraint(type_constraints, "T", ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        OrtCustomOp* op = new MyRelu();
        api->OrtKernelRegistry_RegisterKernel(kernel_registry, op, type_constraints);
        api->ReleaseTypeConstraints(type_constraints);
    };
}

KernelEpFactory::KernelEpFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> OrtExecutionProvider* {
        KernelEpInfo info;
        for (size_t i = 0; i < option_size; i++) {
            if (!strcmp(ep_option_keys[i], "int_property")) info.int_property = std::atoi(ep_option_values[i]);
            else if (!strcmp(ep_option_keys[i], "str_property")) info.str_property = ep_option_values[i];
            // TODO(leca): else throw
        }
        std::unique_ptr<KernelEp> ret = std::make_unique<KernelEp>("KernelEp", std::move(info));
        return ret.release();
    };
}

}

#ifdef __cplusplus
extern "C" {
#endif
OrtExecutionProviderFactory* RegisterCustomEp() {
    std::unique_ptr<onnxruntime::KernelEpFactory> ret = std::make_unique<onnxruntime::KernelEpFactory>();
    return ret.release();
}
#ifdef __cplusplus
}
#endif
