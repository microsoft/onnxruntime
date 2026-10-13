/*
Fixed version of the user's original C++ custom operator implementation.

Key fixes applied:
1. Changed AvggKernel to ThresholdedReluKernel
2. Fixed input/output signature: 1 input -> 1 output
3. Implemented actual ThresholdedRelu operation instead of averaging
4. Fixed operator registration name to match ONNX model
5. Used consistent domain name
6. Simplified registration approach
*/

#include "fixed_original.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>

// FIXED: Renamed and corrected the kernel implementation
struct ThresholdedReluKernel {
    ThresholdedReluKernel() = default;
    ThresholdedReluKernel(const OrtApi& api, const OrtKernelInfo* info) {}
    ThresholdedReluKernel(const OrtApi* api, const OrtKernelInfo* info) {}

    // FIXED: Corrected signature - 1 input, 1 output (not 2 inputs, 1 output)
    void Compute(const Ort::Custom::Tensor<float>& X,  // Input tensor
                 Ort::Custom::Tensor<float>& Y) {      // Output tensor (not Z)
        std::cout << "Calling ThresholdedReluKernel" << std::endl;
        
        const float* x_data = X.Data();
        float* y_data = Y.Allocate(X.Shape());  // FIXED: Use input shape for output
        int64_t count = Y.NumberOfElement();
        
        // FIXED: Implement ThresholdedRelu instead of averaging
        // ThresholdedRelu: Y[i] = X[i] if X[i] > alpha, else 0
        float alpha = 1.0f;  // Default threshold
        
        for (int64_t i = 0; i < count; ++i) {
            y_data[i] = (x_data[i] > alpha) ? x_data[i] : 0.0f;  // FIXED: ThresholdedRelu logic
        }
    }
};

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
    static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
    static std::mutex ort_custom_op_domain_mutex;
    std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
    ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    
    // FIXED: Use consistent domain name matching the ONNX model
    Ort::CustomOpDomain domain("custom.ops");  // FIXED: Was "riscv_test"
    
    // FIXED: Register with correct operator name and kernel type
    static const OrtCustomOp* thresholded_relu = Ort::Custom::CreateLiteCustomOp<ThresholdedReluKernel>(
        "ThresholdedRelu",       // FIXED: Was "AvggKernel" 
        "CPUExecutionProvider"
    );
    domain.Add(thresholded_relu);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
    
    return nullptr;
}