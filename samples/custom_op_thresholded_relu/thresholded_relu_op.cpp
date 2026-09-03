#include "thresholded_relu_op.h"

// Define ORT_API_MANUAL_INIT before including the C++ API
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <onnxruntime_lite_custom_op.h>
#include <vector>
#include <mutex>
#include <iostream>

using namespace Ort::Custom;

/**
 * @brief ThresholdedRelu implementation using the lite custom op API
 * 
 * This implements the ThresholdedRelu operation:
 * Y[i] = X[i] if X[i] > alpha, else 0
 */
struct ThresholdedReluKernel {
    // Constructor - receives API and kernel info
    ThresholdedReluKernel(const OrtApi* api, const OrtKernelInfo* info) {
        // Default threshold value
        alpha_ = 1.0f;
        
        // Try to get alpha attribute if provided
        // Note: Error handling is simplified for this example
        size_t size = 0;
        auto status = api->KernelInfoGetAttribute_float(info, "alpha", &alpha_);
        if (status != nullptr) {
            // If alpha attribute is not provided, use default value
            alpha_ = 1.0f;
            api->ReleaseStatus(status);
        }
        
        std::cout << "ThresholdedRelu initialized with alpha = " << alpha_ << std::endl;
    }
    
    /**
     * @brief Compute function - implements the ThresholdedRelu operation
     * 
     * @param X Input tensor (float)
     * @param Y Output tensor (float) 
     */
    void Compute(const Ort::Custom::Tensor<float>& X, 
                 Ort::Custom::Tensor<float>& Y) {
        // Get input data and shape
        const float* x_data = X.Data();
        const auto& x_shape = X.Shape();
        
        // Allocate output tensor with same shape as input
        float* y_data = Y.Allocate(x_shape);
        
        // Get total number of elements
        int64_t count = X.NumberOfElement();
        
        // Apply ThresholdedRelu: Y[i] = X[i] if X[i] > alpha, else 0
        for (int64_t i = 0; i < count; ++i) {
            y_data[i] = (x_data[i] > alpha_) ? x_data[i] : 0.0f;
        }
        
        std::cout << "ThresholdedRelu computed " << count << " elements with alpha = " << alpha_ << std::endl;
    }
    
private:
    float alpha_;  // Threshold value
};

/**
 * @brief Container to manage custom op domain lifetime
 * 
 * This is important to prevent the domain from being destroyed
 * while the session is still using it.
 */
static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
    static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
    static std::mutex ort_custom_op_domain_mutex;
    std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
    ort_custom_op_domain_container.push_back(std::move(domain));
}

/**
 * @brief Register custom operators with ONNX Runtime
 * 
 * This is the main entry point that ONNX Runtime calls to register custom operators.
 */
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    // Initialize the global API
    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    
    try {
        // Create custom operator domain
        // This domain name MUST match the domain used in the ONNX model
        Ort::CustomOpDomain domain("custom.ops");
        
        // Create the ThresholdedRelu operator using the lite custom op API
        // The name "ThresholdedRelu" MUST match the op_type used in the ONNX model
        static const std::unique_ptr<OrtLiteCustomOp> thresholded_relu_op{
            Ort::Custom::CreateLiteCustomOp<ThresholdedReluKernel>(
                "ThresholdedRelu",        // Operator name - must match ONNX model
                "CPUExecutionProvider"    // Execution provider
            )
        };
        
        // Add the operator to the domain
        domain.Add(thresholded_relu_op.get());
        
        // Register the domain with the session options
        Ort::UnownedSessionOptions session_options(options);
        session_options.Add(domain);
        
        // Store the domain to keep it alive
        AddOrtCustomOpDomainToContainer(std::move(domain));
        
        std::cout << "ThresholdedRelu custom operator registered successfully" << std::endl;
        
        return nullptr;  // Success
    }
    catch (const std::exception& e) {
        // Convert C++ exception to ORT status
        std::cout << "Error registering custom operators: " << e.what() << std::endl;
        return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
    }
}