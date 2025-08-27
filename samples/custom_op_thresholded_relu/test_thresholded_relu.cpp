#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>

#ifdef _WIN32
    #define LIBRARY_PREFIX L""
    #define LIBRARY_EXTENSION L".dll"
#else
    #define LIBRARY_PREFIX "lib"
    #define LIBRARY_EXTENSION ".so"
#endif

/**
 * @brief Test application for ThresholdedRelu custom operator
 * 
 * This demonstrates the correct way to:
 * 1. Load a custom operator library
 * 2. Create a session with the custom operator
 * 3. Run inference with custom operator
 */
int main() {
    try {
        std::cout << "=== ThresholdedRelu Custom Operator Test ===" << std::endl;
        
        // Step 1: Initialize ONNX Runtime environment
        std::cout << "1. Initializing ONNX Runtime environment..." << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ThresholdedReluTest");
        
        // Step 2: Create session options and register custom operator library
        std::cout << "2. Registering custom operator library..." << std::endl;
        Ort::SessionOptions session_options;
        
        // Method 1: Register the library (most common approach)
        #ifdef _WIN32
            std::wstring library_path = LIBRARY_PREFIX L"thresholded_relu_op" LIBRARY_EXTENSION;
        #else
            std::string library_path = LIBRARY_PREFIX "thresholded_relu_op" LIBRARY_EXTENSION;
        #endif
        
        try {
            session_options.RegisterCustomOpsLibrary(library_path);
            std::cout << "   Custom operator library registered successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "   Warning: Could not load library " << 
                #ifdef _WIN32
                    std::string(library_path.begin(), library_path.end()) 
                #else
                    library_path
                #endif
                << " - " << e.what() << std::endl;
            std::cout << "   This is expected if the library is not built yet." << std::endl;
            std::cout << "   Please build the library first." << std::endl;
            return 1;
        }
        
        // Step 3: Load the ONNX model
        std::cout << "3. Loading ONNX model..." << std::endl;
        std::string model_path = "thresholded_relu.onnx";
        
        Ort::Session session(env, 
        #ifdef _WIN32
            std::wstring(model_path.begin(), model_path.end()).c_str(),
        #else
            model_path.c_str(),
        #endif
            session_options);
        
        std::cout << "   Model loaded successfully" << std::endl;
        
        // Step 4: Get input and output info
        std::cout << "4. Getting model input/output information..." << std::endl;
        
        // Get input info
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        std::vector<std::string> input_names;
        std::vector<std::vector<int64_t>> input_shapes;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            // Get input name
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(std::string(input_name.get()));
            
            // Get input shape
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            input_shapes.push_back(shape);
            
            std::cout << "   Input " << i << ": " << input_names[i] << " shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Get output info
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<std::string> output_names;
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(std::string(output_name.get()));
            std::cout << "   Output " << i << ": " << output_names[i] << std::endl;
        }
        
        // Step 5: Prepare input data
        std::cout << "5. Preparing input data..." << std::endl;
        
        // Create test input data: [0.5, 1.5, -0.5, 2.0, 0.0, 1.0, -1.0, 3.0, 0.8, 1.2]
        // With default alpha=1.0, expected output: [0.0, 1.5, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.2]
        std::vector<float> input_data = {0.5f, 1.5f, -0.5f, 2.0f, 0.0f, 1.0f, -1.0f, 3.0f, 0.8f, 1.2f};
        std::vector<int64_t> input_shape = input_shapes[0];
        
        // Verify input size matches expected size
        int64_t input_size = 1;
        for (auto dim : input_shape) {
            input_size *= dim;
        }
        
        if (input_data.size() != static_cast<size_t>(input_size)) {
            std::cout << "   Adjusting input data size to match model (" << input_size << " elements)" << std::endl;
            input_data.resize(input_size, 1.0f);  // Fill with 1.0f if needed
        }
        
        std::cout << "   Input data: [";
        for (size_t i = 0; i < input_data.size(); i++) {
            std::cout << input_data[i];
            if (i < input_data.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Step 6: Create input tensor
        std::cout << "6. Creating input tensor..." << std::endl;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(
            Ort::Value::CreateTensor<float>(
                memory_info,
                input_data.data(),
                input_data.size(),
                input_shape.data(),
                input_shape.size()
            )
        );
        
        // Step 7: Run inference
        std::cout << "7. Running inference..." << std::endl;
        
        // Convert string names to const char* for the API
        std::vector<const char*> input_names_cstr;
        std::vector<const char*> output_names_cstr;
        
        for (const auto& name : input_names) {
            input_names_cstr.push_back(name.c_str());
        }
        for (const auto& name : output_names) {
            output_names_cstr.push_back(name.c_str());
        }
        
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_cstr.data(),
            output_names.size()
        );
        
        std::cout << "   Inference completed successfully" << std::endl;
        
        // Step 8: Process and display results
        std::cout << "8. Processing results..." << std::endl;
        
        if (!output_tensors.empty()) {
            auto& output_tensor = output_tensors[0];
            
            // Get output data
            const float* output_data = output_tensor.GetTensorData<float>();
            auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
            
            // Calculate output size
            int64_t output_size = 1;
            for (auto dim : output_shape) {
                output_size *= dim;
            }
            
            std::cout << "   Output shape: [";
            for (size_t i = 0; i < output_shape.size(); i++) {
                std::cout << output_shape[i];
                if (i < output_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            std::cout << "   Output data: [";
            for (int64_t i = 0; i < output_size; i++) {
                std::cout << output_data[i];
                if (i < output_size - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            // Verify results (with alpha=1.0, values > 1.0 should pass through, others should be 0)
            std::cout << "9. Verifying results..." << std::endl;
            bool test_passed = true;
            
            for (int64_t i = 0; i < std::min(static_cast<int64_t>(input_data.size()), output_size); i++) {
                float expected = (input_data[i] > 1.0f) ? input_data[i] : 0.0f;
                if (std::abs(output_data[i] - expected) > 1e-6f) {
                    std::cout << "   MISMATCH at index " << i << ": expected " << expected 
                              << ", got " << output_data[i] << std::endl;
                    test_passed = false;
                }
            }
            
            if (test_passed) {
                std::cout << "   ✓ Test PASSED - ThresholdedRelu working correctly!" << std::endl;
            } else {
                std::cout << "   ✗ Test FAILED - Results don't match expected values" << std::endl;
                return 1;
            }
        } else {
            std::cout << "   Error: No output tensors received" << std::endl;
            return 1;
        }
        
        std::cout << "=== Test completed successfully ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}