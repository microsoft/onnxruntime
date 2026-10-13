/*
Fixed version of the user's original test code.

Key fixes applied:
1. Simplified registration - use only one method
2. Fixed model file name and path
3. Fixed input data - provide only 1 input (not 2)
4. Removed unnecessary CustomOpConfigs
5. Added proper error handling
6. Fixed data types and shapes
*/

#include <onnxruntime_lite_custom_op.h>
#include <iostream>
#include <vector>
using namespace Ort::Custom;

int main() {
    try {
        Ort::Env env;
        Ort::SessionOptions session_options;
        
        // FIXED: Simplified registration - use only RegisterCustomOpsLibrary
        // Remove the manual domain creation and CustomOpConfigs
        #ifdef _WIN32
            session_options.RegisterCustomOpsLibrary(L"fixed_thresholded_relu_op.dll");
        #else
            session_options.RegisterCustomOpsLibrary("./libfixed_thresholded_relu_op.so");
        #endif
        
        // FIXED: Use the correct model file
        #ifdef _WIN32
            Ort::Session session(env, L"fixed_thresholded_relu.onnx", session_options);
        #else
            Ort::Session session(env, "fixed_thresholded_relu.onnx", session_options);
        #endif

        // Get input/output names
        using AllocatedStringPtr = std::unique_ptr<char, Ort::detail::AllocatedFree>;
        std::vector<AllocatedStringPtr> inputNodeNameAllocatedStrings;
        std::vector<AllocatedStringPtr> outputNodeNameAllocatedStrings;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        Ort::AllocatorWithDefaultOptions allocator;

        size_t numInputNodes = session.GetInputCount();
        for (size_t i = 0; i < numInputNodes; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            inputNodeNameAllocatedStrings.push_back(std::move(name));
            input_names.emplace_back(inputNodeNameAllocatedStrings.back().get());
        }
        
        size_t numOutputNodes = session.GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            outputNodeNameAllocatedStrings.push_back(std::move(name));
            output_names.emplace_back(outputNodeNameAllocatedStrings.back().get());
        }

        Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

        // FIXED: Create only 1 input tensor (not 2)
        std::vector<Ort::Value> input_tensors;
        std::vector<int64_t> input_shape{10};
        
        // FIXED: Create test data that will demonstrate ThresholdedRelu with alpha=1.0
        // Values > 1.0 should pass through, values <= 1.0 should become 0
        std::vector<float> input_data{0.5, 1.5, -0.5, 2.0, 0.0, 1.0, -1.0, 3.0, 0.8, 1.2};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            info, 
            input_data.data(), 
            input_data.size(), 
            input_shape.data(), 
            input_shape.size()
        );
        input_tensors.emplace_back(std::move(input_tensor));

        std::cout << "Start run..." << std::endl;
        
        // FIXED: Use correct input/output counts
        int input_size = 1;  // FIXED: Was 2
        int output_size = 1;
        
        std::vector<Ort::Value> ort_output = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            input_tensors.data(), 
            input_size, 
            output_names.data(), 
            output_size
        );

        std::cout << "Input data: ";
        for (auto& it : input_data) {
            std::cout << it << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Output data: ";
        for (size_t i = 0; i < ort_output.size(); i++) {
            const float* rst = ort_output[i].GetTensorMutableData<float>();
            size_t len = ort_output[i].GetTensorTypeAndShapeInfo().GetElementCount();
            for (size_t j = 0; j != len; j++) {
                std::cout << rst[j] << " ";
            }
            std::cout << std::endl;
        }
        
        // FIXED: Add verification of results
        std::cout << "Expected (ThresholdedRelu with alpha=1.0): ";
        for (auto& val : input_data) {
            std::cout << ((val > 1.0f) ? val : 0.0f) << " ";
        }
        std::cout << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}