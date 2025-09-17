#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <filesystem>
#include <onnxruntime/core/graph/constants.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_run_options_config_keys.h>
#include <onnxruntime/core/session/onnxruntime_session_options_config_keys.h>
#include <onnxruntime/core/providers/nv_tensorrt_rtx/nv_provider_options.h>

namespace fs = std::filesystem;

// Utility: read file into buffer
std::vector<uint8_t> LoadFileToBuffer(const fs::path& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open file: " + filename.string());

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filename.string());
    }
    return buffer;
}

// Utility for timing
template <typename Func>
double MeasureTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();  // seconds
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
            << " <input_model.onnx> <weights_file.onnx.data> <compiled_model_output.onnx> <external_data_filename> [embed_mode] [provider]"
            << std::endl;
        std::cerr << "  external_data_filename: The name used for external data in the model (e.g., 'model.onnx.data')" << std::endl;
        return 1;
    }

    fs::path input_model_path = argv[1];
    fs::path weights_model_path = argv[2];
    fs::path output_model_path = argv[3];
    fs::path external_data_filename_path = argv[4];
    int embed_mode = 0;
    std::string provider = onnxruntime::kNvTensorRTRTXExecutionProvider;

    if (argc >= 6) {
        embed_mode = std::stoi(argv[5]);
        if (embed_mode != 0 && embed_mode != 1) {
            std::cerr << "Invalid embed_mode value. Must be 0 or 1." << std::endl;
            return 1;
        }
    }

    if (argc >= 7) {
        provider = std::string(argv[6]);
        try {
            Ort::SessionOptions test_options;
            // A simple pre-check to see if the provider is available
            test_options.AppendExecutionProvider(provider.c_str());
        }
        catch (const Ort::Exception& ex) {
            std::cerr << "ERROR: Provider '" << provider
                << "' is not available or invalid: "
                << ex.what() << std::endl;
            return 1; // EXIT IMMEDIATELY, no fallback
        }
    }

    std::cout << "> Embed mode set to: " << embed_mode << std::endl;
    std::cout << "> Provider set to: " << provider << std::endl;
    std::cout << "> External data filename: " << external_data_filename_path << std::endl;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EPContextExample");

        Ort::SessionOptions session_options;

        // Configure execution provider based on provider type
        if (provider == onnxruntime::kNvTensorRTRTXExecutionProvider) {
            // Using the new provider option for this specific EP
            std::unordered_map<std::string, std::string> option_map{
                {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, "1"}
            };
            session_options.AppendExecutionProvider(provider.c_str(), option_map);
            std::cout << "> Using Execution Provider: " << provider << " with options." << std::endl;
        }
        else {
            // Fallback for other providers that might not have or need options
            session_options.AppendExecutionProvider(provider.c_str());
            std::cout << "> Using Execution Provider: " << provider << " (no options)." << std::endl;
        }

        // --- Step 1: Load model and weights into buffers ---
        std::vector<uint8_t> model_buffer = LoadFileToBuffer(input_model_path);
        std::vector<uint8_t> weights_buffer = LoadFileToBuffer(weights_model_path);

        // Register external weights explicitly (needed in buffer mode)
        std::vector<std::basic_string<ORTCHAR_T>> file_names = { external_data_filename_path.native() };
        std::vector<char*> file_buffers_data = { static_cast<char*>(static_cast<void*>(weights_buffer.data())) };
        std::vector<size_t> file_buffers_size = { weights_buffer.size() };

        session_options.AddExternalInitializersFromFilesInMemory(
            file_names,
            file_buffers_data,
            file_buffers_size
        );

        // --- Step 2: Regular ONNX load (buffer mode) ---
        std::cout << "> Loading regular onnx (buffer)..." << std::endl;
        double load_time_normal = MeasureTime([&]() {
            Ort::Session session(env, model_buffer.data(), model_buffer.size(), session_options);
            });
        std::cout << "> Session load time: " << load_time_normal << " sec" << std::endl;

        // --- Step 3: Compile model from buffer ---
        std::cout << "> Compiling model (buffer)..." << std::endl;
        void* output_buffer_data = nullptr;
        size_t output_buffer_size = 0;

        // Setup compilation options
        Ort::ModelCompilationOptions compile_options(env, session_options);
        compile_options.SetEpContextEmbedMode(embed_mode);
        compile_options.SetInputModelFromBuffer(model_buffer.data(), model_buffer.size());
        compile_options.SetOutputModelBuffer(
            Ort::AllocatorWithDefaultOptions(),
            &output_buffer_data,
            &output_buffer_size
        );

        // Actual compilation
        double compile_time = MeasureTime([&]() {
            Ort::Status status = Ort::CompileModel(env, compile_options);
            if (!status.IsOK()) {
                throw Ort::Exception(status.GetErrorMessage(), ORT_FAIL);
            }
            });
        std::cout << "> Compiled successfully!" << std::endl;
        std::cout << "> Compile time: " << compile_time << " sec" << std::endl;
        std::cout << "> Compiled model buffer size: " << output_buffer_size << " bytes" << std::endl;

        // --- Step 4: Load compiled model from buffer ---
        std::cout << "> Loading EP context model (buffer)..." << std::endl;
        double load_time_compiled = MeasureTime([&]() {
            Ort::Session compiled_session(env,
                reinterpret_cast<uint8_t*>(output_buffer_data),
                output_buffer_size,
                session_options);
            });
        std::cout << "> Session load time: " << load_time_compiled << " sec" << std::endl;

        // Note: free output_buffer_data if allocator requires it
    }
    catch (const Ort::Exception& ex) {
        std::cerr << "ONNX Runtime error: " << ex.what() << std::endl;
        return 1;
    }
    catch (const std::exception& ex) {
        std::cerr << "Standard exception: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
