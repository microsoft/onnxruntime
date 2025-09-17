#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <onnxruntime/core/graph/constants.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_run_options_config_keys.h>
#include <onnxruntime/core/session/onnxruntime_session_options_config_keys.h>

/**
 * @brief A generic utility to measure the execution time of a function.
 *
 * This template function measures the time taken to execute a callable object
 * (e.g., a lambda function) and returns the duration in seconds.
 *
 * @tparam Func The type of the function to measure.
 * @param func The function to execute and time.
 * @return The duration of the function's execution in seconds.
 */
template <typename Func>
double MeasureTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int main(int argc, char* argv[]) {
    // Check for correct command-line arguments.
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_model.onnx> <compiled_model_output.onnx> [embed_mode] [provider_name]" << std::endl;
        std::cerr << "  embed_mode: 0 = embed engine in EP context (default), 1 = external engine" << std::endl;
        std::cerr << "  provider_name: The name of the execution provider (e.g., 'NvTensorRtRtx')" << std::endl;
        return 1;
    }

    // Parse command-line arguments.
    std::filesystem::path input_model_path = argv[1];
    std::filesystem::path output_model_path = argv[2];
    int embed_mode = 0; // Default: embed engine
    std::string provider = onnxruntime::kNvTensorRTRTXExecutionProvider; // Default provider

    // Automatically create a runtime cache directory for demonstration.
    const std::filesystem::path runtime_cache_dir = "ort_runtime_cache";

    // Delete existing runtime cache directory to ensure clean performance metrics
    try {
        if (std::filesystem::exists(runtime_cache_dir)) {
            std::filesystem::remove_all(runtime_cache_dir);
        }
    }
    catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "WARNING: Failed to delete runtime cache directory: " << ex.what() << std::endl;
        std::cerr << "Performance metrics may not be accurate due to existing cache." << std::endl;
    }

    if (argc >= 4) {
        try {
            embed_mode = std::stoi(argv[3]);
        }
        catch (const std::invalid_argument&) {
            std::cerr << "ERROR: Invalid embed_mode value. Must be an integer." << std::endl;
            return 1;
        }
        if (embed_mode != 0 && embed_mode != 1) {
            std::cerr << "ERROR: Invalid embed_mode value. Must be 0 or 1." << std::endl;
            return 1;
        }
    }

    if (argc >= 5) {
        provider = argv[4];
    }

    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "ONNX Runtime TensorRT Compilation Example" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "> Input Model Path:  " << input_model_path << std::endl;
    std::cout << "> Output Model Path: " << output_model_path << std::endl;
    std::cout << "> Embed Mode:        " << (embed_mode == 1 ? "Embedded" : "External") << std::endl;
    std::cout << "> Execution Provider: " << provider << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    try {
        // Create an ONNX Runtime environment.
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime-EP-Context-Example");

        // Block for normal onnx load and time it
        {
            // Create session options
            Ort::SessionOptions session_options;
            session_options.AppendExecutionProvider(provider.c_str());

            std::cout << "> Loading original ONNX model from disk..." << std::endl;
            double load_time_normal = MeasureTime([&]() {
                Ort::Session session(env, input_model_path.c_str(), session_options);
            });
            std::cout << "> Original session load time: " << load_time_normal << " sec" << std::endl;
        }

        // Block for AOT compilation and time it
        {
            // Create session options
            Ort::SessionOptions session_options;
            std::unordered_map<std::string, std::string> provider_options;
            provider_options["nv_runtime_cache_path"] = runtime_cache_dir.string();
            session_options.AppendExecutionProvider(provider.c_str(), provider_options);

            std::cout << "> Compiling model with " << provider << "..." << std::endl;

            // Setup compilation options
            Ort::ModelCompilationOptions compile_options(env, session_options);
            compile_options.SetEpContextEmbedMode(embed_mode);
            compile_options.SetInputModelPath(input_model_path.c_str());
            compile_options.SetOutputModelPath(output_model_path.c_str());

            double compile_time = MeasureTime([&]() {
                Ort::Status status = Ort::CompileModel(env, compile_options);
                if (!status.IsOK()) {
                    throw Ort::Exception(status.GetErrorMessage(), ORT_FAIL);
                }
            });
            std::cout << "> Model compiled successfully!" << std::endl;
            std::cout << "> Compile time: " << compile_time << " sec" << std::endl;
            std::cout << "> Compiled model saved at " << output_model_path << std::endl;
        }

        // Block for loading compiled model and time it
        {
            // Create session options
            Ort::SessionOptions session_options;
            session_options.AppendExecutionProvider(provider.c_str());


            std::cout << "> Loading compiled model from disk..." << std::endl;
            double load_time_compiled = MeasureTime([&]() {
                Ort::Session session(env, output_model_path.c_str(), session_options);
            });
            std::cout << "> Context model session load time: " << load_time_compiled << " sec" << std::endl;
        }

        // Block for JIT compilation and time it
        {
            // Create session options
            Ort::SessionOptions session_options;
            std::unordered_map<std::string, std::string> provider_options;
            provider_options["nv_runtime_cache_path"] = runtime_cache_dir.string();
            session_options.AppendExecutionProvider(provider.c_str(), provider_options);

            double jit_time = MeasureTime([&]() {
                Ort::Session session(env, output_model_path.c_str(), session_options);
            });
            std::cout << "> Context model session load time with runtime cache: " << jit_time << " sec" << std::endl;
            std::cout << "> Runtime cache has been populated at: " << runtime_cache_dir << std::endl;
        }
    }
    catch (const Ort::Exception& ex) {
        std::cerr << "\nONNX Runtime Exception: " << ex.what() << std::endl;
        return 1;
    }
    catch (const std::exception& ex) {
        std::cerr << "\nStandard Exception: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "\nProgram finished successfully." << std::endl;
    return 0;
}
