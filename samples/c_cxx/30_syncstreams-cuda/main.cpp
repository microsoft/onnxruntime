// Minimal C++ example for using CopyTensors EP agnostically and using syncstream
// Model taken from : https://github.com/yakhyo/fast-neural-style-transfer under MIT license
// Goals:
//   - Avoid serial CPU <-> GPU transfers at each inference.
//

#include <cstdlib>
#include <exception>

#include <cuda_runtime.h>
#include <onnxruntime/core/graph/constants.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_run_options_config_keys.h>
#include <onnxruntime/core/session/onnxruntime_session_options_config_keys.h>
#include <stdio.h>

#include "utils.h"

using StreamUniquePtr = std::unique_ptr<OrtSyncStream, std::function<void(OrtSyncStream*)>>;
using OrtFileString = std::basic_string<ORTCHAR_T>;

static OrtFileString toOrtFileString(const std::filesystem::path& path) {
  std::string string(path.string());
  return {string.begin(), string.end()};
}

// The dimensions of the image file we are loading from disk
constexpr int LOADED_IMAGE_DIM = 1080;
// The dimensions of the sub-region we will run inference on. Using whole image for inference.
constexpr int INFERENCE_IMAGE_DIM = 1080;

// Use pinned (page-locked) memory for the large input buffer to enable true async HtoD copies
// The output buffer does not need to be pinned
std::vector<float> cpuOutputFloat(3 * INFERENCE_IMAGE_DIM * INFERENCE_IMAGE_DIM);

int main() {
  try {
    OrtApi const& ortApi = Ort::GetApi();
    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "HelloOrtNv");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    CHECK_ORT(ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", 1));

    std::string trtLibPath = get_executable_parent_path() / DLL_NAME("onnxruntime_providers_nv_tensorrt_rtx");
    CHECK_ORT(
        ortApi.RegisterExecutionProviderLibrary(ortEnvironment, "NvTensorRtRtx", toOrtFileString(trtLibPath).c_str()));

    std::string cudaLibPath = get_executable_parent_path() / DLL_NAME("onnxruntime_providers_cuda");
    if (std::filesystem::is_regular_file(cudaLibPath)) {
      try {
        CHECK_ORT(ortApi.RegisterExecutionProviderLibrary(ortEnvironment, "Cuda", toOrtFileString(cudaLibPath).c_str()));
      } catch (std::exception& ex) {
        LOG("Failed to load Cuda execution provider.");
      }
    }
    sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_ep_devices;
    CHECK_ORT(ortApi.GetEpDevices(ortEnvironment, &ep_devices, &num_ep_devices));
    const OrtEpDevice* trt_ep_device = nullptr;
    for (uint32_t i = 0; i < num_ep_devices; i++) {
      if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]), onnxruntime::kNvTensorRTRTXExecutionProvider) == 0) {
        trt_ep_device = ep_devices[i];
        break;
      }
    }
    if (trt_ep_device == nullptr) {
      LOG("Error: could not select TensorRT RTX execution provider!");
      return EXIT_FAILURE;
    }

    OrtSyncStream* stream = nullptr;
    StreamUniquePtr stream_ptr;
    OrtSyncStream* upload_stream = nullptr;
    StreamUniquePtr upload_stream_ptr;
    CHECK_ORT(ortApi.CreateSyncStreamForEpDevice(trt_ep_device, nullptr, &stream));
    CHECK_ORT(ortApi.CreateSyncStreamForEpDevice(trt_ep_device, nullptr, &upload_stream));
    stream_ptr = StreamUniquePtr(stream, [ortApi](OrtSyncStream* stream) { ortApi.ReleaseSyncStream(stream); });
    upload_stream_ptr = StreamUniquePtr(
        upload_stream, [ortApi](OrtSyncStream* upload_stream) { ortApi.ReleaseSyncStream(upload_stream); });

    size_t stream_addr_val = reinterpret_cast<size_t>(ortApi.SyncStream_GetHandle(stream));
    auto streamAddress = std::to_string(stream_addr_val);
    const char* option_keys[] = {"user_compute_stream", "has_user_compute_stream"};
    const char* option_values[] = {streamAddress.c_str(), "1"};
    for (size_t i = 0; i < num_ep_devices; i++) {
      if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]), onnxruntime::kCpuExecutionProvider) != 0)
        CHECK_ORT(ortApi.SessionOptionsAppendExecutionProvider_V2(sessionOptions, ortEnvironment, &ep_devices[i], 1,
                                                                  option_keys, option_values, 2));
    }

    Ort::Session session(ortEnvironment, toOrtFileString(get_executable_parent_path() / "candy.onnx").c_str(),
                         sessionOptions);

    Ort::MemoryInfo pinned_memory_info("CudaPinned", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemType::OrtMemTypeCPU);
    Ort::Allocator pinned_allocator(session, pinned_memory_info);

    const size_t input_buffer_elements = 3 * LOADED_IMAGE_DIM * LOADED_IMAGE_DIM;
    const size_t input_buffer_size = input_buffer_elements * sizeof(float);

    auto deleter = [&](void* p) { pinned_allocator.Free(p); };
    std::unique_ptr<void, decltype(deleter)> pinned_buffer(pinned_allocator.Alloc(input_buffer_size), deleter);
    float* cpuInputFloat = static_cast<float*>(pinned_buffer.get());

    size_t num_inputs = session.GetInputCount();
    const OrtEpDevice* session_epDevices = {nullptr};
    CHECK_ORT(ortApi.SessionGetEpDeviceForInputs(session, &session_epDevices, num_inputs));

    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;

    Ort::AllocatorWithDefaultOptions cpu_allocator;
    Ort::AllocatedStringPtr InputTensorName = session.GetInputNameAllocated(0, cpu_allocator);
    Ort::AllocatedStringPtr OutputTensorName = session.GetOutputNameAllocated(0, cpu_allocator);

    loadInputImage(cpuInputFloat, (char*)((get_executable_parent_path() / "Input.png").c_str()), false);

    std::vector<int64_t> full_shape{1, 3, LOADED_IMAGE_DIM, LOADED_IMAGE_DIM};
    std::vector<int64_t> inference_shape{1, 3, INFERENCE_IMAGE_DIM, INFERENCE_IMAGE_DIM};

    Ort::Value full_cpu_tensor = Ort::Value::CreateTensor<float>(
        pinned_memory_info, cpuInputFloat, input_buffer_elements, full_shape.data(), full_shape.size());

    Ort::Value inference_cpu_output =
        Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), cpuOutputFloat.data(), cpuOutputFloat.size(),
                                        inference_shape.data(), inference_shape.size());

    OrtMemoryInfo* input_memory_info_agnostic = nullptr;
    const OrtHardwareDevice* hw_device = ortApi.EpDevice_Device(session_epDevices);
    uint32_t vID = ortApi.HardwareDevice_VendorId(hw_device);
    CHECK_ORT(ortApi.CreateMemoryInfo_V2("Input_Agnostic", OrtMemoryInfoDeviceType_GPU, /*vendor_id*/ vID,
                                         /*device_id*/ 0, OrtDeviceMemoryType_DEFAULT, /*default alignment*/ 0,
                                         OrtArenaAllocator, &input_memory_info_agnostic));
    const OrtMemoryInfo* mem_info = input_memory_info_agnostic;

    OrtAllocator* gpu_allocator = nullptr;
    CHECK_ORT(ortApi.GetSharedAllocator(ortEnvironment, mem_info, &gpu_allocator));

    Ort::Value full_gpu_tensor = Ort::Value::CreateTensor<float>(gpu_allocator, full_shape.data(), full_shape.size());
    Ort::Value inference_gpu_input_tensor =
        Ort::Value::CreateTensor<float>(gpu_allocator, inference_shape.data(), inference_shape.size());
    Ort::Value inference_gpu_output_tensor =
        Ort::Value::CreateTensor<float>(gpu_allocator, inference_shape.data(), inference_shape.size());

    void* cuda_compute_stream_handle = ortApi.SyncStream_GetHandle(stream);

    const OrtSyncStreamImpl* uploadStreamImpl;
    OrtSyncNotificationImpl* uploadNotification;
    OrtEpApi ortEpApi = *ortApi.GetEpApi();
    uploadStreamImpl = ortEpApi.SyncStream_GetImpl(upload_stream);
    CHECK_ORT(
        uploadStreamImpl->CreateNotification(const_cast<OrtSyncStreamImpl*>(uploadStreamImpl), &uploadNotification));

    // This should now be a truly asynchronous copy because the source (cpuInputFloat) is pinned memory.
    std::vector<const OrtValue*> cpu_src_ptrs = {full_cpu_tensor};
    std::vector<OrtValue*> gpu_dst_ptrs = {full_gpu_tensor};
    CHECK_ORT(ortApi.CopyTensors(ortEnvironment, cpu_src_ptrs.data(), gpu_dst_ptrs.data(), upload_stream,
                                 cpu_src_ptrs.size()));

    CHECK_ORT(uploadNotification->Activate(uploadNotification));
    CHECK_ORT(uploadNotification->WaitOnDevice(uploadNotification, stream));

    // This D2D copy is on a different stream and will race with the HtoD copy above.
    const float* full_gpu_ptr = full_gpu_tensor.GetTensorData<float>();
    float* inference_gpu_ptr = inference_gpu_input_tensor.GetTensorMutableData<float>();

    for (int c = 0; c < 3; ++c) {
      const float* channel_src_start = full_gpu_ptr + c * (LOADED_IMAGE_DIM * LOADED_IMAGE_DIM);
      float* channel_dst_start = inference_gpu_ptr + c * (INFERENCE_IMAGE_DIM * INFERENCE_IMAGE_DIM);

      const float* slice_src_start = channel_src_start + (LOADED_IMAGE_DIM - INFERENCE_IMAGE_DIM) * LOADED_IMAGE_DIM +
                                     (LOADED_IMAGE_DIM - INFERENCE_IMAGE_DIM);

      CHECK_CUDA(cudaMemcpy2DAsync(channel_dst_start, INFERENCE_IMAGE_DIM * sizeof(float), slice_src_start,
                                   LOADED_IMAGE_DIM * sizeof(float), INFERENCE_IMAGE_DIM * sizeof(float),
                                   INFERENCE_IMAGE_DIM, cudaMemcpyDeviceToDevice,
                                   static_cast<cudaStream_t>(cuda_compute_stream_handle)));
    }

    input_tensors.push_back(std::move(inference_gpu_input_tensor));
    output_tensors.push_back(std::move(inference_gpu_output_tensor));

    Ort::IoBinding iobinding(session);
    iobinding.BindInput(InputTensorName.get(), input_tensors[0]);
    iobinding.BindOutput(OutputTensorName.get(), output_tensors[0]);

    session.Run(Ort::RunOptions{}, iobinding);

    std::vector<const OrtValue*> output_src_tensor_ptrs = {output_tensors[0]};
    std::vector<OrtValue*> output_dst_tensor_ptrs = {inference_cpu_output};
    CHECK_ORT(ortApi.CopyTensors(ortEnvironment, output_src_tensor_ptrs.data(), output_dst_tensor_ptrs.data(),
                                 upload_stream, 1));

    saveOutputImage(cpuOutputFloat.data(), (char*)((get_executable_parent_path() / "output.png").c_str()), false);

    uploadNotification->Release(uploadNotification);
    ortApi.ReleaseMemoryInfo(input_memory_info_agnostic);
  } catch (const Ort::Exception& e) {
    printf("ONNX Runtime exception caught: %s\n", e.what());
    return -1;
  } catch (const std::exception& e) {
    printf("Runtime exception caught: %s\n", e.what());
    return -1;
  }

  return 0;
}
