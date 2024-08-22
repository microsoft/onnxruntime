#include <memory>
#include <fstream>
#include <cuda_runtime.h>
#include "tensorrt_execution_provider.h"
#include "onnx_ctx_model_helper.h"
namespace onnxruntime {

TensorrtLogger& GetTensorrtLogger(bool verbose_log) {
  const auto log_level = verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING;
  static TensorrtLogger trt_logger(log_level);
  if (log_level != trt_logger.get_level()) {
    trt_logger.set_level(verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
  }
  return trt_logger;
}

const OrtApi* TensorrtExecutionProvider::api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);

TensorrtExecutionProvider::TensorrtExecutionProvider(const char* ep_type, const ProviderOptions& ep_info) : OrtExecutionProvider() {
    OrtExecutionProvider::GetCapability = [](const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph*** indexed_sub_graph) {
    };

    OrtExecutionProvider::Compile = [](OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo** node_compute_info) -> OrtStatusPtr {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        TensorrtExecutionProvider* p = static_cast<TensorrtExecutionProvider*>(this_);
        this_->extra_param_for_create_state_func = p;
        this_->extra_param_for_compute_func = p;
        for (size_t j = 0; j < cnt; j++) {
            std::unordered_map<std::string, size_t> input_map, output_map;
            size_t input_size = 0;
            api->OrtNode_GetInputSize(node[j], &input_size);
            for (size_t i = 0; i < input_size; i++) {
                const char* ith_input_name = nullptr;
                api->OrtNode_GetIthInputName(node[j], i, &ith_input_name);
                input_map[ith_input_name] = i;
            }

            size_t output_size = 0;
            api->OrtNode_GetOutputSize(node[j], &output_size);
            for (size_t i = 0; i < output_size; i++) {
                const char* ith_output_name = nullptr;
                api->OrtNode_GetIthOutputName(node[j], i, &ith_output_name);
                output_map[ith_output_name] = i;
            }

            OrtStatusPtr ret = nullptr;
            if (GraphHasCtxNode(graph[j])) {
                ret = p->CreateNodeComputeInfoFromPrecompiledEngine(graph[j], node[j], input_map, output_map, &node_compute_info[j]);
            } else {

            }
            if (ret != nullptr) return api->CreateStatus(api->GetErrorCode(ret), api->GetErrorMessage(ret));
        }
        return nullptr;
    };

    OrtExecutionProvider::CanCopy = [](const OrtDevice* source, const OrtDevice* target) {
      const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
      OrtMemoryInfoDeviceType source_device_type, target_device_type;
      api->DeviceGetDeviceType(source, &source_device_type);
      api->DeviceGetDeviceType(target, &target_device_type);
      OrtMemoryType source_mem_type, target_mem_type;
      api->DeviceGetMemoryType(source, &source_mem_type);
      api->DeviceGetMemoryType(target, &target_mem_type);

      return source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU ||
             source_mem_type == OrtMemoryType::OrtMemoryType_CUDA_PINNED ||
             target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU ||
             target_mem_type == OrtMemoryType::OrtMemoryType_CUDA_PINNED;
    };

    OrtExecutionProvider::CopyTensor = [](const void* src, OrtMemoryInfoDeviceType source_device_type, OrtMemoryType source_mem_type, void* dst, OrtMemoryInfoDeviceType target_device_type, size_t count, void* stream) -> OrtStatusPtr {
        // TODO(leca): convert cudaError_t to OrtStatusPtr
        if (source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU && target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU) {
            if (src != dst) {
                stream ? cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)) : cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
            }
            return nullptr;
        }
        if (source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU && target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU) {
            if (stream) cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
            else {
                cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
                cudaStreamSynchronize(nullptr);
            }
            return nullptr;
        }
        if (source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU && target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU) {
            if (stream) cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
            else {
                cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(nullptr);
            }
            return nullptr;
        }
        if (stream && source_mem_type == OrtMemoryType::OrtMemoryType_CUDA_PINNED) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        }
        memcpy(dst, src, count);
        return nullptr;
    };

    type = ep_type;
    create_stream = new OrtCreateStream();
    create_stream->CreateStreamFunc = [](const OrtDevice* device) -> void* {
        cudaStream_t stream = nullptr;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        return stream;
    };

    api_->CreateDevice(OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU, OrtMemoryType::OrtMemoryType_Default, 0, &default_device);
}

TensorrtExecutionProviderFactory::TensorrtExecutionProviderFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> OrtExecutionProvider* {
        ProviderOptions options;
        for (size_t i = 0; i < option_size; i++) options[ep_option_keys[i]] = ep_option_values[i];
        std::unique_ptr<TensorrtExecutionProvider> ret = std::make_unique<TensorrtExecutionProvider>("TensorrtExecutionProvider", std::move(options));
        return ret.release();
    };
}

OrtStatusPtr TensorrtExecutionProvider::RefitEngine(std::string onnx_model_filename,
                                                      std::string& onnx_model_folder_path,
                                                      std::string& weight_stripped_engine_cath_path,
                                                      bool path_check,
                                                      nvinfer1::ICudaEngine* trt_engine,
                                                      bool serialize_refitted_engine,
                                                      bool detailed_build_log) {
#if NV_TENSORRT_MAJOR >= 10
  std::filesystem::path onnx_model_path{onnx_model_folder_path};
  onnx_model_path.append(onnx_model_filename);
  if (path_check && IsAbsolutePath(onnx_model_path.string())) {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                           std::string("For security purpose, the ONNX model path should be set with "
                           "a relative path, but it is an absolute path: " +
                               onnx_model_path.string()).c_str());
  }
  if (path_check && IsRelativePathToParentPath(onnx_model_path.string())) {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                           "The ONNX model path has '..'. For security purpose, it's not "
                           "allowed to point outside the directory.");
  }

  if (!std::filesystem::exists(onnx_model_path)) {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                           std::string("The ONNX model " + onnx_model_path.string() +
                               " does not exist.").c_str());
  }

  // weight-stripped engine refit logic
  TensorrtLogger& trt_logger = GetTensorrtLogger(detailed_build_log);
  auto refitter = std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(*trt_engine, trt_logger));
  auto parser_refitter = std::unique_ptr<nvonnxparser::IParserRefitter>(
      nvonnxparser::createParserRefitter(*refitter, trt_logger));
  if (!parser_refitter->refitFromFile(onnx_model_path.string().c_str())) {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                           std::string("TensorRT EP's IParserRefitter could not refit deserialized weight-stripped engine with weights contained in: " + onnx_model_path.string()).c_str());
  }
  if (refitter->refitCudaEngine()) {
//    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Successfully refitted the weight-stripped engine.";
  } else {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                           std::string("TensorRT EP's IRefitter could not refit deserialized weight-stripped engine with weights contained in: " + onnx_model_path.string()).c_str());
  }

  // serialize the refitted engine to disk
  if (serialize_refitted_engine) {
    std::string refitted_engine_cache = GetWeightRefittedEnginePath(weight_stripped_engine_cath_path);
    nvinfer1::IHostMemory* serialized_engine = trt_engine->serialize();
    std::ofstream engine_file(refitted_engine_cache, std::ios::binary | std::ios::out);
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
//    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialize the refitted engine to " << refitted_engine_cache;
  }
  return nullptr;
#else
  return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, "TensorRT EP's IParserRefitter can only be used on TRT 10.0 onwards.");
#endif
}

OrtStatusPtr TensorrtExecutionProvider::CreateNodeComputeInfoFromPrecompiledEngine(const OrtGraphViewer* graph_body_viewer, const OrtNode* fused_node,
                                                                           std::unordered_map<std::string, size_t>& input_map,
                                                                           std::unordered_map<std::string, size_t>& output_map,
                                                                           OrtNodeComputeInfo** node_compute_funcs) {
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context;
  std::unordered_map<std::string, size_t> input_indexes;   // TRT engine input name -> ORT kernel context input index
  std::unordered_map<std::string, size_t> output_indexes;  // TRT engine output name -> ORT kernel context output index
  std::unordered_map<std::string, size_t> output_types;    // TRT engine output name -> ORT output tensor type

  // Get engine binary data and deserialize it
  auto trt_cache_model_handler = TensorRTCacheModelHandler(&trt_engine,
                                                           runtime_.get(),
                                                           model_path_,
                                                           compute_capability_,
                                                           weight_stripped_engine_enable_,
                                                           onnx_model_folder_path_,
                                                           detailed_build_log_);
  auto status = trt_cache_model_handler.GetEpContextFromGraph(graph_body_viewer);
  if (status != nullptr) {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, api_->GetErrorMessage(status));
  }

  // Build context
  //
  // Note: Creating an execution context from an engine is thread safe per TRT doc
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
  if (context_memory_sharing_enable_) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    size_t mem_size = trt_engine->getDeviceMemorySize();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (mem_size > max_ctx_mem_size_) {
      max_ctx_mem_size_ = mem_size;
    }
#if NV_TENSORRT_MAJOR < 10
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContextWithoutDeviceMemory());
#else
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
#endif
  } else {
    trt_context = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
  }

  const char* fused_node_name = nullptr;
  api_->OrtNode_GetName(fused_node, &fused_node_name);
  if (!trt_context) {
    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL,
                           std::string("TensorRT EP could not build execution context for fused node: " + std::string(fused_node_name)).c_str());
  }

  // Create input/output to index maps
  for (int32_t i = 0; i < trt_engine->getNbIOTensors(); ++i) {
    auto const& name = trt_engine->getIOTensorName(i);
    auto const& mode = trt_engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      const auto& iter = input_map.find(name);
      if (iter != input_map.end()) {
        input_indexes[name] = iter->second;
      }
    } else {
      const auto& iter = output_map.find(name);
      if (iter != output_map.end()) {
        output_indexes[name] = iter->second;
      }
    }
  }

  // Create output to type map
  size_t graph_output_size = api_->OrtGraph_GetOutputSize(graph_body_viewer);
  for (size_t i = 0; i < graph_output_size; i++) {
    output_types[api_->OrtGraph_GetIthOutputName(graph_body_viewer, i)] = api_->OrtGraph_GetIthOutputElemType(graph_body_viewer, i);
  }

  // Save TRT engine, TRT context and input/output info to map
  engines_.emplace(fused_node_name, std::move(trt_engine));
  contexts_.emplace(fused_node_name, std::move(trt_context));
  input_info_[fused_node_name].push_back(input_indexes);
  output_info_[fused_node_name].push_back(output_indexes);
  output_info_[fused_node_name].push_back(output_types);

  // Create function state
  (*node_compute_funcs)->CreateFunctionStateFunc = [](OrtComputeContext* context, void* extra_param, void** state) -> int {
    TensorrtExecutionProvider* this_ = reinterpret_cast<TensorrtExecutionProvider*>(extra_param);
    std::unique_ptr<TensorrtShortFuncState> p = std::make_unique<TensorrtShortFuncState>();
    *p = { context->AllocateFunc,
           context->DestroyFunc,
           context->allocator_handle,
           context->node_name,
           &(this_->engines_[context->node_name]),
           &(this_->contexts_[context->node_name]),
           this_->input_info_[context->node_name],
           this_->output_info_[context->node_name],
           this_->context_memory_sharing_enable_,
           &this_->max_ctx_mem_size_};
    *state = p.release();
    return 0;
  };

  // Release function state
  (*node_compute_funcs)->DestroyFunctionStateFunc = [](void* state) {
    delete reinterpret_cast<TensorrtShortFuncState*>(state);
  };

  // Create compute function
  (*node_compute_funcs)->ComputeFunc = [](void* state, void* extra_param, const OrtApi* api, OrtKernelContext* context) -> OrtStatusPtr {
    TensorrtExecutionProvider* this_ = reinterpret_cast<TensorrtExecutionProvider*>(extra_param);
    TensorrtShortFuncState* trt_state = reinterpret_cast<TensorrtShortFuncState*>(state);

    // The whole compute_function should be considered the critical section.
    // More details here, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
//    std::lock_guard<OrtMutex> lock(*(trt_state->tensorrt_mu_ptr));

    const std::unordered_map<std::string, size_t>& input_indexes = (trt_state->input_info)[0];
    const std::unordered_map<std::string, size_t>& output_indexes = (trt_state->output_info)[0];
    const std::unordered_map<std::string, size_t>& output_types = (trt_state->output_info)[1];
    auto fused_node_name = trt_state->fused_node_name;
    auto& dds_output_allocator_map = this_->dds_output_allocator_maps_[fused_node_name];
//    auto trt_engine = trt_state->engine->get();
//    auto trt_context = trt_state->context->get();
//    auto max_context_mem_size_ptr = trt_state->max_context_mem_size_ptr;
//    int num_outputs = static_cast<int>(output_indexes.size());
//    std::unordered_map<std::string, std::vector<int32_t>> shape_tensor_values;        // This map holds "shape tensor -> shape values" for the shape tensor input across this inference run
//    std::unordered_map<std::string, std::vector<int64_t>> shape_tensor_values_int64;  // same as above but for int64 shape tensor input
//
//    OrtDevice device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, narrow<OrtDevice::DeviceId>(device_id_));
//    OrtMemoryInfo mem_info("", OrtAllocatorType::OrtDeviceAllocator, device, device_id_);
//    if (alloc_ == nullptr) {
//      Ort::ThrowOnError(api->KernelContext_GetAllocator(context, &mem_info, &alloc_));
//    }
//    OrtAllocator* alloc = alloc_;
//
//    void* cuda_stream;
//    Ort::ThrowOnError(api->KernelContext_GetGPUComputeStream(context, &cuda_stream));
//    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
//
//    // Get input and output binding names
//    int total_bindings = trt_engine->getNbIOTensors();
//    std::vector<char const*> input_binding_names, output_binding_names;
//    for (int i = 0, end = total_bindings; i < end; ++i) {
//      auto const& name = trt_engine->getIOTensorName(i);
//      auto const& mode = trt_engine->getTensorIOMode(name);
//      if (mode == nvinfer1::TensorIOMode::kINPUT) {
//        input_binding_names.push_back(name);
//      } else {
//        output_binding_names.push_back(name);
//      }
//    }
//
//    /*
//     * Set input shapes and bind input buffers
//     */
//    std::vector<IAllocatorUniquePtr<void>> scratch_buffers;
//    for (size_t i = 0, end = input_binding_names.size(); i < end; ++i) {
//      char const* input_name = input_binding_names[i];
//
//      size_t input_index = 0;
//      const auto iter = input_indexes.find(input_name);
//      if (iter != input_indexes.end()) {
//        input_index = iter->second;
//      }
//
//      Status status = BindContextInput(ctx, trt_engine, trt_context, input_name, input_index, shape_tensor_values, shape_tensor_values_int64, scratch_buffers, alloc, stream);
//      if (status != Status::OK()) {
//        return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, status.ErrorMessage());
//      }
//    }
//
//    /*
//     * Set output shapes and bind output buffers
//     */
//    std::unordered_map<char const*, void*> buffers;
//    buffers.reserve(num_outputs);
//    using OutputOrtValue = Ort::UnownedValue;
//    std::unordered_map<size_t, OutputOrtValue> output_tensors;
//    output_tensors.reserve(num_outputs);
//    std::unordered_map<size_t, int> output_dim_sizes;
//    output_dim_sizes.reserve(num_outputs);
//
//    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
//      char const* output_name = output_binding_names[i];
//
//      size_t output_index = 0;
//      const auto& index_iter = output_indexes.find(output_name);
//      if (index_iter != output_indexes.end()) {
//        output_index = index_iter->second;
//      }
//
//      size_t output_type = 0;
//      const auto type_iter = output_types.find(output_name);
//      if (type_iter != output_types.end()) {
//        output_type = type_iter->second;
//      }
//
//      Status status = BindContextOutput(ctx, trt_context, output_name, output_index, output_type, i, output_tensors, output_dim_sizes,
//                                        dds_output_allocator_map, scratch_buffers, alloc, buffers);
//      if (status != Status::OK()) {
//        return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, status.ErrorMessage());
//      }
//    }
//
//    // Set execution context memory
//    if (trt_state->context_memory_sharing_enable) {
//#if defined(_MSC_VER)
//#pragma warning(push)
//#pragma warning(disable : 4996)
//#endif
//      size_t mem_size = trt_engine->getDeviceMemorySize();
//#if defined(_MSC_VER)
//#pragma warning(pop)
//#endif
//      if (mem_size > *max_context_mem_size_ptr) {
//        *max_context_mem_size_ptr = mem_size;
//      }
//      trt_context->setDeviceMemory(IAllocator::MakeUniquePtrFromOrtAllocator<void>(alloc, *max_context_mem_size_ptr).get());
//    }
//
//    // Start CUDA graph capture.
//    // Note: The reason we don't put graph capture in OnRunStart() like CUDA EP does is because
//    // current ORT TRT doesn't get cuda stream until compute time and graph capture requires cuda stream.
//    if (cuda_graph_enable_ && IsGraphCaptureAllowed() && !IsGraphCaptured(0)) {
//      LOGS_DEFAULT(INFO) << "Capturing the cuda graph for this model";
//      cuda_graph_.SetStream(stream);
//      CaptureBegin(0);
//    }
//
//    // Run TRT inference
//    if (!trt_context->enqueueV3(stream)) {
//      return api_->CreateStatus(OrtErrorCode::ORT_FAIL, "TensorRT EP execution context enqueue failed.");
//    }
//
//    /*
//     * Given that InferenceSession::Run() is guaranteed to be thread-safe meaning multiple threads can call this function concurrently,
//     * TRT EP needs to carefully take care of concurrency here, if not, following concurrent issue might happen:
//     *
//     * It's suggested that to perform inference concurrently in multiple streams, use one trt execution context per stream.
//     * In the design of TRT EP (Not apply per-thread context implementation) and if multiple threads are calling InferenceSession::Run() concurrently,
//     * the trt execution context instance is shared by all the threads and each thread aquires different stream from ORT.
//     * So TRT EP will end up having one trt execution context using multiple streams which is not suggested.
//     * But, since the whole compute_func() is protected by the lock and if cudaStreamSynchronize() is enforced here, one trt execution context per stream
//     * is guaranteed.
//     *
//     * Therefore, TRT EP needs to call cudaStreamSynchronize() which means to wait until stream has completed all operations to prevent the concurrent issue mentioned above.
//     * However, if cuda graph is enabled, TRT EP won't call cudaStreamSynchronize() since it's not allowed during graph capture.
//     */
//    if (sync_stream_after_enqueue_) {
//      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
//    }
//
//    // Assign TRT output back to ORT output
//    // (1) Bind TRT DDS output to ORT kernel context output. (It needs to wait until enqueueV3 is finished)
//    // (2) Cast TRT INT32 output to ORT INT64 output or TRT double output to float output
//    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i) {
//      char const* output_name = output_binding_names[i];
//
//      size_t output_type = 0;
//      const auto& iter = output_types.find(output_name);
//      if (iter != output_types.end()) {
//        output_type = iter->second;
//      }
//
//      if (dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end()) {
//        size_t output_index = 0;
//        const auto& index_iter = output_indexes.find(output_name);
//        if (index_iter != output_indexes.end()) {
//          output_index = index_iter->second;
//        }
//        auto status = BindKernelOutput(ctx, &mem_info, dds_output_allocator_map, output_name, output_index, output_type, stream);
//        if (status != Status::OK()) {
//          return api_->CreateStatus(OrtErrorCode::ORT_FAIL, status.ErrorMessage());
//        }
//      } else {
//        auto& output_tensor = output_tensors[i];
//#if NV_TENSORRT_MAJOR < 10
//        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
//          auto output_tensor_ptr = output_tensor.GetTensorMutableData<int64_t>();
//          if (output_tensor_ptr != nullptr) {
//            cuda::Impl_Cast<int32_t, int64_t>(stream, reinterpret_cast<int32_t*>(buffers[output_name]), output_tensor_ptr, output_dim_sizes[i]);
//          }
//        }
//#endif
//        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
//          auto output_tensor_ptr = output_tensor.GetTensorMutableData<double>();
//          if (output_tensor_ptr != nullptr) {
//            cuda::Impl_Cast<float, double>(stream, reinterpret_cast<float*>(buffers[output_name]), output_tensor_ptr, output_dim_sizes[i]);
//          }
//        }
//      }
//    }
//
//    // End CUDA graph capture.
//    // Note: One reason we don't put end of graph capture in OnRunEnd() like CUDA EP does is because of cuda stream mentioned in graph capture
//    // above, another reason is because OnRunEnd() is not synchronized with OnRunStart() and ExecuteGraph() per inference_session.cc.
//    // It's safe to start/end CUDA graph capture in compute_func() here since cuda graph object is maintained by a per thread basis.
//    if (cuda_graph_enable_ && !IsGraphCaptured(0)) {
//      if (IsGraphCaptureAllowed()) {
//        CaptureEnd(0);
//        // CUDA work issued to a capturing stream doesn’t actually run on the GPU,
//        // so run the captured graph here to actually execute the work.
//        ORT_RETURN_IF_ERROR(ReplayGraph(0));
//      } else {
//        IncrementRegularRunCountBeforeGraphCapture();
//      }
//    }

    return nullptr;
  };

  return nullptr;
}

}   // namespace onnxruntime

#ifdef __cplusplus
extern "C" {
#endif
OrtExecutionProviderFactory* RegisterCustomEp() {
    std::unique_ptr<onnxruntime::TensorrtExecutionProviderFactory> ret = std::make_unique<onnxruntime::TensorrtExecutionProviderFactory>();
    return ret.release();
}
#ifdef __cplusplus
}
#endif
