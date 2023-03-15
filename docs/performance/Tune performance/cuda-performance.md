---
title: CUDA performance
grand_parent: Performance
parent: Model Optimization
nav_order: 4
---

# CUDA Performance

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## IOBinding

When working with non-CPU execution providers, it's most efficient to have inputs (and/or outputs) arranged on the target device (abstracted by the execution provider used) prior to executing the graph (calling `Run()`). When the input is not copied to the target device, ORT copies it from the CPU as part of the `Run()` call. Similarly, if the output is not pre-allocated on the device, ORT assumes that the output is requested on the CPU and copies it from the device as the last step of the `Run()` call. This eats into the execution time of the graph, misleading users into thinking ORT is slow when the majority of the time is spent in these copies. 

To address this, we've introduced the notion of IOBinding. The key idea is to arrange for inputs to be copied to the device and for outputs to be pre-allocated on the device prior to calling `Run()`. IOBinding is available in all our language bindings. 

Following are code snippets in various languages demonstrating the usage of this feature.

* <details><summary>C++</summary>

    ```c++
    Ort::Env env;
    Ort::Session session(env, model_path, session_options);
    Ort::IoBinding io_binding{session};
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    io_binding.BindInput("input1", input_tensor);
    Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0,
                                    OrtMemTypeDefault};
    // Use this to bind output to a device when the shape is not known in advance. If the shape is known you can use the other overload of this function that takes an Ort::Value as input (IoBinding::BindOutput(const char* name, const Value& value)).
    // This internally calls the BindOutputToDevice C API.

    io_binding.BindOutput("output1", output_mem_info);
    session.Run(run_options, io_binding);
    ```
    </details>


* Python (see [Python API docs](https://onnxruntime.ai/docs/api/python))

* C# (see [OrtIoBindingAllocationTest.cs](https://github.com/microsoft/onnxruntime/blob/main/csharp/test/Microsoft.ML.OnnxRuntime.Tests.Common/OrtIoBindingAllocationTest.cs))

## Convolution-heavy models

ORT leverages CuDNN for convolution operations and the first step in this process is to determine which "optimal" convolution algorithm to use while performing the convolution operation for the given input configuration (input shape, filter shape, etc.) in each `Conv` node. This sub-step involves querying CuDNN for a "workspace" memory size and have this allocated so that CuDNN can use this auxiliary memory while determining the "optimal" convolution algorithm to use. 

By default, ORT clamps the workspace size to 32 MB which may lead to a sub-optimal convolution algorithm getting picked by CuDNN. To allow ORT to allocate the maximum possible workspace as determined by CuDNN, a provider option named `cudnn_conv_use_max_workspace` needs to get set (as shown below). 

Keep in mind that using this flag may increase the peak memory usage by a factor (sometimes a few GBs) but this does help CuDNN pick the best convolution algorithm for the given input. We have found that this is an important flag to use while using an fp16 model as this allows CuDNN to pick tensor core algorithms for the convolution operations (if the hardware supports tensor core operations). This flag may or may not result in performance gains for other data types (`float` and `double`).

* <details><summary>Python</summary>

    ```python
    providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession("my_conv_heavy_fp16_model.onnx", sess_options=sess_options, providers=providers)
    ```
    </details>

* <details><summary>C/C++</summary>

    ```c++
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    CreateCUDAProviderOptions(&cuda_options);

    std::vector<const char*> keys{"cudnn_conv_use_max_workspace"};
    std::vector<const char*> values{"1"};

    UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);

    OrtSessionOptions* session_options = /* ... */;
    SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);

    // Finally, don't forget to release the provider options
    ReleaseCUDAProviderOptions(cuda_options);
    ```
    </details>

* <details><summary>C#</summary>

    ```csharp
    var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

    var providerOptionsDict = new Dictionary<string, string>();
    providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";

    cudaProviderOptions.UpdateOptions(providerOptionsDict);

    SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);  // Dispose this finally
    ```
    </details>

## Convolution Input Padding

ORT leverages CuDNN for convolution operations. While CuDNN only takes 4-D or 5-D tensor as input for convolution operations, dimension padding is needed if the input is 3-D tensor. Given an input tensor of shape [N, C, D], it can be padded to [N, C, D, 1] or [N, C, 1, D]. While both of these two padding ways produce same output, the performance may be a lot different because different convolution algorithms are selected, especially on some devices such as A100. By default the input is padded to [N, C, D, 1]. A provider option named `cudnn_conv1d_pad_to_nc1d` needs to get set (as shown below) if [N, C, 1, D] is preferred.

* <details><summary>Python</summary>

    ```python
    providers = [("CUDAExecutionProvider", {"cudnn_conv1d_pad_to_nc1d": '1'})]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession("my_conv_model.onnx", sess_options=sess_options, providers=providers)
    ```
</details>

* <details><summary>C/C++</summary>

    ```c++
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    CreateCUDAProviderOptions(&cuda_options);

    std::vector<const char*> keys{"cudnn_conv1d_pad_to_nc1d"};
    std::vector<const char*> values{"1"};

    UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);

    OrtSessionOptions* session_options = /* ... */;
    SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);

    // Finally, don't forget to release the provider options
    ReleaseCUDAProviderOptions(cuda_options);
    ```
    </details>

* <details><summary>C#</summary>

    ```csharp
    var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

    var providerOptionsDict = new Dictionary<string, string>();
    providerOptionsDict["cudnn_conv1d_pad_to_nc1d"] = "1";

    cudaProviderOptions.UpdateOptions(providerOptionsDict);

    SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);  // Dispose this finally
    ```
    </details>

## Using CUDA Graphs (Preview)

While using the CUDA EP, ORT supports the usage of [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) to remove CPU overhead associated with launching CUDA kernels sequentially. To enable the usage of CUDA Graphs, use the provider option as shown in the samples below.
Currently, there are some constraints with regards to using the CUDA Graphs feature:

* Models with control-flow ops (i.e. `If`, `Loop` and `Scan` ops) are not supported.

* Usage of CUDA Graphs is limited to models where-in all the model ops (graph nodes) can be partitioned to the CUDA EP.

* The input/output types of models need to be tensors.

* Shapes of inputs/outputs cannot change across inference calls. Dynamic shape models are supported - the only constraint is that the input/output shapes should be the same across all inference calls.

* By design, [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) is designed to read from/write to the same CUDA virtual memory addresses during the graph replaying step as it does during the graph capturing step. Due to this requirement, usage of this feature requires using IOBinding so as to bind memory which will be used as input(s)/output(s) for the CUDA Graph machinery to read from/write to (please see samples below).

* While updating the input(s) for subsequent inference calls, the fresh input(s) need to be copied over to the corresponding CUDA memory location(s) of the bound `OrtValue` input(s) (please see samples below to see how this can be achieved). This is due to the fact that the "graph replay" will require reading inputs from the same CUDA virtual memory addresses.

* Multi-threaded usage is currently not supported, i.e. `Run()` MAY NOT be invoked on the same `InferenceSession` object from multiple threads while using CUDA Graphs.

NOTE: The very first `Run()` performs a variety of tasks under the hood like making CUDA memory allocations, capturing the CUDA graph for the model, and then performing a graph replay to ensure that the graph runs. Due to this, the latency associated with the first `Run()` is bound to be high. Subsequent `Run()`s only perform graph replays of the graph captured and cached in the first `Run()`. 


* <details><summary>Python</summary>

    ```python
    providers = [("CUDAExecutionProvider", {"enable_cuda_graph": '1'})]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession("my_model.onnx", sess_options=sess_options, providers=providers)

    providers = [("CUDAExecutionProvider", {'enable_cuda_graph': True})]
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
    y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0)

    session = onnxrt.InferenceSession("matmul_2.onnx", providers=providers)
    io_binding = session.io_binding()

    # Bind the input and output
    io_binding.bind_ortvalue_input('X', x_ortvalue)
    io_binding.bind_ortvalue_output('Y', y_ortvalue)

    # One regular run for the necessary memory allocation and cuda graph capturing
    session.run_with_iobinding(io_binding)
    expected_y = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
    np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

    # After capturing, CUDA graph replay happens from this Run onwards
    session.run_with_iobinding(io_binding)
    np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

    # Update input and then replay CUDA graph with the updated input
    x_ortvalue.update_inplace(np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32))
    session.run_with_iobinding(io_binding)
    ```
</details>

* <details>
  <summary>C/C++</summary>
      ```c++
    const auto& api = Ort::GetApi();

    struct CudaMemoryDeleter {
    explicit CudaMemoryDeleter(const Ort::Allocator* alloc) {
        alloc_ = alloc;
    }

    void operator()(void* ptr) const {
        alloc_->Free(ptr);
    }
    
    const Ort::Allocator* alloc_;
    };
    
    // Enable cuda graph in cuda provider option.
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    api.CreateCUDAProviderOptions(&cuda_options);
    std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
    std::vector<const char*> keys{"enable_cuda_graph"};
    std::vector<const char*> values{"1"};
    api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1);

    Ort::SessionOptions session_options;
    api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(session_options), rel_cuda_options.get();


    // Create IO bound inputs and outputs.
    Ort::Session session(*ort_env, ORT_TSTR("matmul_2.onnx"), session_options);
    Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::Allocator cuda_allocator(session, info_cuda);

    const std::array<int64_t, 2> x_shape = {3, 2};
    std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(x_values.size() * sizeof(float)),
                                                            CudaMemoryDeleter(&cuda_allocator));
    cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

    // Create an OrtValue tensor backed by data on CUDA memory
    Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

    const std::array<int64_t, 2> expected_y_shape = {3, 2};
    std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
    auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(expected_y.size() * sizeof(float)),
                                                                CudaMemoryDeleter(&cuda_allocator));

    // Create an OrtValue tensor backed by data on CUDA memory
    Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

    Ort::IoBinding binding(session);
    binding.BindInput("X", bound_x);
    binding.BindOutput("Y", bound_y);

    // One regular run for necessary memory allocation and graph capturing
    session.Run(Ort::RunOptions(), binding);

    // After capturing, CUDA graph replay happens from this Run onwards
    session.Run(Ort::RunOptions(), binding);

    // Update input and then replay CUDA graph with the updated input
    x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);
    session.Run(Ort::RunOptions(), binding);
    ```
</details>

* C# (future)
