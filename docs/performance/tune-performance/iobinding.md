---
title: I/O Binding
grand_parent: Performance
parent: Tune performance
nav_order: 5
---

# I/O Binding

When working with non-CPU execution providers, it's most efficient to have inputs (and/or outputs) arranged on the target device (abstracted by the execution provider used) prior to executing the graph (calling `Run()`). When the input is not copied to the target device, ORT copies it from the CPU as part of the `Run()` call. Similarly, if the output is not pre-allocated on the device, ORT assumes that the output is requested on the CPU and copies it from the device as the last step of the `Run()` call. This eats into the execution time of the graph, misleading users into thinking ORT is slow when the majority of the time is spent in these copies. 

To address this, we've introduced the notion of IOBinding. The key idea is to arrange for inputs to be copied to the device and for outputs to be pre-allocated on the device prior to calling `Run()`. IOBinding is available in all our language bindings. 

Following are code snippets in various languages demonstrating the usage of this feature.

* C++
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

* Python (see [Python API docs](https://onnxruntime.ai/docs/api/python))

* C# (see [OrtIoBindingAllocationTest.cs](https://github.com/microsoft/onnxruntime/blob/main/csharp/test/Microsoft.ML.OnnxRuntime.Tests.Common/OrtIoBindingAllocationTest.cs))

