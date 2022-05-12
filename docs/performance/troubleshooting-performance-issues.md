---
title: Tune performance > Troubleshooting
parent: Performance
nav_order: 5
description: Checklist to troubleshoot Onnx Runtime performance tuning issues and Frequently Asked Questions.
redirect_from: /docs/how-to/tune-performance
---
<div class="container">

## Troubleshooting Performance Issues

Troubleshooting Onnx Runtime performance issues may vary depending on the model and usage scenario.

### ORT Performance Troubleshooting Checklist

Here is a checklist to troubleshoot ORT performance.

1. Are you using OpenMP? - OpenMP will parallelize some of the code for potential performance improvements. This is not recommended for running on single threads.
2. Have you enabled all [graph optimizations](graph-optimizations.md)? - The official published packages do enable all by default, but when building from source, check that these are enabled in your build.
3. Have you searched through prior filed [Github issues?](https://github.com/microsoft/onnxruntime/issues) - Checking the issues solved earlier will help you in troubleshooting. You can file a new issue if your performance questions are unique.
4. Do you have the right versions of the dependent libraries installed? - For CUDA or TensorRT, performance improves with the correct version of the dependent libraries.


<p><a href="#" id="back-to-top">Back to top</a></p>

### ORT Performance Tuning FAQs

Here are some FAQs for the OnnxRuntime performance tuning.

### 1. How do I optimize BERT models in ORT?

For some BERT models, ONNX Runtime cannot apply the best optimization due to framework version updates. We recommend trying out the [BERT optimization tool](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers), which reflects the latest changes in graph pattern matching and model conversions. The Transformer Model Optimization tool automatically applies the optimization while loading a model. 

A set of [Jupyter notebooks](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers/notebooks) and [Onnx Runtime Inference Examples](https://github.com/microsoft/onnxruntime-inference-examples/) are available to help you optimize your BERT models in ORT. For [fine tuning a BERT model using Tensorflow](https://www.tensorflow.org/text/tutorials/fine_tune_bert), you can refer to additional tutorials.


### 2. Why is the ONNX model graph not optimized even with graph_optimization_level set to ORT_ENABLE_ALL?

The ONNX model from IR_VERSION 4 only treats initializers that appear in graph input as non-constant. This may fail some of the graph optimizations, like const folding, operator fusion, and so on.

You can move initializers out of graph inputs if there is no need to override them, by either re-generating the model with latest exporter/converter or with the tool [remove_initializer_from_input.py](https://github.com/microsoft/onnxruntime/tree/master/tools/python/remove_initializer_from_input.py).

### 3. Why is my ONNX model running slower on GPU than CPU?

Depending on the execution provider you are using, all the operators may not have full support for your model. Fallback to CPU operators can cause hits in the performance speed.

Even though an operator is implemented by the CUDA execution provider, it may not necessarily assign/place the operator to the CUDA EP due to performance reasons. To see the placement decided by ORT, you can turn on verbose logging and look at the console output.

### 4. Why is my converted Tensorflow ONNX model slow?

Number-Channel-Height-Width (NCHW) and Number-Height-Width-Channel (NHWC) are two different memory layouts for 4-D tensors.

Most TensorFlow operations used by a CNN support both the NCHW and the NHWC data format. Tensorflow team suggests that on a GPU - NCHW is faster but on a CPU - NHWC is faster. However, ONNX only supports NCHW.

If the original model is in NHWC format, extra transposes may be added when the model is converted. The [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) converter does remove many of these transposes, but if this doesn't help, consider retraining the model using NCHW.

### 5. How do I mitigate high latency variance in Onnx Runtime?

On some platforms, OnnxRuntime may exhibit high latency variance during inferencing. This is caused by the 'constant cost model' that OnnxRuntime uses to parallelize tasks in the thread pool.

For each task, the 'constant cost model' will calculate a granularity for parallelization among threads that stays constant to the end of the task execution. This approach can bring imbalanced load causing high latency variance.

To mitigate this, OnnxRuntime provides a 'dynamic cost model' which can be enabled by setting the **dynamic_block_base** in the Python code:

```python
sess_options.add_session_config_entry('session.dynamic_block_base', '4')
```

Whenever the session is set with a positive value, OnnxRuntime thread pool will parallelize internal tasks with a decreasing granularity.

Let's assume that there is a function expected to run 'N' number of times by the thread pool. With the 'dynamic cost model' enabled, each thread in the pool will claim the total number of threads whenever it is ready to run.

Here is the Python code for the threads.

```python
residual_of_N / (dynamic_block_base * num_of_threads)
```

Over a period, threads in the pool are likely to be load balanced, thereby reducing the latency variance. The ORT 'dynamic cost model' setting may also be suitable for models when threads are more likely be preempted. As per our tests, the best configuration for 'dynamic_block_base' is 4, which lowers the variance while maintaining optimal performance.

<p><a href="#" id="back-to-top">Back to top</a></p>

</div>