---
title: Troubleshooting
grand_parent: Performance
parent: Tune performance
nav_order: 6
---

# Troubleshooting performance issues

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

Here is a list of things to check when assessing performance issues:

* Have you enabled all [graph optimizations](../model-optimizations/graph-optimizations.md)? The official published packages do enable all by default but when building from source, check that these are enabled in your build.
* Have you searched through prior-filed [GitHub issues](https://github.com/microsoft/onnxruntime/issues) to see if your problem has been discussed previously? Please do this before filing new issues.
* If using CUDA or TensorRT, do you have the right versions of the dependent libraries installed? [CUDA EP](../../execution-providers/CUDA-ExecutionProvider.md#requirements) / [TensorRT EP](../../execution-providers/TensorRT-ExecutionProvider.md#requirements)


## Why is the model graph not optimized even with graph_optimization_level set to ORT_ENABLE_ALL?

The ONNX model from IR_VERSION 4 only treats initializers that appear in graph input as non-constant. This may prevent some of the graph optimizations like const folding, operator fusion etc. Move initializers out of graph inputs if there is no need to override them, by either re-generating the model with the latest exporter/converter or with the tool [remove_initializer_from_input.py](https://github.com/microsoft/onnxruntime/tree/main/tools/python/remove_initializer_from_input.py).


## Why is my model running slower on GPU than on CPU?

Depending on which execution provider you're using, it may not have full support for all the operators in your model. Fallback to CPU ops can cause hits in performance speed. Moreover, even if an op is implemented by the CUDA execution provider, it may not necessarily assign/place the op to the CUDA EP due to performance reasons. To see the placement decided by ORT, turn on verbose logging and look at the console output.


## My converted TensorFlow model is slow - why?

NCHW and NHWC are two different memory layout for 4-D tensors.

Most TensorFlow operations used by a CNN support both NHWC and NCHW data format. The TensorFlow team suggests that on GPUs NCHW is faster but on CPUs NHWC is sometimes faster in TensorFlow. However, ONNX only supports NCHW. As a result, if the original model is in NHWC format, extra transposes may be added when the model is converted. The [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) converter does remove many of these transposes, but if this doesn't help sufficiently, consider retraining the model in the NCHW format.



## I am seeing high latency variance.

On some platforms, onnxruntime may exhibit high latency variance during inferencing. This is caused by the constant cost model that onnxruntime uses to parallelize tasks in the thread pool.
For each task, the constant cost model will calculate a granularity for parallelization among threads, which stays constant to the end of the task execution. This approach can bring imbalanced load sometimes, causing high latency variance.
To mitigate this, onnxruntime provides a dynamic cost model which can be enabled as a session option:

```python
sess_options.add_session_config_entry('session.dynamic_block_base', '4')
```

Whenever set with a positive value, the onnxruntime thread pool will parallelize internal tasks with a decreasing granularity.
Specifically, assuming there is a function expected to run N number of times by the thread pool, with the dynamic cost model enabled, each thread in the pool will claim

```python
residual_of_N / (dynamic_block_base * num_of_threads)
```

whenever it is ready to run. So over a period of time, threads in the pool are likely to be better load balanced, thereby lowering the latency variance.

Due to the same reason, the dynamic cost model may also improve the performance for cases when threads are more likely be preempted.
Per our tests, by far the best configuration for dynamic_block_base is 4, which lowers the variance while keeping good performance.
    

## I am seeing high CPU usage on windows

It is observed that for machines have more than 64 logical cores, CPU usage could be notably lowered by letting the thread pool use a lock-free task queue,
which utilizes spinlock instead of mutex for synchronization. The lock-free task queue could be enabled by building onnxruntime from source with following flag:

```
--use_lock_free_queue
```