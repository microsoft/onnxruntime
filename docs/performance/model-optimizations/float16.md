---
title: Float16 and mixed precision models
grand_parent: Performance
parent: Model optimizations
nav_order: 2
redirect_from: /docs/performance/float16
---
# Create Float16 and Mixed Precision Models
{: .no_toc }

Converting a model to use float16 instead of float32 can decrease the model size (up to half) and improve performance on some GPUs. There may be some accuracy loss, but in many models the new accuracy is acceptable. Tuning data is not needed for float16 conversion, which can make it preferable to quantization.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Float16 Conversion
Convert a model to float16 by following these steps:

1. Install onnx and [onnxconverter-common](https://github.com/microsoft/onnxconverter-common)

    `pip install onnx onnxconverter-common`

2. Use the `convert_float_to_float16` function in python.
    ```python
    import onnx
    from onnxconverter_common import float16

    model = onnx.load("path/to/model.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "path/to/model_fp16.onnx")
    ```

### Float16 Tool Arguments

If the converted model does not work or has poor accuracy, you may need to set additional arguments.

```python
convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False,
                         disable_shape_infer=False, op_block_list=None, node_block_list=None)
```

- `model`: The ONNX model to convert.
- `min_positive_val`, `max_finite_val`: Constant values will be clipped to these bounds. `0.0`, `nan`, `inf`, and `-inf` will be unchanged.
- `keep_io_types`: Whether model inputs/outputs should be left as float32.
- `disable_shape_infer`: Skips running onnx shape/type inference. Useful if shape inference is crashing, shapes/types are already present in the model, or types are not needed (types are used to determine where cast ops are needed for unsupported/blocked ops).
- `op_block_list`: List of op types to leave as float32. By default uses the list from `float16.DEFAULT_OP_BLOCK_LIST`. This list has ops that are not supported for float16 in ONNX Runtime.
- `node_block_list`: List of node names to leave as float32.

**NOTE**: Blocked ops will have have casts inserted around them to/from float16/float32. Currently, if two blocked ops are next to each other, the casts will still be inserted, creating a redundant pair. ORT will optimize this pair out at runtime, so the results will remain at full-precision.

## Mixed Precision

If float16 conversion is giving poor results, you can convert most of the ops to float16 but leave some in float32. The `auto_mixed_precision.auto_convert_mixed_precision` tool finds a minimal set of ops to skip while retaining a certain level of accuracy. You will need to provide a sample input for the model.

Since the CPU version of ONNX Runtime doesn't support float16 ops and the tool needs to measure the accuracy loss, **the mixed precision tool must be run on a device with a GPU**.

```python
from onnxconverter_common import auto_mixed_precision
import onnx

model = onnx.load("path/to/model.onnx")
# Assuming x is the input to the model
feed_dict = {'input': x.numpy()}
model_fp16 = auto_convert_mixed_precision(model, feed_dict, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "path/to/model_fp16.onnx")
```

### Mixed Precision Tool Arguments

```python
auto_convert_mixed_precision(model, feed_dict, validate_fn=None, rtol=None, atol=None, keep_io_types=False)
```

- `model`: The ONNX model to convert.
- `feed_dict`: Test data used to measure the accuracy of the model during conversion. Format is similar to InferenceSession.run (map of input names to values)
- `validate_fn`: A function accepting two lists of numpy arrays (the outputs of the float32 model and the mixed-precision model, respectively) that returns `True` if the results are sufficiently close and `False` otherwise. Can be used instead of or in addition to `rtol` and `atol`.
- `rtol`, `atol`: Absolute and relative tolerances used for validation. See [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) for more information.
- `keep_io_types`: Whether model inputs/outputs should be left as float32.

The mixed precision tool works by converting clusters of ops to float16. If a cluster fails, it is split in half and both clusters are tried independently. A visualization of the cluster sizes is printed as the tool works.
