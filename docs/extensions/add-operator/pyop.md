---
title: Create a Python Operator
description: Instructions to create a custom operator using Python functions and ORT inference integration.
parent: Add Operators
grand_parent: Extensions
nav_order: 1
---

# Creating custom operators using Python functions

Custom operators are a powerful feature in ONNX Runtime that allows users to extend the functionality of the runtime by implementing their own operators to perform specific operations not available in the standard ONNX operator set.

In this document, we will introduce how to create a custom operator using Python functions and integrate it into ONNX Runtime for inference.


## Step 1: Define the Python function for the custom operator
Start by defining the Python function that will serve as the implementation for your custom operator. Ensure that the function is compatible with the input and output tensor shapes you expect for your custom operator.
the Python decorator @onnx_op will convert the function to be a custom operator implementation. The following is example we create a function for a tokenizer 

```Python
@onnx_op(op_type="GPT2Tokenizer",
            inputs=[PyCustomOpDef.dt_string],
            outputs=[PyCustomOpDef.dt_int64, PyCustomOpDef.dt_int64],
            attrs={"padding_length": PyCustomOpDef.dt_int64})
def bpe_tokenizer(s, **kwargs):
    padding_length = kwargs["padding_length"]
    input_ids, attention_mask = cls.tokenizer.tokenizer_sentence([s[0]], padding_length)
    return input_ids, attention_mask
```
Because ONNXRuntimme needs the custom operator schema on loading a model, please specify them by onnx_op arguments. Also 'attrs' is needed if there are attributes for the ONNX node, which can be dict that mapping from its name to its type, or be a list if all types are string only.

## Step 2: Create an ONNX model with the custom operator
Now that the custom operator is registered with ONNX Runtime, you can create an ONNX model that utilizes it. You can either modify an existing ONNX model to include the custom operator or create a new one from scratch.

To create a new ONNX model with the custom operator, you can use the ONNX Python API. Here is an example: [test_pyops.py](https://github.com/microsoft/onnxruntime-extensions/blob/main/test/test_pyops.py)
