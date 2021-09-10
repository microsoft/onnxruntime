---
title: Python
parent: Get Started
toc: true
nav_order: 1
---
# Get started with ORT for Python
{: .no_toc }

Below is a quick guide to get the packages installed to use ONNX for model serialization and infernece with ORT.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Install ONNX Runtime (ORT)

```python
pip install onnxruntime
```

## Install ONNX for model export


```python
## pytorch
pip install onnx-pytorch
```
```python
## tensorflow
pip install onnx-tf
```
```python
## sklearn
pip install skl2onnx
```

## Examples
Train a model using your favorite framework, export to ONNX format and inference in any supported ONNX Runtime language!

### PyTorch CV
In this example we will go over how to export the model into onnx format and then inference with ORT. The code to create the model is from the [PyTorch Fundamentals learning path on Microsoft Learn](aka.ms/learnpytorch).

- Export model

```python
torch.onnx.export(model,                                # model being run
                  torch.randn(1, 28, 28).to(device),    # model input (or a tuple for multiple inputs)
                  "fashion_mnist_model.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output names
```
- Load model
```python
import onnx
onnx_model = onnx.load("fashion_mnist_model.onnx")
onnx.checker.check_model(onnx_model)
```
- Create inference session

```python
import onnxruntime as ort
import numpy as np
x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})
```
- Print result

```python
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

### PyTorch NLP
In this example we will go over how to export the model into onnx format and then inference with ORT. The code to create the AG News model is from [this PyTorch tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html).

- Process text
```python
import torch
text = "Text from the news article"
text = torch.tensor(text_pipeline(text))
offsets = torch.tensor([0])
```
- Export Model
```python
# Export the model
torch.onnx.export(model,                     # model being run
                  (text, offsets),           # model input (or a tuple for multiple inputs)
                  "ag_news_model.onnx",      # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input', 'offsets'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```
- Load Model
```python
import onnx
onnx_model = onnx.load("ag_news_model.onnx")
onnx.checker.check_model(onnx_model)
```

- Create inference session
```python
import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession('ag_news_model.onnx')
outputs = ort_sess.run(None, {'input': text.numpy(),
                              'offsets':  torch.tensor([0]).numpy()})
```
- Print result
```python
result = outputs[0].argmax(axis=1)+1
print("This is a %s news" %ag_news_label[result[0]])
```

### TensorFlow CV
TODO

### Tensorflow NLP
TODO

### SciKit Learn CV

We’ll use the famous iris datasets.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
clr = LogisticRegression()
clr.fit(X_train, y_train)
print(clr)

LogisticRegression()
```

- Convert or export the model into ONNX format
ONNX is a format to describe the machine learned model. It defines a set of commonly used operators to compose models. There are tools to convert other model formats into ONNX. Here we will use ONNXMLTools.

```python

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

    
- Load and run the model using ONNX Runtime
We will use ONNX Runtime to compute the predictions for this machine learning model.

```python

import numpy
import onnxruntime as rt

sess = rt.InferenceSession("logreg_iris.onnx")
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

OUTPUT:
 [0 1 0 0 1 2 2 0 0 2 1 0 2 2 1 1 2 2 2 0 2 2 1 2 1 1 1 0 2 1 1 1 1 0 1 0 0
  1]
```

- Get predicted class

The code can be changed to get one specific output by specifying its name into a list.

```python
import numpy
import onnxruntime as rt

sess = rt.InferenceSession("logreg_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

```

### SciKit Learn NLP
TODO

### ORT Training package

```
pip install torch-ort
python -m torch_ort.configure
```

**Note**: This installs the default version of the `torch-ort` and `onnxruntime-training` packages that are mapped to specific versions of the CUDA libraries. Refer to the install options in [ONNXRUNTIME.ai](https://onnxruntime.ai).

### Add ORTModule in the `train.py`

```python
   from torch_ort import ORTModule
   .
   .
   .
   model = ORTModule(model)
```

## Python API Reference Docs

 <span class="fs-5"> [Go to the ORT Python API Docs](../api/python/api_summary.html){: .btn  .mr-4 target="_blank"} </span> 

## Builds
If using pip, run pip install `--upgrade pip` prior to downloading.	 

| Artifact      | Description | Supported Platforms |
|-----------    |-------------|---------------------|
|[onnxruntime](https://pypi.org/project/onnxruntime)|CPU (Release)| Windows (x64), Linux (x64, ARM64), Mac (X64),  |
|[ort-nightly](https://test.pypi.org/project/ort-nightly)|CPU (Dev)    | Same as above |
|[onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu)|GPU (Release)| Windows (x64), Linux (x64, ARM64) |
|[ort-gpu-nightly](https://test.pypi.org/project/ort-gpu-nightly)|GPU (Dev) | Same as above |


For Python compiler version notes, see [this page](https://github.com/microsoft/onnxruntime/tree/master/docs/Python_Dev_Notes.md)


## Supported Versions
Python 3.6 - 3.9

## Learn More
- [Python Tutorials](../tutorials/api-basics)
- [Python Github Quickstart Templates](https://github.com/onnxruntime)
* [TensorFlow with ONNX Runtime](../tutorials/tf-get-started.md)
* [PyTorch with ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* [scikit-learn with ONNX Runtime](https://www.onnxruntime.ai/python/tutorial.html)
 