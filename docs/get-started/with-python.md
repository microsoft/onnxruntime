---
title: Python
parent: Get Started
toc: true
nav_order: 1
---
# Python ORT Inference Quickstart
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
{: .no_toc }
TODO

### PyTorch NLP
{: .no_toc }
TODO

### TensorFlow CV
{: .no_toc }
TODO

### Tensorflow NLP
{: .no_toc }
TODO

### SciKit Learn CV
{: .no_toc }

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
{: .no_toc }
TODO


## API Reference Docs

[Go to the Python API Reference](./api/python-api.html)


## Tutorials

[Tutorials](./python/tutorial.html)

## Builds
If using pip, run pip install `--upgrade pip` prior to downloading.	 

| Artifact      | Description | Supported Platforms |
|-----------    |-------------|---------------------|
|[onnxruntime](https://pypi.org/project/onnxruntime)|CPU (Release)| Windows (x64), Linux (x64, ARM64), Mac (X64),  |
|[ort-nightly](https://test.pypi.org/project/ort-nightly)|CPU (Dev)    | Same as above |
|[onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu)|GPU (Release)| Windows (x64), Linux (x64, ARM64) |
|[ort-gpu-nightly](https://test.pypi.org/project/ort-gpu-nightly)|GPU (Dev) | Same as above |


For Python compiler version notes, see [this page](https://github.com/microsoft/onnxruntime/tree/master/docs/Python_Dev_Notes.md)


## Samples
See [Tutorials: API Basics - Python](../tutorials/inferencing/api-basics.md#python)

## Supported Versions
Python 3.6 - 3.9

## Learn More
- [Python Tutorials](./Tutorials/)
- [Python Github Quickstart Templates](https://github.com/onnxruntime)
- [Python API Reference](./api/csharp-api.html)
 