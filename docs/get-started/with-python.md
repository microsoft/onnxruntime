---
title: Python
parent: Get Started
toc: true
nav_order: 1
---
# Get started with ONNX Runtime in Python
{: .no_toc }

Below is a quick guide to get the packages installed to use ONNX for model serialization and infernece with ORT.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Install ONNX Runtime

There are two Python packages for ONNX Runtime. Only one of these packages should be installed at a time in any one environment. The GPU package encompasses most of the CPU functionality.

```bash
pip install onnxruntime-gpu
```

Use the CPU package if you are running on Arm CPUs and/or macOS.

```bash
pip install onnxruntime
```

## Install ONNX for model export


```python
## ONNX is built into PyTorch
pip install torch
```
```python
## tensorflow
pip install tf2onnx
```
```python
## sklearn
pip install skl2onnx
```

## Quickstart Examples for PyTorch, TensorFlow, and SciKit Learn
Train a model using your favorite framework, export to ONNX format and inference in any supported ONNX Runtime language!

### PyTorch CV
{: .no_toc }
In this example we will go over how to export a PyTorch CV model into ONNX format and then inference with ORT. The code to create the model is from the [PyTorch Fundamentals learning path on Microsoft Learn](https://aka.ms/learnpytorch).

- Export the model using `torch.onnx.export`

```python
torch.onnx.export(model,                                # model being run
                  torch.randn(1, 28, 28).to(device),    # model input (or a tuple for multiple inputs)
                  "fashion_mnist_model.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output names
```
- Load the onnx model with `onnx.load`
```python
import onnx
onnx_model = onnx.load("fashion_mnist_model.onnx")
onnx.checker.check_model(onnx_model)
```
- Create inference session using `ort.InferenceSession`

```python
import onnxruntime as ort
import numpy as np
x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

### PyTorch NLP
{: .no_toc }
In this example we will go over how to export a PyTorch NLP model into ONNX format and then inference with ORT. The code to create the AG News model is from [this PyTorch tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html).

- Process text and create the sample data input and offsets for export.
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
- Load the model using `onnx.load`
```python
import onnx
onnx_model = onnx.load("ag_news_model.onnx")
onnx.checker.check_model(onnx_model)
```

- Create inference session with `ort.infernnce`
```python
import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession('ag_news_model.onnx')
outputs = ort_sess.run(None, {'input': text.numpy(),
                              'offsets':  torch.tensor([0]).numpy()})
# Print Result
result = outputs[0].argmax(axis=1)+1
print("This is a %s news" %ag_news_label[result[0]])
```

### TensorFlow CV
{: .no_toc }
In this example we will go over how to export a TensorFlow CV model into ONNX format and then inference with ORT. The model used is from this [GitHub Notebook for Keras resnet50](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/keras-resnet50.ipynb).

- Get the pretrained model

```python
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import onnxruntime

model = ResNet50(weights='imagenet')

preds = model.predict(x)
print('Keras Predicted:', decode_predictions(preds, top=3)[0])
model.save(os.path.join("/tmp", model.name))
```
- Convert the model to onnx and export

```python
import tf2onnx
import onnxruntime as rt

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
```
- Create inference session with `rt.infernnce`

```python
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(output_names, {"input": x})

print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])
```

### SciKit Learn CV
{: .no_toc }
In this example we will go over how to export a SciKit Learn CV model into ONNX format and then inference with ORT. We’ll use the famous iris datasets.

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

## Python API Reference Docs

 <span class="fs-5"> [Go to the ORT Python API Docs](../api/python/api_summary.html){: .btn  .mr-4 target="_blank"} </span>

## Builds
If using pip, run `pip install --upgrade pip` prior to downloading.

| Artifact      | Description | Supported Platforms |
|-----------    |-------------|---------------------|
|[onnxruntime](https://pypi.org/project/onnxruntime)|CPU (Release)| Windows (x64), Linux (x64, ARM64), Mac (X64),  |
|[ort-nightly](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly)|CPU (Dev)    | Same as above |
|[onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu)|GPU (Release)| Windows (x64), Linux (x64, ARM64) |
|[ort-nightly-gpu for CUDA 11.*](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-gpu) |GPU (Dev) | Windows (x64), Linux (x64, ARM64) |
|[ort-nightly-gpu for CUDA 12.*](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ort-cuda-12-nightly/PyPI/ort-nightly-gpu) |GPU (Dev) | Windows (x64), Linux (x64, ARM64) |

Before installing nightly package, you will need install dependencies first.
```
python -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
```

Example to install ort-nightly-gpu for CUDA 11.*:
```
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

Example to install ort-nightly-gpu for CUDA 12.*:
```
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
```

For Python compiler version notes, see [this page](https://github.com/microsoft/onnxruntime/tree/main/docs/Python_Dev_Notes.md)

## Learn More
- [Python Tutorials](../tutorials/api-basics)
* [TensorFlow with ONNX Runtime](../tutorials/tf-get-started.md)
* [PyTorch with ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* [scikit-learn with ONNX Runtime](http://onnx.ai/sklearn-onnx/index_tutorial.html)
