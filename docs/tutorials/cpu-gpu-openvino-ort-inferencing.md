# Optimal Inferencing on Flexible Hardware with ONNX Runtime

As a developer who wants to deploy a PyTorch or ONNX model and maximize performance and hardware flexibility, you can leverage ONNX Runtime to optimally execute your model on your hardware platform.

In this tutorial, you'll learn:

1. how to use the [PyTorch](https://pytorch.org/vision/stable/models.html) ResNet-50 modelfor image classification
2. convert to ONNX, and
3. deploy to the default CPU, NVIDIA CUDA (GPU), and Intel OpenVINO with ONNX Runtime -- using the same application code to load and execute the inference across hardware platforms.

[ONNX](https://onnx.ai/) was developed as the open-sourced ML model format by Microsoft, Meta, Amazon, and other tech companies to standardize and make it easy to deploy Machine Learning models on various types of hardware.[ONNX Runtime](https://onnxruntime.ai/) was contributed and is maintained by Microsoft to optimize ONNX model performance over frameworks like PyTorch, Tensorflow, and more. The ResNet-50 model is a commonly used model for image classification that is pretrained on ImageNet.

This tutorial demonstrates how to run an ONNX model on CPU, GPU, and OpenVINO hardware with ONNX Runtime, using [Microsoft Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/).

&nbsp; 

**Setup**

- **OS Prerequisities:** Your environment should have [curl](https://curl.se/download.html) installed. This tutorial uses a Windows OS.
- **Device Prerequisites** : The onnxruntime-gpu library needs access to a [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#:~:text=You%20can%20verify%20that%20you,that%20GPU%20is%20CUDA%2Dcapable.) accelerator in your device or compute cluster, but running on just CPU works for the CPU and OpenVINO-CPU demos.
- **Inference Prerequisites:** Ensure that you have an image to inference on. For this tutorial, we have a "cat.jpg" image located in the same directory as the Notebook files.
- **Environment Prerequisites:** In Azure Notebook Terminal or AnaConda prompt window, run the following commands to create your 3 environments for CPU, GPU, and/or OpenVINO (differences are bolded).

**CPU**

conda create -n **cpu\_env\_demo** python=3.8

conda activate **cpu\_env\_demo**

conda install -c anaconda ipykernel

conda install -c conda-forge ipywidgets

python -m ipykernel install --user --name= **cpu\_env\_demo**

jupyter notebook

&nbsp; 

**GPU**

conda create -n **gpu\_env\_demo** python=3.8

conda activate **gpu\_env\_demo**

conda install -c anaconda ipykernel

conda install -c conda-forge ipywidgets

python -m ipykernel install --user --name= **gpu\_env\_demo**

jupyter notebook

&nbsp; 

**OpenVINO**

conda create -n **openvino\_env\_demo** python=3.8

conda activate **openvino\_env\_demo**

conda install -c anaconda ipykernel

conda install -c conda-forge ipywidgets

python -m ipykernel install --user --name= **openvino\_env\_demo**

**python**  **-**** m pip install **** -- ****upgrade pip**

**pip install openvino**

&nbsp; 

- **Library Requirements:** In the first code cell, install the necessary libraries with the following code snippets (differences are bolded).

**CPU + GPU**

import sys

if sys.platform in ['linux', 'win32']: # Linux or Windows

!{sys.executable} -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch\_stable.html

else: # Mac

print("PyTorch 1.9 MacOS Binaries do not support CUDA, install from source instead")

!{sys.executable} -m pip install **onnxruntime-gpu** onnx onnxconverter\_common==1.8.1 pillow

&nbsp;  

**OpenVINO**

import sys

if sys.platform in ['linux', 'win32']: # Linux or Windows

!{sys.executable} -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch\_stable.html

else: # Mac

print("PyTorch 1.9 MacOS Binaries do not support CUDA, install from source instead")

!{sys.executable} -m pip install **onnxruntime-openvino** onnx onnxconverter\_common==1.8.1 pillow

**import openvino.utils as utils**

**utils.add\_openvino\_libs\_to\_path()**

&nbsp; 

## ResNet-50 Demo

1. **Environment Setup** : Import necessary libraries to get models and run inferencing.

```bash
from torchvision import models, datasets, transforms as T

import torch

from PIL import Image

import numpy as np
```

&nbsp; 

2. **Load and Export Pre-trained ResNet-50 model to ONNX**: Download a pretrained ResNet-50 model from PyTorch and export to ONNX format.

```bash
resnet50 = models.resnet50(pretrained=True)

# Download ImageNet labels

!curl -o imagenet\_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet\_classes.txt

# Read the categories

with open("imagenet\_classes.txt", "r") as f:

categories = [s.strip() for s in f.readlines()]

# Export the model to ONNX

image\_height = 224

image\_width = 224

x = torch.randn(1, 3, image\_height, image\_width, requires\_grad=True)

torch\_out = resnet50(x)

torch.onnx.export(resnet50, # model being run

x, # model input (or a tuple for multiple inputs)

"resnet50.onnx", # where to save the model (can be a file or file-like object)

export\_params=True, # store the trained parameter weights inside the model file

opset\_version=12, # the ONNX version to export the model to

do\_constant\_folding=True, # whether to execute constant folding for optimization

input\_names = ['input'], # the model's input names

output\_names = ['output']) # the model's output names
```

Sample Output:

% Total % Received % Xferd Average Speed Time Time Time Current

Dload Upload Total Spent Left Speed

100 10472 100 10472 0 0 50581 0 --:--:-- --:--:-- --:--:-- 50834

&nbsp; 

3. **Set up Pre-Processing for Inferencing**: Create preprocessing for the image (ex. cat.jpg) you want to use the model to inference on.

```bash
# Pre-processing for ResNet-50 Inferencing, from https://pytorch.org/hub/pytorch\_vision\_resnet/

resnet50.eval()

filename = 'cat.jpg' # change to your filename

input\_image = Image.open(filename)

preprocess = T.Compose([

T.Resize(256),

T.CenterCrop(224),

T.ToTensor(),

T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

input\_tensor = preprocess(input\_image)

input\_batch = input\_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available

print("GPU Availability: ", torch.cuda.is\_available())

if torch.cuda.is\_available():

input\_batch = input\_batch.to('cuda')

resnet50.to('cuda')
```
Sample Output:

GPU Availability: False

&nbsp; 

4. **Inference ResNet-50 ONNX Model with ONNX Runtime:** Inference the model with ONNX Runtime by selecting the appropriate Execution Provider for the environment. If your environment uses CPU, uncomment CPUExecutionProvider, if the environment uses NVIDIA CUDA, uncomment CUDAExecutionProvider, and if the environment uses OpenVINOExecutionProvider, uncomment OpenVINOExecutionProvider – commenting out the other "onnxruntime.InferenceSession" lines of code.

```bash
# Inference with ONNX Runtime

import onnxruntime

from onnx import numpy\_helper

import time

session\_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['CPUExecutionProvider'])

# session\_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['CUDAExecutionProvider'])

# session\_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['OpenVINOExecutionProvider'])

def softmax(x):

"""Compute softmax values for each sets of scores in x."""

e\_x = np.exp(x - np.max(x))

return e\_x / e\_x.sum()

latency = []

def run\_sample(session, image\_file, categories, inputs):

start = time.time()

input\_arr = inputs.cpu().detach().numpy()

ort\_outputs = session.run([], {'input':input\_arr})[0]

latency.append(time.time() - start)

output = ort\_outputs.flatten()

output = softmax(output) # this is optional

top5\_catid = np.argsort(-output)[:5]

for catid in top5\_catid:

print(categories[catid], output[catid])

return ort\_outputs

ort\_output = run\_sample(session\_fp32, 'cat.jpg', categories, input\_batch)

print("ONNX Runtime **CPU/GPU/OpenVINO** Inference time = {} ms".format(format(sum(latency) \* 1000 / len(latency), '.2f')))
```
Sample output:

Egyptian cat 0.78605634

tabby 0.117310025

tiger cat 0.020089425

Siamese cat 0.011728076

plastic bag 0.0052174763

ONNX Runtime CPU Inference time = 32.34 ms

&nbsp; 

5. **Comparison with PyTorch:** Use PyTorch inferencing to benchmark with ONNX Runtime CPU and GPU inferencing accuracy and runtime.

```bash
# PyTorch Inferencing on ResNet50

import time

latency = []

with torch.no\_grad():

start = time.time()

pt\_output = resnet50(input\_batch)

latency.append(time.time() - start)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.

probabilities = torch.nn.functional.softmax(pt\_output[0], dim=0)

# Show top categories per image

top5\_prob, top5\_catid = torch.topk(probabilities, 5)

for i in range(top5\_prob.size(0)):

print(categories[top5\_catid[i]], top5\_prob[i].item())

print("PyTorch **CPU/GPU** Inference time = {} ms".format(format(sum(latency) \* 1000 / len(latency), '.2f')))

print("\*\*\*\*\* Verifying correctness \*\*\*\*\*")

for i in range(2):

print('PyTorch and ONNX Runtime output {} are close:'.format(i), np.allclose(ort\_output, pt\_output.cpu(), rtol=1e-05, atol=1e-04))
```
Sample output:

Egyptian cat 0.7860558032989502

tabby 0.1173100471496582

tiger cat 0.020089421421289444

Siamese cat 0.011728067882359028

plastic bag 0.005217487458139658

PyTorch CPU Inference time = 58.14 ms

\*\*\*\*\* Verifying correctness \*\*\*\*\*

PyTorch and ONNX Runtime output 0 are close: True

PyTorch and ONNX Runtime output 1 are close: True

&nbsp; 

6. **Comparison with OpenVINO:** Use OpenVINO inferencing to benchmark with ONNX Runtime OpenVINO inferencing accuracy and runtime.

```bash
# Inference with OpenVINO

from openvino.runtime import Core

ie = Core()

onnx\_model\_path = "./resnet50.onnx"

model\_onnx = ie.read\_model(model=onnx\_model\_path)

compiled\_model\_onnx = ie.compile\_model(model=model\_onnx, device\_name="CPU")

# inference

output\_layer = next(iter(compiled\_model\_onnx.outputs))

latency = []

input\_arr = input\_batch.detach().numpy()

inputs = {'input':input\_arr}

start = time.time()

request = compiled\_model\_onnx.create\_infer\_request()

output = request.infer(inputs=inputs)

outputs = request.get\_output\_tensor(output\_layer.index).data

latency.append(time.time() - start)

print("OpenVINO CPU Inference time = {} ms".format(format(sum(latency) \* 1000 / len(latency), '.2f')))

print("\*\*\*\*\* Verifying correctness \*\*\*\*\*")

for i in range(2):

print('OpenVINO and ONNX Runtime output {} are close:'.format(i), np.allclose(ort\_output, outputs, rtol=1e-05, atol=1e-04))
```
Sample output:

Egyptian cat 0.7820879

tabby 0.113261245

tiger cat 0.020114701

Siamese cat 0.012514038

plastic bag 0.0056432663

OpenVINO CPU Inference time = 31.83 ms

\*\*\*\*\* Verifying correctness \*\*\*\*\*

PyTorch and ONNX Runtime output 0 are close: True

PyTorch and ONNX Runtime output 1 are close: True

&nbsp; 

## Conclusion

We've demonstrated that ONNX Runtime is an effective way to run your PyTorch or ONNX model on CPU, NVIDIA CUDA (GPU), and Intel OpenVINO (Mobile). ONNX Runtime enables deployment to more types of hardware that can be found on [Execution Providers](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html). As you try ONNX Runtime, feel free to leverage the onnxruntime.ai [documentation and tutorials](https://onnxruntime.ai/docs/), and we'd love to hear your feedback by participating in our ONNX Runtime [Github repo](https://github.com/microsoft/onnxruntime).

&nbsp; 

## Video Demonstration

Watch the video [here](https://www.youtube.com/embed/sbc3Bmv2Kwo?feature=oembed) for more explanation on ResNet-50 Deployment and Flexible Inferencing with the step by step guide.
