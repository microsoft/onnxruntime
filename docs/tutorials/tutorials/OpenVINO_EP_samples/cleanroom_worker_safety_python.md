This is a python sample application for demoing industrial/clean room use case using ONNXRuntime OpenVINO Execution Provider.

1. The Object detection sample uses a pre-trained Tiny YOLOv2 Deep Learning ONNX Model from Intel. 
 
2. The Clean Room Worker Safety sample demonstrates object detection in an industrial/clean room use case. The sample involves presenting a frame-by-frame video to the ONNX Runtime (RT), which uses the OpenVINO Execution Provider to run inference on various Intel hardware, such as CPU, iGPU, accelerator cards like NCS2 and VAD-M. This sample uses a pretrained Tiny YOLOv2 Deep Learning ONNX Model for the detection of safety gear (e.g., bunny suit, safety glasses), robots, and heads, and can be used for hazard detection purposes.

# Working
The Worker Safety Demo is a python script that runs deep learning inference on a public video of an Intel Cleanroom with a Tiny-YoloV2 model (public topology, weights generated from training on the public video). The script loads the model into the open source OnnxRuntime framework, performs post-processing on the results of the inference, and generates a new video containing the bounding boxes and labels for objects superimposed on each frame.

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime/tree/master/samples/python/OpenVINO_EP_samples/cleanroom_worker_safety).

# How to build

## Prerequisites
1. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html)

## Install ONNX Runtime for OpenVINO Execution Provider

## Build steps
[build instructions](https://www.onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html#build)


## Reference Documentation
[Documentation](https://www.onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html)

## Requirements

* ONNX Runtime 1.6+
* numpy version 1.19.5+
* opencv 4.5.1+
* python 3+
* use the sample videos provided in the same directory to test the sample.
* use the tinyYOLOv2 pre-trained model available in the same directory.

## Running the ONNXRuntime OpenVINO Execution Provider sample

```bash
python3 cleanroom_sample_OV_EP.py
```

## To stop the sample from running

```bash
Just press the letter 'q'
```