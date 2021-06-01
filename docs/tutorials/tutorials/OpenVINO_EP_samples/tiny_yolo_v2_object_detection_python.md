---
nav_exclude: true
---

# Object detection with tinyYOLOv2 in Python using OpenVINO Execution Provider:
{: .no_toc }

1. The Object detection sample again uses a tinyYOLOv2 Deep Learning ONNX Model from the ONNX Model Zoo.
 
2. The sample involves presenting a frame-by-frame video to ONNX Runtime (RT), which uses the OpenVINO Execution Provider to run inference on various Intel hardware devices as mentioned before and perform object detection to detect up to 20 different objects like birds, buses, cars, people and much more.

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime/tree/master/samples/python/OpenVINO_EP/tiny_yolo_v2_object_detection).

# How to build

## Prerequisites
1. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html)

2. Download the latest tinyYOLOv2 model from the ONNX Model Zoo.
   This model was adapted from [ONNX Model Zoo](https://github.com/onnx/models).Download the latest version of the [tinyYOLOv2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2) model from here.

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
* use any sample video with objects as test input to this sample
* Download the tinyYOLOv2 model from the ONNX Model Zoo

## Running the ONNXRuntime OpenVINO Execution Provider sample

```bash
python3 tiny_yolov2_obj_detection_sample.py
```

## To stop the sample from running

```bash
Just press the letter 'q' or Ctrl+C if on Windows
```

