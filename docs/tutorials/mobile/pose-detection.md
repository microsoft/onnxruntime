---
title: Pose detection with Yolov8
description: Run the Yolov8 model with built-in pre and post processing
image: /images/tutorial-superres-og-image.png
parent: Deploy on mobile
grand_parent: Tutorials
nav_order: 3
---

# Pose detection with the Yolov8 model

Learn how to build an ONNX model for pose detection with built-in pre and post processing.

Note: this tutorial uses Python. Android and iOS samples are coming soon!

## Build the Yolov8 model with pre and post processing

Create a Python environment and install the following packages.

```bash
onnx
onnxruntime
onnxruntime-extensions
```

Download the following script to build the model.

```bash
https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/tutorials/yolov8_pose_e2e.py > yolov8_pose_e2e.py
```

Run the script.

```bash
python yolov8_pose_e2e.py 
```

After the script has run, you will see one PyTorch model and two ONNX models:
* yolov8n-pose.pt: The original Yolov8 PyTorch model
* yolov8n-pose.onnx: The exported Yolov8 ONNX model
* yolov8n-pose.with_pre_post_processing.onnx: The ONNX model with pre and post processing included in the model


## Run examples of pose detection

You can use the same script to run the model, supplying your own image to detect poses.

```bash
pytthon yolov8_pose_e2e.py --input_image person.jpg
```

And the output is drawn on the original image!

![Person with pose drawn](../../../images/person-with-pose.png)


## Develop your mobile application

You can use the Python inference code as a basis for developing your mobile application. 





