Until ORT is officially released with Xamarin support in the nuget package you need to
  - get the managed and native nuget packages from the internal Zip-Nuget-Java packaging pipeline for a build of master
  - put that in a local directory
  - update the nuget.config to point to that directory

Additionally, the FasterRCNN model is required to be in the Models directory.

From this directory:
```
> mkdir Models
> cd Models
> wget https://github.com/onnx/models/blob/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx
```