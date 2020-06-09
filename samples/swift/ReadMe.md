# MNIST Sample - Number recognition

This sample uses the MNIST model from the Model Zoo: https://github.com/onnx/models/tree/master/vision/classification/mnist

![Screenshot](Screenshot.png)

## Requirements

* MacOS Catalina
* Xcode 11
* Compiled libonnxruntime.dll / lib


## Build

Command line:

```
$ xcodebuild -project SwiftMnist.xcodeproj 
```

From Xcode, open `SwiftMnist.xcodeproj` and run with Command-R.

## How to use it

Just draw a number on the surface, when you lift your finger from the
mouse or the trackpad, the guess will be displayed. 

Note that when drawing numbers requiring multiple drawing strokes, the
model will be run at the end of each stroke with probably wrong
predictions (but it's amusing to see and avoids needing to press a
'run model' button).

## How it works

(Add once it is added)

### Preprocessing the data

MNIST's input is a {1,1,28,28} shaped float tensor, which is basically
a 28x28 floating point grayscale image (0.0 = background, 1.0 =
foreground).

### Postprocessing the output

MNIST's output is a simple {1,10} float tensor that holds the
likelihood weights per number. The number with the highest value is
the model's best guess.

The MNIST structure uses std::max_element to do this and stores it in
result_:

https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L31

