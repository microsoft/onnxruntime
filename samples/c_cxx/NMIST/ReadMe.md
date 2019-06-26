# NMIST Sample - Number recognition

This sample uses the NMIST model from the Model Zoo: https://github.com/onnx/models/tree/master/mnist

(inline image of app)

## Requirements

Compiled Onnxruntime.dll / lib (link to instructions on how to build dll)
Windows Visual Studio Compiler (cl.exe)

## Build

Run 'build.bat' in this directory to call cl.exe to generate NMIST.exe
Then just run NMIST.exe

## How to use it

Just draw a number with the left mouse button (or use touch) in the box on the left side. After releasing the mouse button the model will be run and the outputs of the model will be displayed.

To clear the image, click the right mouse button anywhere.

## How it works

A single Ort::Env is created globally to initialize the runtime.

The NMIST structure abstracts away all of the interaction with the Onnx Runtime, creating the tensors, and running the model.

ConvertDibToNmist converts the image data in the 32-bit Windows DIB into the NMIST model's input tensor format.

WWinMain is the Windows entry point, it creates the main window.

WndProc is the window procedure for the window, handling the mouse input and drawing the graphics

