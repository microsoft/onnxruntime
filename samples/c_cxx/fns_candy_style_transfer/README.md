# FNS Candy
FNS Candy is a style transfer model. In this sample application, we use the ONNX Runtime C API to process an image using the FNS Candy model in ONNX format.

# Build Instructions
See [../README.md](../README.md)

# Prepare data
First, download the FNS Candy ONNX model from [here](https://raw.githubusercontent.com/microsoft/Windows-Machine-Learning/master/Samples/FNSCandyStyleTransfer/UWP/cs/Assets/candy.onnx).

Then, prepare an image:
1. PNG format
2. Dimension of 720x720

# Run
Command to run the application:
```
fns_candy_style_transfer.exe <model_path> <input_image_path> <output_image_path> [cpu|cuda|dml]
```

To use the CUDA or DirectML execution providers, specify `cuda` or `dml` on the command line. `cpu` is the default.