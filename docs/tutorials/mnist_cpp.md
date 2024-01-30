---
nav_exclude: true 
---

# Number recognition with MNIST in C++
{: .no_toc }

This sample uses the MNIST model from the Model Zoo: https://github.com/onnx/models/tree/main/validated/vision/classification/mnist

![Screenshot](../../../images/mnist-screenshot.png)

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

Compiled Onnxruntime.dll / lib (link to instructions on how to build dll)
Windows Visual Studio Compiler (cl.exe)

## Build

Run 'build.bat' in this directory to call cl.exe to generate MNIST.exe
Then just run MNIST.exe

## How to use it

Just draw a number with the left mouse button (or use touch) in the box on the left side. After releasing the mouse button the model will be run and the outputs of the model will be displayed. Note that when drawing numbers requiring multiple drawing strokes, the model will be run at the end of each stroke with probably wrong predictions (but it's amusing to see and avoids needing to press a 'run model' button).

To clear the image, click the right mouse button anywhere.

## How it works

A single Ort::Env is created globally to initialize the runtime.

```
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
```
[[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L12)
 

The MNIST structure abstracts away all of the interaction with the Onnx Runtime, creating the tensors, and running the model.

WWinMain is the Windows entry point, it creates the main window.

WndProc is the window procedure for the window, handling the mouse input and drawing the graphics

### Preprocessing the data

MNIST's input is a {1,1,28,28} shaped float tensor, which is basically a 28x28 floating point grayscale image (0.0 = background, 1.0 = foreground).

The sample stores the image in a 32-bit per pixel windows DIB section, since that's easy to draw into and draw to the screen for windows. The DIB is created here:
```
 {
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
    bmi.bmiHeader.biWidth = MNIST::width_;
    bmi.bmiHeader.biHeight = -MNIST::height_;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits;
    dib_ = CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
  }
  ```
[[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L109-L121)

The function to convert the DIB data and writ it into the model's input tensor:
```
void ConvertDibToMnist() {
  DIBInfo info{dib_};

  const DWORD* input = reinterpret_cast<const DWORD*>(info.Bits());
  float* output = mnist_.input_image_.data();

  std::fill(mnist_.input_image_.begin(), mnist_.input_image_.end(), 0.f);

  for (unsigned y = 0; y < MNIST::height_; y++) {
    for (unsigned x = 0; x < MNIST::width_; x++) {
      output[x] += input[x] == 0 ? 1.0f : 0.0f;
    }
    input = reinterpret_cast<const DWORD*>(reinterpret_cast<const BYTE*>(input) + info.Pitch());
  }
  output += MNIST::width_;
}
```
[[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L77-L92)

### Postprocessing the output

MNIST's output is a simple {1,10} float tensor that holds the likelihood weights per number. The number with the highest value is the model's best guess.

The MNIST structure uses std::max_element to do this and stores it in result_:
```   
result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
```
[[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L31)

To make things more interesting, the window painting handler graphs the probabilities and shows the weights here:
```
 // Hilight the winner
      RECT rc{graphs_left, mnist_.result_ * 16, graphs_left + graph_width + 128, (mnist_.result_ + 1) * 16};
      FillRect(hdc, &rc, brush_winner_);

      // For every entry, draw the odds and the graph for it
      SetBkMode(hdc, TRANSPARENT);
      wchar_t value[80];
      for (unsigned i = 0; i < 10; i++) {
        int y = 16 * i;
        float result = mnist_.results_[i];

        auto length = wsprintf(value, L"%2d: %d.%02d", i, int(result), abs(int(result * 100) % 100));
        TextOut(hdc, graphs_left + graph_width + 5, y, value, length);

        Rectangle(hdc, graphs_zero, y + 1, graphs_zero + result * graph_width / range, y + 14);
      }

      // Draw the zero line
      MoveToEx(hdc, graphs_zero, 0, nullptr);
      LineTo(hdc, graphs_zero, 16 * 10);
```
[[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L164-L183)

### The Ort::Session

1. Creation: The Ort::Session is created inside the MNIST structure here:
     ```c++
    Ort::Session session_{env, ORT_TSTR("model.onnx"), Ort::SessionOptions{nullptr}};
    ```
    [[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L43)

2. Setup inputs & outputs: The input & output tensors are created here:
    ```c++
    MNIST() {
    auto allocator_info = Ort::AllocatorInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    }
    ```
    [[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L19-L23)

    In this usage, we're providing the memory location for the data instead of having Ort allocate the buffers. This is simpler in this case since the buffers are small and can just be fixed members of the MNIST struct.

3. Run: Running the session is done in the Run() method:
    ```c++
     int Run() {
    const char* input_names[] = {"Input3"};
    const char* output_names[] = {"Plus214_Output_0"};

    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
    }
     ```
    [[Source]](https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L25-L33)


