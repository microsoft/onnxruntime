
# Overview

ImageTests are to test all kinds of scenarios that are related to images.

# Purpose

To make image related tests data driven and increase test code coverage.

# Image Test Data Table

|Table Name    |   Device     |EvaluationStrategies|Models                     |Input Files      |Output Binding Strategies|Input Image Sources  |InputPixelFormats|
|--------------|--------------|--------------------|---------------------------|-----------------|-------------------------|---------------------|-----------------|
|Table Rows    |GPU           |Async               |fns-candy_Bgr8_freeDimInput|fish_720.png     |Bound                    |FromImageFeatureValue|Bgra8
|              |CPU           |Sync                |fns-candy_Bgr8             |fish_720_Gray.png|Unbound                  |FromVideoFrame       |Rgba8
|              |              |                    |fns-candy_Rgb8             |                 |                         |FromCPUResource      |Gray8
|              |              |                    |fns-candy_tensor           |                 |                         |FromGPUResource      |

# Test Coverage

### 1. Devices

> Devices the Test uses.
* GPU
* CPU

### 2. Evaluation Strategy
> The strategy to use during evalution.

* EvalutateAsync
* EvaluateSync
* MultipleEvaluations (**TODO**)
* SwapChain (**TODO**)
* Chain (**TODO**)

### 3. Output Binding Strategy

* Bound
	```
	VideoFrame outputimage(BitmapPixelFormat::Rgba8, outputtensorshape.GetAt(3), outputtensorshape.GetAt(2));
	ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);
	```

* Unbound

### 4. Models
> Des: Currently, we are using four models, FNS-candy_Bgr8_freeDimInput, FNS-candy_Bgr8, FNS-candy_Rgb8, and FNS-candy_Tensor. We would try to get more models to cover the following listed cases.

a. [Input Image Metadata](https://github.com/onnx/onnx/blob/master/docs/MetadataProps.md)

1.  No metadata provided (**TODO**)
2. BitmapPixelFormat

	* Bgr8
	* Rgb8
	* Gray8
	* Bgra8 (**TODO**, not supported in RS5)
	* Rgba8 (**TODO**, not supported in RS5)
	* yuv (**TODO**, not in ONNX1.2.2 spec)

* ColorSpaceGamma
	* sRGB
	* Linear (**TODO**, not supported in RS5)

* NominalPixelRange
	*  NominalRange_0_255
	* Normalized_0_1 (No such model)
	* Normalized_1_1 (No such model)
	* NominalRange_16_235 (No such model)

b. Model Input Output Type
* free dimension
* image  :  corresponding to ImageFeatureDescriptor
* tensor  : corresponding to TensorFeatureDescriptor

c. Models we are using.

* Model takes Gray8 image as input and also output image (**TODO**, MNIST outputs tensor)
* FNS-candy_Bgr8_freeDimInput
* FNS-candy_Bgr8
* FNS-candy_Rgb8
* FNS-candy_Tensor


### 5. Input Image Files

* fish_720.png
* fish_720_gray.png

### 6. Input

a. input image source

* FromImageFeatureValue
	```
	VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
    VERIFY_NO_THROW(m_modelBinding.Bind(inputFeature.Current().Name(), frame));
    ```

* FromVideoFrame
    ```
	VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
    VERIFY_NO_THROW(m_modelBinding.Bind(inputFeature.Current().Name(), frame));
    ```

* FromGPUResource
    > Bind input from GPU Resources as described in this [PR](https://mscodehub.visualstudio.com/_git/WindowsAI/pullrequest/4542?_a=overview)

* FromCPUResource
    > Bind input from GPU Resources as described in this [PR](https://mscodehub.visualstudio.com/_git/WindowsAI/pullrequest/4507?_a=overview)

b. Multiple inputs (**TODO**)
1. test the case where 2 inputs requiring GPU processing are bound back to back forcing bind calls that potentially overlap GPU work from previous bind
2. Test models require multiple inputs

c. Bind with Property (**TODO**)
* _If bind with Property, what properties should be tested? And how to make properties as parameters?_

### 7. InputPixelFormats
> We could convert the input image to the following PixelFormats, and then used as input of model. The pixel format expected from model input can be different from the pixel format of real input. But WinML can handle the conversion between them.

* Bgra8
* Rgba8
* Gray8

# Implementation
Googletest's [value parameterized tests](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#value-parameterized-tests) are used.
