
# WebNN Overview

As the use of AI/ML in apps become more popular, the WebNN API provides the following benefits: 

* *Performance Optimizations* – By utilizing DirectML, WebNN helps to enable web apps and frameworks to take advantage of the best available hardware and software optimizations for each platform and device, without requiring complex and platform-specific code. 
* *Low Latency* - In-browser inference helps enable novel use cases with local media sources, such as real-time video analysis, face detection, and speech recognition, without the need to send data to remote servers and wait for responses. 
* *Privacy Preservation* - User data stays on-device and preserves user-privacy, as web apps and frameworks do not need to upload sensitive or personal information to cloud services for processing. 
* *High Availability* - No reliance on the network after initial asset caching for offline case, as web apps and frameworks can run neural network models locally even when the internet connection is unavailable or unreliable. 
* *Low Server Cost* - Computing on client devices means no servers needed, which helps web apps to reduce the operational and maintenance costs of running AI/ML services in the cloud. 

AI/ML scenarios supported by WebNN include generative AI, person detection, face detection, semantic segmentation, skeleton detection, style transfer, super resolution, image captioning, machine translation, and noise suppression.

> [!NOTE]
> The WebNN API is still in progress, with GPU support in a preview state and NPU support coming soon. The WebNN API should not currently be used in a production environment.

## WebNN requirements

You can check information about your browser by navigating to about://version in your chromium browser's address bar.

To view WebNN hardware and OS requirements, visit the [WebNN Overview Doc](https://learn.microsoft.com/en-us/windows/ai/directml/webnn-overview#webnn-requirements).

![Diagram of the structure behind integrating WebNN into your web app](images/webnn-diagram.png)

> [!NOTE]
> Chromium based browsers can currently support WebNN, but will depend on the individual browser's implementation status.

## Model support 

You can a set of demos and examples applications on our [Developer Preview Site](https://microsoft.github.io/webnn-developer-preview/).

### GPU (Preview):
When running on GPUs, WebNN currently supports the following models:

* [Stable Diffusion Turbo](https://microsoft.github.io/webnn-developer-preview/demos/sd-turbo/)
* [Stable Diffusion 1.5](https://microsoft.github.io/webnn-developer-preview/demos/stable-diffusion-1.5/)
* [Whisper-base](https://microsoft.github.io/webnn-developer-preview/demos/whisper-base/)
* [MobileNetv2](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/)
* [Segment Anything](https://microsoft.github.io/webnn-developer-preview/demos/segment-anything/)
* [ResNet](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/?provider=webnn&devicetype=gpu&model=resnet-50&run=5)
* [EfficientNet](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/?provider=webnn&devicetype=gpu&model=efficientnet-lite4&run=5)
* SqueezeNet 

WebNN also works with custom models as long as operator support is sufficient. Check status of operators [here](https://webmachinelearning.github.io/webnn-status/).

### NPU (Coming Soon): 
On Intel’s® Core™ Ultra processors with Intel® AI Boost NPU, WebNN aims to support: 

* [Whisper-base](https://microsoft.github.io/webnn-developer-preview/demos/whisper-base/)
* [MobileNetV2](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/)
* [ResNet](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/?provider=webnn&devicetype=npu&model=resnet-50&run=5)
* [EfficientNet](https://microsoft.github.io/webnn-developer-preview/demos/image-classification/?provider=webnn&devicetype=npu&model=efficientnet-lite4&run=5)
* ESRGAN 
 

## FAQ

#### **How do I file an issue with WebNN or WebNN execution provider?**

For general issues with WebNN, please file an issue on our [WebNN Developer Preview GitHub](https://github.com/microsoft/webnn-developer-preview/issues)

For issues with ONNX Runtime Web or the WebNN Execution Provider, go to the [ONNXRuntime Github](https://github.com/microsoft/onnxruntime/issues).

#### **How do I debug issues with WebNN or WebNN execution provider?**

The [WebNN W3C Spec](https://www.w3.org/TR/webnn/) has information on error propagation, typically through DOM exceptions. The log at the end of about://gpu may also have helpful information. For further issues please file an issue as linked above.

#### Does WebNN support other operating systems?

Currently, WebNN best supports the Windows operating system. Versions for other operating systems are in progress.

#### What hardware back-ends are currently available? Are certain models only supported with specific hardware back-ends?

You can find information about operator support in WebNN at [Implementation Status of WebNN Operations | Web Machine Learning](https://webmachinelearning.github.io/webnn-status/).

#### What are the steps to update the Intel driver for NPU Support (Coming Soon)? 

1. Uncompress the ZIP file. 
2. Press Win+R to open the Run dialog box. 
3. Type devmgmt.msc into the text field. 
4. Press Enter or click OK. 
5. In the Device Manager, open the "Neural processors" node 
6. Right click on the NPU who's driver you wish to update. 
7. Select "Update Driver" from the context menu 
8. Select "Browse my computer for drivers" 
9. Select "Let me pick from a list of available drivers on my computer" 
10. Press the "Have disk" button 
11. Press the "Browse" button 
12. Navigate to the place where you decompressed the aforementioned zip file. 
13. Press OK.


# WebNN API Tutorial

This tutorial will show you how to use WebNN with onnxruntime-web to build an image classification system on the web that is hardware accelerated using on-device GPU. We will be leveraging the **MobileNetv2** model, which is an open source model on [Hugging Face](https://huggingface.co/docs/transformers/model_doc/mobilenet_v2) used to classify images. 

If you want to view and run the final code of this tutorial, you can find it on our [WebNN Developer Preview GitHub](https://github.com/microsoft/webnn-developer-preview/tree/main/Get%20Started/WebNN%20Tutorial).

## Requirements and set-up: 

**Setting Up Windows**

Ensure you have the correct versions of Edge, Windows, and hardware drivers as detailed in the [WebNN Requirements section](webnn-overview.md#webnn-requirements).

**Setting Up Edge**

1. Download and install [Microsoft Edge Dev](https://www.microsoft.com/en-us/edge/download/insider?form=MA13FJ). 

2. Launch Edge Beta, and navigate to `about:flags` in the address bar.

3. Search for "WebNN API", click the dropdown, and set to 'Enabled'.

4. Restart Edge, as prompted.

![An image of WebNN enabled in the Edge beta](images/webnn-edge-flags.png)

**Setting Up Developer Environment**

1. Download and install [Visual Studio Code (VSCode)](https://code.visualstudio.com/). 

2. Launch VSCode.

3. Download and install the [Live Server extension for VSCode](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) within VSCode.

4. Select `File --> Open Folder`, and create a blank folder in your desired location.

## Step 1: Initialize the web app

1. To begin, create a new `index.html` page. Add the following boilerplate code to your new page:
 
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Website</title>
  </head>
  <body>
    <main>
        <h1>Welcome to My Website</h1>
    </main>
  </body>
</html>
```
2. Verify the boilerplate code and developer setup worked by selecting the **Go Live** button at the bottom right hand side of VSCode. This should launch a local server in Edge Beta running the boilerplate code.
3. Now, create a new file called `main.js`. This will contain the javascript code for your app.
4. Next, create a subfolder off the root directory named `images`. Download and save any image within the folder. For this demo, we'll use the default name of `image.jpg`.
5. Download the **mobilenet** model from the [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet). For this tutorial, you'll be using the [mobilenet2-10.onnx](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx) file. Save this model to the root folder of your web app.
6. Finally, download and save this [image classes file](https://github.com/microsoft/webnn-developer-preview/blob/33a497a6747eb7a0a9146a78335f15ed1bea57be/Get%20Started/WebNN%20Tutorial/imagenetClasses.js), `imagenetClasses.js`. This provides 1000 common classifications of images for your model to use.

## Step 2: Add UI elements and parent function

1. Within the body of the `<main>` html tags you added in the previous step, replace the existing code with the following elements. These will create a button and display a default image.

```html
<h1>Image Classification Demo!</h1> 
<div><img src="./images/image.jpg"></div> 
<button onclick="classifyImage('./images/image.jpg')"  type="button">Click Me to Classify Image!</button> 
<h1 id="outputText"> This image displayed is ... </h1>
```

2. Now, you'll add **ONNX Runtime Web** to your page, which is a JavaScript library you'll use to access the WebNN API. Within the body of the `<head>` html tags, add the following javascript source links.

```html
<script src="./main.js"></script> 
<script src="imagenetClasses.js"></script>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0-dev.20240311-5479124834/dist/ort.webgpu.min.js"></script> 
```

3. Open your `main.js` file, and add the following code snippet.
 
```js
async function classifyImage(pathToImage){ 
  var imageTensor = await getImageTensorFromPath(pathToImage); // Convert image to a tensor
  var predictions = await runModel(imageTensor); // Run inference on the tensor
  console.log(predictions); // Print predictions to console
  document.getElementById("outputText").innerHTML += predictions[0].name; // Display prediction in HTML
} 
```

## Step 3: Pre-process data
1. The function you just added calls `getImageTensorFromPath`, another function you have to implement. You'll add it below, as well as another async function it calls to retrieve the image itself.

```js
  async function getImageTensorFromPath(path, width = 224, height = 224) {
    var image = await loadImagefromPath(path, width, height); // 1. load the image
    var imageTensor = imageDataToTensor(image); // 2. convert to tensor
    return imageTensor; // 3. return the tensor
  } 

  async function loadImagefromPath(path, resizedWidth, resizedHeight) {
    var imageData = await Jimp.read(path).then(imageBuffer => { // Use Jimp to load the image and resize it.
      return imageBuffer.resize(resizedWidth, resizedHeight);
    });

    return imageData.bitmap;
  }
```

2. You also need to add the `imageDataToTensor` function that is referenced above, which will render the loaded image into a tensor format that will work with our ONNX model. This is a more involved function, though it might seem familiar if you've worked with similar image classification apps before. For an extended explanation, you can view [this ONNX tutorial](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html#imagehelperts).

```js
  function imageDataToTensor(image) {
    var imageBufferData = image.data;
    let pixelCount = image.width * image.height;
    const float32Data = new Float32Array(3 * pixelCount); // Allocate enough space for red/green/blue channels.

    // Loop through the image buffer, extracting the (R, G, B) channels, rearranging from
    // packed channels to planar channels, and converting to floating point.
    for (let i = 0; i < pixelCount; i++) {
      float32Data[pixelCount * 0 + i] = imageBufferData[i * 4 + 0] / 255.0; // Red
      float32Data[pixelCount * 1 + i] = imageBufferData[i * 4 + 1] / 255.0; // Green
      float32Data[pixelCount * 2 + i] = imageBufferData[i * 4 + 2] / 255.0; // Blue
      // Skip the unused alpha channel: imageBufferData[i * 4 + 3].
    }
    let dimensions = [1, 3, image.height, image.width];
    const inputTensor = new ort.Tensor("float32", float32Data, dimensions);
    return inputTensor;
  }
```

## Step 4: Call ONNX Runtime Web

1. You've now added all the functions needed to retrieve your image and render it as a tensor. Now, using the ONNX Runtime Web library that you loaded above, you'll run your model. Note that to use WebNN here, you simply specify `executionProvider = "webnn"` - ONNX Runtime's support makes it very straightforward to enable WebNN.

```js
  async function runModel(preprocessedData) { 
    // Set up environment.
    ort.env.wasm.numThreads = 1; 
    ort.env.wasm.simd = true; 
    // Uncomment for additional information in debug builds:
    // ort.env.wasm.proxy = true; 
    // ort.env.logLevel = "verbose";  
    // ort.env.debug = true; 

    // Configure WebNN.
    const modelPath = "./mobilenetv2-7.onnx";
    const devicePref =  "gpu"; // other option include "npu"
    const options = {
	    executionProviders: [{ name: "webnn", deviceType: devicePref, powerPreference: "default" }],
      freeDimensionOverrides: {"batch": 1, "channels": 3, "height": 224, "width": 224}
      // the key names in freeDimensionOverrides should map to the real input dim names in the model.
      // For example, if a model's only key is batch_size, you only need to set
      // freeDimensionOverrides: {"batch_size": 1}
    };
    modelSession = await ort.InferenceSession.create(modelPath, options); 

    // Create feeds with the input name from model export and the preprocessed data. 
    const feeds = {}; 
    feeds[modelSession.inputNames[0]] = preprocessedData; 
    // Run the session inference.
    const outputData = await modelSession.run(feeds); 
    // Get output results with the output name from the model export. 
    const output = outputData[modelSession.outputNames[0]]; 
    // Get the softmax of the output data. The softmax transforms values to be between 0 and 1.
    var outputSoftmax = softmax(Array.prototype.slice.call(output.data)); 
    // Get the top 5 results.
    var results = imagenetClassesTopK(outputSoftmax, 5);

    return results; 
  } 
```

## Step 5: Post-process data
1. Finally, you'll add a `softmax` function, then add your final function to return the most likely image classification. The `softmax` transforms your values to be between 0 and 1, which is the probability form needed for this final classification.

First, add the following source files for helper libraries **Jimp** and **Lodash** in the head tag of `main.js`.

```js
<script src="https://cdnjs.cloudflare.com/ajax/libs/jimp/0.22.12/jimp.min.js" integrity="sha512-8xrUum7qKj8xbiUrOzDEJL5uLjpSIMxVevAM5pvBroaxJnxJGFsKaohQPmlzQP8rEoAxrAujWttTnx3AMgGIww==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script src="https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js"></script>
```

Now, add these following functions to `main.js`.

```js
// The softmax transforms values to be between 0 and 1.
function softmax(resultArray) {
  // Get the largest value in the array.
  const largestNumber = Math.max(...resultArray);
  // Apply the exponential function to each result item subtracted by the largest number, using reduction to get the
  // previous result number and the current number to sum all the exponentials results.
  const sumOfExp = resultArray 
    .map(resultItem => Math.exp(resultItem - largestNumber)) 
    .reduce((prevNumber, currentNumber) => prevNumber + currentNumber);

  // Normalize the resultArray by dividing by the sum of all exponentials.
  // This normalization ensures that the sum of the components of the output vector is 1.
  return resultArray.map((resultValue, index) => {
    return Math.exp(resultValue - largestNumber) / sumOfExp
  });
}

function imagenetClassesTopK(classProbabilities, k = 5) { 
  const probs = _.isTypedArray(classProbabilities)
    ? Array.prototype.slice.call(classProbabilities)
    : classProbabilities;

  const sorted = _.reverse(
    _.sortBy(
      probs.map((prob, index) => [prob, index]),
      probIndex => probIndex[0]
    )
  );

  const topK = _.take(sorted, k).map(probIndex => {
    const iClass = imagenetClasses[probIndex[1]]
    return {
      id: iClass[0],
      index: parseInt(probIndex[1].toString(), 10),
      name: iClass[1].replace(/_/g, " "),
      probability: probIndex[0]
    }
  });
  return topK;
}
```

2. You've now added all the scripting needed to run image classification with WebNN in your basic web app. Using the Live Server extension for VS Code, you can now launch your basic webpage in-app to see the results of the classification for yourself.

