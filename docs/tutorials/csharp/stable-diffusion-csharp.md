---
title: Stable Diffusion with C#
description: Generate images from text in C# with stable diffusion and ONNX Runtime.
image: /images/sample-output-stablediff.png
parent: Inference with C#
grand_parent: Tutorials
has_children: false
nav_order: 1
---

# Inference Stable Diffusion with C# and ONNX Runtime
{: .no_toc }

In this tutorial we will learn how to do inferencing for the popular Stable Diffusion deep learning model in C#. Stable Diffusion models take a text prompt and create an image that represents the text. See the example below:

```text
"make a picture of green tree with flowers around it and a red sky" 
```

|----------------------|----------------------|
| <img src="../../../images/sample-output-stablediff.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> | <img src="../../../images/stablediff-example-image.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> |


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Prerequisites
This tutorial can be run locally or in the cloud by leveraging Azure Machine Learning compute.

- [Download the Source Code from GitHub](https://github.com/cassiebreviu/StableDiffusion)

To run locally:

- [Visual Studio](https://visualstudio.microsoft.com/downloads/) or [VS Code](https://code.visualstudio.com/Download)

- A GPU enabled machine with CUDA or DirectML on Windows
    - Configure CUDA EP.  Follow [this tutorial to configure CUDA and cuDNN for GPU with ONNX Runtime and C# on Windows 11](https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html)
    - Windows comes with DirectML support. No additional configuration is needed. Be sure to clone the [`direct-ML-EP`](https://github.com/cassiebreviu/StableDiffusion/tree/direct-ML-EP) branch of this repo if you choose this option.
    - This was built on a GTX 3070 and it has not been tested on anything smaller.


To run in the cloud with Azure Machine Learning:

- [Azure Subscription](https://azure.microsoft.com/free/)
- [Azure Machine Learning Resource](https://azure.microsoft.com/services/machine-learning/)

## Use Hugging Face to download the Stable Diffusion models

The Hugging Face site has a great library of open source models. We will leverage and download the [ONNX Stable Diffusion models from Hugging Face](https://huggingface.co/models?sort=downloads&search=Stable+Diffusion).

 - [Stable Diffusion Models v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/onnx)
 - [Stable Diffusion Models v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx)


Once you have selected a model version repo, click `Files and Versions`, then select the `ONNX` branch. If there isn't an ONNX model branch available, use the `main` branch and convert it to ONNX. See the [ONNX conversion tutorial for PyTorch](https://learn.microsoft.com/windows/ai/windows-ml/tutorials/pytorch-convert-model) for more information.

- Clone the repo:
```text
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 -b onnx
```

- Copy the folders with the ONNX files to the C# project folder `\StableDiffusion\StableDiffusion`. The folders to copy are: `unet`, `vae_decoder`, `text_encoder`, `safety_checker`. 

## Understanding the model in Python with Diffusers from Hugging Face

When taking a prebuilt model and operationalizing it, its useful to take a moment and understand the models in this pipeline. This code is based on the Hugging Face Diffusers Library and Blog. If you want to learn more about how it works [check out this amazing blog post](https://huggingface.co/blog/stable_diffusion) for more details!

## Inference with C#
Now lets start to breakdown how to inference in C#! The `unet` model takes the text embedding of the user prompt created by the [CLIP model](https://huggingface.co/docs/transformers/model_doc/clip) that connects text and image. The latent noisy image is created as a starting point. The scheduler algorithm and the `unet` model work together to denoise the image to create an image that represents the text prompt. Lets look at the code.

## Main Function
The main function sets the prompt, number of inference steps, and the guidance scale. It then calls the `UNet.Inference` function to run the inference.

The properties that need to be set are:
 - `prompt` - The text prompt to use for the image
 - `num_inference_steps` - The number of steps to run inference for. The more steps the longer it will take to run the inference loop but the image quality should improve.
 - `guidance_scale` - The scale for the classifier-free guidance. The higher the number the more it will try to look like the prompt but the image quality may suffer.
 - `batch_size` - The number of images to create
 - `height` - The height of the image. Default is 512 and must be a multiple of 8.
 - `width` - The width of the image. Default is 512 and must be a multiple of 8.

<i>* NOTE: Check out the [Hugging Face Blog](https://huggingface.co/blog/stable_diffusion) for more details.</i>

```csharp
//Default args
var prompt = "make a picture of green tree with flowers around it and a red sky";
// Number of steps
var num_inference_steps = 10;

// Scale for classifier-free guidance
var guidance_scale = 7.5;
//num of images requested
var batch_size = 1;
// Load the tokenizer and text encoder to tokenize and encodethe text.
var textTokenized = TextProcessing.TokenizeText(prompt);
var textPromptEmbeddings = TextProcessing.TextEncode(textTokenized).ToArray();
// Create uncond_input of blank tokens
var uncondInputTokens = TextProcessing.CreateUncondInput();
var uncondEmbedding = TextProcessing.TextEncode(uncondInputTokens).ToArray();
// Concat textEmeddings and uncondEmbedding
DenseTensor<float> textEmbeddings = new DenseTensor<float>(ne[] { 2, 77, 768 });
for (var i = 0; i < textPromptEmbeddings.Length; i++)
{
    textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
    textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
}
var height = 512;
var width = 512;
// Inference Stable Diff
var image = UNet.Inference(num_inference_steps, textEmbeddings,guidance_scale, batch_size, height, width);
// If image failed or was unsafe it will return null.
if( image == null )
{
    Console.WriteLine("Unable to create image, please try again.");
}
```

## Tokenization with ONNX Runtime Extensions

The `TextProcessing` class has the functions to tokenize the text prompt and encoded it with the [CLIP model](https://huggingface.co/docs/transformers/model_doc/clip) text encoder.

Instead of reimplementing the CLIP tokenizer in C#, we can leverage the cross-platform CLIP tokenizer implementation in [ONNX Runtime Extensions](https://github.com/microsoft/onnxruntime-extensions). The ONNX Runtime Extensions has a `custom_op_cliptok.onnx` file tokenizer that is used to tokenize the text prompt. The tokenizer is a simple tokenizer that splits the text into words and then converts the words into tokens. 

- Text Prompt: a sentence or phrase that represents the image you want to create.
```text
make a picture of green tree with flowers aroundit and a red sky
```
- Text Tokenization: The text prompt is tokenized into a list of tokens. Each token id is a number that represents a word in the sentence, then its filled with a blank token to create the `maxLength` of 77 tokens. The token ids are then converted to a tensor of shape (1,77).

- Below is the code to tokenize the text prompt with ONNX Runtime Extensions.

```csharp
public static int[] TokenizeText(string text)
{
            // Create Tokenizer and tokenize the sentence.
            var tokenizerOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\text_tokenizer\\custom_op_cliptok.onnx");

            // Create session options for custom op of extensions
            using var sessionOptions = new SessionOptions();
            var customOp = "ortextensions.dll";
            sessionOptions.RegisterCustomOpLibraryV2(customOp, out var libraryHandle);

            // Create an InferenceSession from the onnx clip tokenizer.
            using var tokenizeSession = new InferenceSession(tokenizerOnnxPath, sessionOptions);

            // Create input tensor from text
            using var inputTensor = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 1 });
            inputTensor.StringTensorSetElementAt(text.AsSpan(), 0);

            var inputs = new Dictionary<string, OrtValue>
            {
                {  "string_input", inputTensor }
            };

            // Run session and send the input data in to get inference output. 
            using var runOptions = new RunOptions();
            using var tokens = tokenizeSession.Run(runOptions, inputs, tokenizeSession.OutputNames);

            var inputIds = tokens[0].GetTensorDataAsSpan<long>();

            // Cast inputIds to Int32
            var InputIdsInt = new int[inputIds.Length];
            for(int i = 0; i < inputIds.Length; i++)
            {
                InputIdsInt[i] = (int)inputIds[i];
            }

            Console.WriteLine(String.Join(" ", InputIdsInt));

            var modelMaxLength = 77;
            // Pad array with 49407 until length is modelMaxLength
            if (InputIdsInt.Length < modelMaxLength)
            {
                var pad = Enumerable.Repeat(49407, 77 - InputIdsInt.Length).ToArray();
                InputIdsInt = InputIdsInt.Concat(pad).ToArray();
            }
            return InputIdsInt;
}

```

```text
tensor([[49406,  1078,   320,  1674,   539,  1901,  2677,   593,  4023,  1630,
           585,   537,   320,   736,  2390, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
```


## Text embedding with the CLIP text encoder model

The tokens are sent to the text encoder model and converted into a tensor of shape (1, 77, 768) where the first dimension is the batch size, the second dimension is the number of tokens and the third dimension is the embedding size.  The text encoder is a [OpenAI CLIP](https://openai.com/blog/clip/) model that connects text to images.

The text encoder creates the text embedding which is trained to encode the text prompt into a vector that is used to guide the image generation. The text embedding is then concatenated with the uncond embedding to create the text embeddings that is sent to the unet model for inferencing.

- Text Embedding: A vector of numbers that represents the text prompt created from the tokenization result. The text embedding is created by the `text_encoder` model. 

```csharp
        public static float[] TextEncoder(int[] tokenizedInput)
        {
            // Create input tensor. OrtValue will not copy, will read from managed memory
            using var input_ids = OrtValue.CreateTensorValueFromMemory<int>(tokenizedInput,
                new long[] { 1, tokenizedInput.Count() });

            var textEncoderOnnxPath = Directory.GetCurrentDirectory().ToString() + ("\\text_encoder\\model.onnx");

            using var encodeSession = new InferenceSession(textEncoderOnnxPath);

            // Pre-allocate the output so it goes to a managed buffer
            // we know the shape
            var lastHiddenState = new float[1 * 77 * 768];
            using var outputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(lastHiddenState, new long[] { 1, 77, 768 });

            string[] input_names = { "input_ids" };
            OrtValue[] inputs = { input_ids };

            string[] output_names = { encodeSession.OutputNames[0] };
            OrtValue[] outputs = { outputOrtValue };

            // Run inference.
            using var runOptions = new RunOptions();
            encodeSession.Run(runOptions, input_names, inputs, output_names, outputs);

            return lastHiddenState;
        }
```

```text
torch.Size([1, 77, 768])
tensor([[[-0.3884,  0.0229, -0.0522,  ..., -0.4899, -0.3066,  0.0675],
         [ 0.0520, -0.6046,  1.9268,  ..., -0.3985,  0.9645, -0.4424],
         [-0.8027, -0.4533,  1.7525,  ..., -1.0365,  0.6296,  1.0712],
         ...,
         [-0.6833,  0.3571, -1.1353,  ..., -1.4067,  0.0142,  0.3566],
         [-0.7049,  0.3517, -1.1524,  ..., -1.4381,  0.0090,  0.3777],
         [-0.6155,  0.4283, -1.1282,  ..., -1.4256, -0.0285,  0.3206]]],
```


## The Inference Loop: UNet model, Timesteps and LMS Scheduler

 
### Scheduler

The scheduler algorithm and the `unet` model work together to denoise the image to create an image that represents the text prompt. There are different scheduler algorithms that can be used, to learn more about them [check out this blog from Hugging Face](https://huggingface.co/docs/diffusers/using-diffusers/schedulers). In this example we will use the `LMSDiscreteScheduler, which was created based on the HuggingFace [scheduling_lms_discrete.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py).

### Timesteps
The inference loop is the main loop that runs the scheduler algorithm and the `unet` model. The loop runs for the number of `timesteps` which are calculated by the scheduler algorithm based on the number of inference steps and other parameters.

For this example we have 10 inference steps which calculated the following timesteps:

```csharp
// Get path to model to create inference session.
var modelPath = Directory.GetCurrentDirectory().ToString() + ("\\unet\\model.onnx");
var scheduler = new LMSDiscreteScheduler();
var timesteps = scheduler.SetTimesteps(numInferenceSteps);
```

```text
tensor([999., 888., 777., 666., 555., 444., 333., 222., 111.,   0.])
```

### Latents

The `latents` is the noisy image tensor that is used in the model input. It is created using the `GenerateLatentSample` function to create a random tensor of shape (1,4,64,64). The `seed` can be set to a random number or a fixed number. If the `seed` is set to a fixed number the same latent tensor will be used each time. This is useful for debugging or if you want to create the same image each time.

```csharp
var seed = new Random().Next();
var latents = GenerateLatentSample(batchSize, height, width,seed, scheduler.InitNoiseSigma);
```

<img src="../../../images/latents-noise-example.png" width="256" height="256" alt="Image of browser inferencing on sample images."/>

### Inference Loop

For each inference step the latent image is duplicated to create the tensor shape of (2,4,64,64), it is then scaled and inferenced with the unet model. The output tensors (2,4,64,64) are split and guidance is applied. The resulting tensor is then sent into the `LMSDiscreteScheduler` step as part of the denoising process and the resulting tensor from the scheduler step is returned and the loop completes again until the `num_inference_steps` is reached. 

```csharp
var modelPath = Directory.GetCurrentDirectory().ToString() + ("\\unet\\model.onnx");
var scheduler = new LMSDiscreteScheduler();
var timesteps = scheduler.SetTimesteps(numInferenceSteps);

var seed = new Random().Next();
var latents = GenerateLatentSample(batchSize, height, width, seed, scheduler.InitNoiseSigma);

// Create Inference Session
using var options = new SessionOptions();
using var unetSession = new InferenceSession(modelPath, options);

var latentInputShape = new int[] { 2, 4, height / 8, width / 8 };
var splitTensorsShape = new int[] { 1, 4, height / 8, width / 8 };

for (int t = 0; t < timesteps.Length; t++)
{
    // torch.cat([latents] * 2)
    var latentModelInput = TensorHelper.Duplicate(latents.ToArray(), latentInputShape);

    // Scale the input
    latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);

    // Create model input of text embeddings, scaled latent image and timestep
    var input = CreateUnetModelInput(textEmbeddings, latentModelInput, timesteps[t]);

    // Run Inference
    using var output = unetSession.Run(input);
    var outputTensor = output[0].Value as DenseTensor<float>;

    // Split tensors from 2,4,64,64 to 1,4,64,64
    var splitTensors = TensorHelper.SplitTensor(outputTensor, splitTensorsShape);
    var noisePred = splitTensors.Item1;
    var noisePredText = splitTensors.Item2;

    // Perform guidance
    noisePred = performGuidance(noisePred, noisePredText, guidanceScale);

    // LMS Scheduler Step
    latents = scheduler.Step(noisePred, timesteps[t], latents);
}

```
## Postprocess the `output` with the VAEDecoder
After the inference loop is complete, the resulting tensor is scaled and then sent to the `vae_decoder` model to decode the image. Lastly the decoded image tensor is converted to an image and saved to disc.

```csharp
public static Tensor<float> Decoder(List<NamedOnnxValue> input)
{
    // Load the model which will be used to decode the latents into image space. 
   var vaeDecoderModelPath = Directory.GetCurrentDirectory().ToString() + ("\\vae_decoder\\model.onnx");
    
    // Create an InferenceSession from the Model Path.
    var vaeDecodeSession = new InferenceSession(vaeDecoderModelPath);

   // Run session and send the input data in to get inference output. 
    var output = vaeDecodeSession.Run(input);
    var result = (output.ToList().First().Value as Tensor<float>);
    return result;
}

public static Image<Rgba32> ConvertToImage(Tensor<float> output, int width = 512, int height = 512, string imageName = "sample")
{
    var result = new Image<Rgba32>(width, height);
    for (var y = 0; y < height; y++)
    {
        for (var x = 0; x < width; x++)
        {
            result[x, y] = new Rgba32(
                (byte)(Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                (byte)(Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                (byte)(Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
            );
        }
    }
    result.Save($@"C:/code/StableDiffusion/{imageName}.png");
    return result;
}
```

The result image:

![image](../../../images/sample-output-stablediff.png)

## Conclusion

This is a high level overview of how to run Stable Diffusion in C#. It covered the main concepts and provided examples on how to implement it. To get the full code, check out the [Stable Diffusion C# Sample](https://github.com/cassiebreviu/StableDiffusion).

## Resources
- [Stable Diffusion C# Sample Source Code](https://github.com/cassiebreviu/StableDiffusion)
- [C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
