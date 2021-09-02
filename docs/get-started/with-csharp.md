---
title: C#
parent: Get Started
toc: true
nav_order: 2
---
# C# ORT Quickstart
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Install the Nuget Packages with the .NET CLI

```bash
dotnet add package Microsoft.ML.OnnxRuntime --version 1.2.0
dotnet add package System.Numerics.Tensors --version 0.1.0
```

## Import the libraries

```csharp
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
```

## Create method for inference

This is an Azure Function example:

```csharp
 public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log, ExecutionContext context)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string review = req.Query["review"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            review = review ?? data?.review;

            // Get path to model to create inference session.
            var modelPath = "./model.onnx";

            // create input tensor (nlp example)
            var inputTensor = new DenseTensor<string>(new string[] { review }, new int[] { 1, 1 });

            // Create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("input", inputTensor) };

            // Create an InferenceSession from the Model Path.
            var session = new InferenceSession(modelPath);

            // Run session and send input data in to get inference output. The run is returned as an object but it is a list. Call ToList then get the Last item. Then use the AsEnumerable extension method to return the Value result as an Enumerable of NamedOnnxValue.
            var output = session.Run(input).ToList().Last().AsEnumerable<NamedOnnxValue>();

            // From the Enumerable output create the inferenceResult by getting the First value and using the AsDictionary extension method of the NamedOnnxValue.
            var inferenceResult = output.First().AsDictionary<string, float>();

            // Return the inference result as json.
            return new JsonResult(inferenceResult);
        }
```

## Learn More
- [C# Tutorials](./Tutorials/)
- [C# Github Quickstart Templates](https://github.com/onnxruntime)
- [C# API Reference](./api/csharp-api.html)