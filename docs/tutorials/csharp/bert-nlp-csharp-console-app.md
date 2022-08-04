---
title: Inference with C# BERT NLP Deep Learning and ONNX Runtime
description: We will learn how to use BERT in a C# Console App.
parent: Inference with C#
grand_parent: Tutorials
has_children: false
nav_order: 1
---


# Inference with C# BERT NLP Deep Learning and ONNX Runtime
{: .no_toc }

In this tutorial we will learn how to do inferencing for the popular BERT Natural Language Processing deep learning model in C#.

In order to be able to preprocess our text in C# we will leverage the open source [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers) that includes tokenizers for most BERT models. See below for supported models. 

- BERT Base
- BERT Large
- BERT German
- BERT Multilingual
- BERT Base Uncased
- BERT Large Uncased

There are many models (including the one for this tutorial) that have been fine tuned based on these base models. The tokenizer for the model is still the same as the base model that it was fine tuned from.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Prerequisites
This tutorial can be run locally or by leveraging Azure Machine Learning compute.

To run locally:

- [Visual Studio](https://visualstudio.microsoft.com/downloads/)
- [VS Code](https://code.visualstudio.com/Download) with the [Jupyter notebook extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
- [Anacaonda](https://www.anaconda.com/)

To run in the cloud with Azure Machine Learning:

- [Azure Subscription](https://azure.microsoft.com/free/)
- [Azure Machine Learning Resource](https://azure.microsoft.com/services/machine-learning/)

## Use Hugging Face to download the BERT model

Hugging Face has a great API for downloading open source models and then we can use python and Pytorch to export them to ONNX format. This is a great option when using an open source model that is not already part of the [ONNX Model Zoo](https://github.com/onnx/models). 

### Steps to download and export our model in Python

Use the `transformers` API to download the `BertForQuestionAnswering` model named `bert-large-uncased-whole-word-masking-finetuned-squad`

```python
import torch
from transformers import BertForQuestionAnswering

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model_path = "./" + model_name + ".onnx"
model = BertForQuestionAnswering.from_pretrained(model_name)

# set the model to inference mode
# It is important to call torch_model.eval() or torch_model.train(False) before exporting the model, 
# to turn the model to inference mode. This is required since operators like dropout or batchnorm 
# behave differently in inference and training mode.
model.eval()
```

Now that we have downloaded the model we need to export it to an `ONNX` format. This is built into Pytorch with the `torch.onnx.export` function. 

- The `inputs` variable indicates what the input shape will be. You can either create a dummy input like below, or use a sample input from testing the model.

- Set the `opset_version` to the highest and compatible version with the model. Learn more about the opset versions [here](https://onnxruntime.ai/docs/reference/compatibility.html#:~:text=ONNX%20Runtime%20supports%20all%20opsets%20from%20the%20latest,with%20ONNX%20opset%20versions%20in%20the%20range%20%5B7-9%5D.).

- Set the `input_names` and `output_names` for the model.

- Set the `dynamic_axes` for the dynamic length input because the `sentence` and `context` variables will be of different lengths for each question inferenced.

```python
# Generate dummy inputs to the model. Adjust if neccessary.
inputs = {
        # list of numerical ids for the tokenized text
        'input_ids':   torch.randint(32, [1, 32], dtype=torch.long), 
        # dummy list of ones
        'attention_mask': torch.ones([1, 32], dtype=torch.long),     
        # dummy list of ones
        'token_type_ids':  torch.ones([1, 32], dtype=torch.long)     
    }

symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
torch.onnx.export(model,                                         
# model being run
                  (inputs['input_ids'],
                   inputs['attention_mask'], 
                   inputs['token_type_ids']),                    # model input (or a tuple for multiple inputs)
                  model_path,                                    # where to save the model (can be a file or file-like object)
                  opset_version=11,                              # the ONNX version to export the model to
                  do_constant_folding=True,                      # whether to execute constant folding for optimization
                  input_names=['input_ids',
                               'input_mask', 
                               'segment_ids'],                   # the model's input names
                  output_names=['start_logits', "end_logits"],   # the model's output names
                  dynamic_axes={'input_ids': symbolic_names,
                                'input_mask' : symbolic_names,
                                'segment_ids' : symbolic_names,
                                'start_logits' : symbolic_names, 
                                'end_logits': symbolic_names})   # variable length axes/dynamic input
```
## Understanding the model in Python
When taking a prebuilt model and operationalizing it, its useful to take a moment and understand the models pre and post processing, and the input/output shapes and labels. Many models have sample code provided in Python. We will be inferencing our model with C# but first lets test it and see how its done in Python. This will help us with our C# logic in the next step.

- The code to test out the model is provided [in this tutorial](https://onnxruntime.ai/docs/tutorials/azureml.html). Check out the source for testing and inferencing this model in Python. Below is a sample `input` sentence and a sample `output` from running the model.

- Sample `input`

```python
input = "{\"question\": \"What is Dolly Parton's middle name?\", \"context\": \"Dolly Rebecca Parton is an American singer-songwriter\"}"

print(run(input))
```

- Here is what the output should look like for the above question. You can use the `input_ids` to validate the tokenization in C#.

```text
Output:
{'input_ids': [101, 2054, 2003, 19958, 2112, 2239, 1005, 1055, 2690, 2171, 1029, 102, 19958, 9423, 2112, 2239, 2003, 2019, 2137, 3220, 1011, 6009, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'answer': 'Rebecca'}
```
## Inference with C#

Now that we have tested the model in Python its time to build it out in C#. The first thing we need to do is to create our project. For this example we will be using a Console App however you could use this code in any C# application.

- Open Visual Studio and [Create a Console App](https://docs.microsoft.com/en-us/visualstudio/get-started/csharp/tutorial-console?view=vs-2022)

### Install the Nuget Packages
- Install the Nuget packages `BERTTokenizers`, `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.Managed`, `Microsoft.ML`
```PowerShell
dotnet add package Microsoft.ML.OnnxRuntime --version 1.12.0
dotnet add package Microsoft.ML.OnnxRuntime.Managed --version 1.12.0
dotnet add package dotnet add package Microsoft.ML
dotnet add package dotnet add package BERTTokenizers --version 1.1.0
```

### Create the App

- Import the packages

```csharp
using BERTTokenizers;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
```
- Add the `namespace`, `class` and `Main` function.

```csharp

namespace MyApp // Note: actual namespace depends on the project name.
{
    internal class Program
    {
        static void Main(string[] args)
        {

        }
    }
}
```
### Create the BertInput class for encoding
- Add the `BertInput` class

```csharp
    public class BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public long[] TypeIds { get; set; }
    }
```
### Tokenize the sentence with the `BertUncasedLargeTokenizer`
- Create a sentence (question and context) and tokenize the sentence with the `BertUncasedLargeTokenizer`. The base model for this fine tuned model was the BERT Uncased Large so the tokenizer is the same.

```csharp
  var sentence = "{\"question\": \"Where is Bob Dylan From?\", \"context\": \"Bob Dylan is from Duluth, Minnesota and is an American singer-songwriter\"}";
  Console.WriteLine(sentence);

  // Create Tokenizer and tokenize the sentence.
  var tokenizer = new BertUncasedLargeTokenizer();

  // Get the sentence tokens.
  var tokens = tokenizer.Tokenize(sentence);
  // Console.WriteLine(String.Join(", ", tokens));

  // Encode the sentence and pass in the count of the tokens in the sentence.
  var encoded = tokenizer.Encode(tokens.Count(), sentence);

  // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
  var bertInput = new BertInput()
  {
      InputIds = encoded.Select(t => t.InputIds).ToArray(),
      AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
      TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
  };
 
```
### Create the Tensors
- Create the `ConvertToTensor` function. Set the shape of the Tensor `new[] { 1, inputDimension }` and the values to be added to the `NamedOnnxValue` input list.

```csharp
        public static Tensor<long> ConvertToTensor(long[] inputArray, int inputDimension)
        {
            // Create a tensor with the shape the model is expecting. Here we are sending in 1 batch with the inputDimension as the amount of tokens.
            Tensor<long> input = new DenseTensor<long>(new[] { 1, inputDimension });

            // Loop through the inputArray (InputIds, AttentionMask and TypeIds)
            for (var i = 0; i < inputArray.Length; i++)
            {
                // Add each to the input Tenor result.
                // Set index and array value of each input Tensor.
                input[0,i] = inputArray[i];
            }
            return input;
        }
```

### Create the `input` of `List<NamedOnnxValue>` that is needed for inference

- Get the model, call the `ConvertToTensor` function to create the tensor and create the list of `NamedOnnxValue` input variables for inferencing.

```csharp
  // Get path to model to create inference session.
  var modelPath = @"C:\code\bert-nlp-csharp\BertNlpTest\BertNlpTest\bert-large-uncased-finetuned-qa.onnx";

  // Create input tensor.

  var input_ids = ConvertToTensor(bertInput.InputIds, bertInput.InputIds.Length);
  var attention_mask = ConvertToTensor(bertInput.AttentionMask, bertInput.InputIds.Length);
  var token_type_ids = ConvertToTensor(bertInput.TypeIds, bertInput.InputIds.Length);


  // Create input data for session.
  var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids), 
                                         NamedOnnxValue.CreateFromTensor("input_mask", attention_mask), 
                                         NamedOnnxValue.CreateFromTensor("segment_ids", token_type_ids) };


```
### Run Inference
- Create the `InferenceSession`, run the inference and print out the result.

```csharp
  // Create an InferenceSession from the Model Path.
  var session = new InferenceSession(modelPath);

  // Run session and send the input data in to get inference output. 
  var output = session.Run(input);
```
### Postprocess the `output` and print the result

- Here we get the index for the start position (`startLogit`) and end position (`endLogits`). Then we take the original `tokens` of the input sentence and get the vocabulary value for the token ids predicted.

```csharp
  // Call ToList on the output.
  // Get the First and Last item in the list.
  // Get the Value of the item and cast as IEnumerable<float> to get a list result.
  List<float> startLogits = (output.ToList().First().Value as IEnumerable<float>).ToList();
  List<float> endLogits = (output.ToList().Last().Value as IEnumerable<float>).ToList();

  // Get the Index of the Max value from the output lists.
  var startIndex = startLogits.ToList().IndexOf(startLogits.Max()); 
  var endIndex = endLogits.ToList().IndexOf(endLogits.Max());

  // From the list of the original tokens in the sentence
  // Get the tokens between the startIndex and endIndex and convert to the vocabulary from the ID of the token.
  var predictedTokens = tokens
              .Skip(startIndex)
              .Take(endIndex + 1 - startIndex)
              .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
              .ToList();

  // Print the result.
  Console.WriteLine(String.Join(" ", predictedTokens));
  ```

## Deploy with Azure Web App

In this example we created a simple console app however this could easily be implemented in something like a C# Web App. Check out the docs on how to [Quickstart: Deploy an ASP.NET web app](https://docs.microsoft.com/en-us/azure/app-service/quickstart-dotnetcore?tabs=net60&pivots=development-environment-vs).

## Next steps

There are many different BERT models that have been fine tuned for different tasks and different base models you could fine tune for your specific task. This code will work for most BERT models, just update the input and output and postprocessing for your specific model.

- [C# API Doc](https://onnxruntime.ai/docs/api/csharp-api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
