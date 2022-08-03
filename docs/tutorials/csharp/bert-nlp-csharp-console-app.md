---
title: Inference with C# BERT NLP and ONNX Runtime
description: We will learn how to use BERT in a C# Console App.
parent: Inference with C#
grand_parent: Tutorials
has_children: false
nav_order: 1
---


# Inference with C# BERT NLP and ONNX Runtime
{: .no_toc }

In this tutorial we will look at how we can do inferencing for the popular BERT Natural Language Processing model in a C# console app.

In order to be able to preprocess our text in C# we will leverage the open source [BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers) that includes tokenizers for most BERT models. See below for supported models. There are models (including the one for this tutorial) that have been finetuned based on these models. The tokenizer for the original model is still the same regardless of the finetuning performed. 

- BERT Base
- BERT Large
- BERT German
- BERT Multilingual
- BERT Base Uncased
- BERT Large Uncased

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Prerequisties
This tutorial can be run locally or by leveraging Azure Machine Learning compute.

To run locally:

- [vs code]() and/or [visual studio]()
- [Anacaonda]()

To run in Azure Machine Learning:

- [Azure Subscription]()
- [Azure Machine Learning Resource]()

## Use Hugging Face to download the BERT model

Hugging Face has a great API for downloading open source models and then we can use python and pytorch to export them to onnx format. This is a great option when using an open source model that is not already part of the [ONNX Model Zoo](). 

Below we will go through the step to download and export our model.

- Use the `transformers` API to download the `BertForQuestionAnswering` model named `bert-large-uncased-whole-word-masking-finetuned-squad`

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

- Now that we have downloaded the model we need to export it to an `ONNX` format. This is built into Pytorch with the `torch.onnx.export` function. 

- The `inputs` variable indicates what the input shape will be. You can either create a dummy input like we did below, or use a sample input.

- Set the `opset_version` to the highest that is compatible with the model. Learn more about the opset versions [here]().

- Set the `input_names` and `output_names` for the model.

- Set the `dynamic_axes` for the dynamic length input because the `sentence` and `context` variables will be of different lengths for each question inferenced.

```python
# Generate dummy inputs to the model. Adjust if neccessary
inputs = {
        'input_ids':   torch.randint(32, [1, 32], dtype=torch.long), # list of numerical ids for the tokenized text
        'attention_mask': torch.ones([1, 32], dtype=torch.long),     # dummy list of ones
        'token_type_ids':  torch.ones([1, 32], dtype=torch.long)     # dummy list of ones
    }

symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
torch.onnx.export(model,                                         # model being run
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
## Understanding the model
When taking a prebuilt model and operationalizing it, its useful to take a moment and understand the model pre and post processing, and the input/output shapes and labels. Many models have sample code provided in Python to use the model there. We will be inferencing our model with C# but first lets see how its done in Python. This will help us with our C# logic in the next step.

- Create the `preprocess`, `postprocess`, `init` and `run` fuctions in python to test the model.

```python
import os
import logging
import json
import numpy as np
import onnxruntime
import transformers
import torch

# The pre process function take a question and a context, and generates the tensor inputs to the model:
# - input_ids: the words in the question encoded as integers
# - attention_mask: not used in this model
# - token_type_ids: a list of 0s and 1s that distinguish between the words of the question and the words of the context
# This function also returns the words contained in the question and the context, so that the answer can be decoded into a phrase. 
def preprocess(question, context):
    encoded_input = tokenizer(question, context)
    print(encoded_input)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
    return (encoded_input.input_ids, encoded_input.attention_mask, encoded_input.token_type_ids, tokens)

# The post process function maps the list of start and end log probabilities onto a text answer, using the text tokens from the question
# and context. 
def postprocess(tokens, start, end):
    results = {}
    answer_start = np.argmax(start)
    answer_end = np.argmax(end)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
        results['answer'] = answer.capitalize()
    else:
        results['error'] = "I am unable to find the answer to this question. Can you please ask another question?"
    return results

# Perform the one-off initialization for the prediction. The init code is run once when the endpoint is setup.
def init():
    global tokenizer, session, model

    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = transformers.BertForQuestionAnswering.from_pretrained(model_name)
    model_path = "./" + model_name + ".onnx"

    # Create the tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # Create an ONNX Runtime session to run the ONNX model
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])  


# Run the ONNX model with ONNX Runtime
def run(raw_data):
    logging.info("Request received")
    inputs = json.loads(raw_data)
    logging.info(inputs)

    # Preprocess the question and context into tokenized ids
    input_ids, input_mask, segment_ids, tokens = preprocess(inputs["question"], inputs["context"])
    logging.info("Running inference")
    
    # Format the inputs for ONNX Runtime
    model_inputs = {
        'input_ids':   [input_ids], 
        'input_mask':  [input_mask],
        'segment_ids': [segment_ids]
        }
                  
    outputs = session.run(['start_logits', 'end_logits'], model_inputs)
    logging.info("Post-processing")  

    # Post process the output of the model into an answer (or an error if the question could not be answered)
    results = postprocess(tokens, outputs[0], outputs[1])
    logging.info(results)
    return results

```
- Call the `init()` to download the model and create the `InferenceSession`

```python
init()
```
- Create the `input` and run the model. 
```python
input = "{\"question\": \"What is Dolly Parton's middle name?\", \"context\": \"Dolly Rebecca Parton is an American singer-songwriter\"}"

print(run(input))
```
- Here is what the output should look like for the above question. We will use the `input_ids` to validate our tokenization in C#.

```text
Output:
{'input_ids': [101, 2054, 2003, 19958, 2112, 2239, 1005, 1055, 2690, 2171, 1029, 102, 19958, 9423, 2112, 2239, 2003, 2019, 2137, 3220, 1011, 6009, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'answer': 'Rebecca'}
```
## Inference with C#

Now that we have tested the model in Python its time to build it out in C#. The first thing we need to do is to create our project. For this example we will be using a Console App however you could use this code in any C# application.

- Open Visual Studio and [Create a Console App](https://docs.microsoft.com/en-us/visualstudio/get-started/csharp/tutorial-console?view=vs-2022)

- Install the Nuget packages `BERTTokenizers`, `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.OnnxRuntime.Managed`, `Microsoft.ML`
```PowerShell
dotnet add package Microsoft.ML.OnnxRuntime --version 1.12.0
dotnet add package Microsoft.ML.OnnxRuntime.Managed --version 1.12.0
dotnet add package dotnet add package Microsoft.ML
dotnet add package dotnet add package BERTTokenizers --version 1.1.0
```
- Import the packages

```csharp
using BERTTokenizers;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
```
- Add the namespace, class and Main function.

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

```csharp
    public class BertInput
    {
        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }

        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }

        [ColumnName("token_type_ids")]
        public long[] TypeIds { get; set; }
    }
```

```csharp
  var sentence = "{\"question\": \"Where is Bob Dylan From?\", \"context\": \"Bob Dylan is from Duluth, Minnesota and is an American singer-songwriter\"}";
  Console.WriteLine(sentence);

  var tokenizer = new BertUncasedLargeTokenizer();
  var tokens = tokenizer.Tokenize(sentence);
  // Console.WriteLine(String.Join(", ", tokens));

  // encode sentence based
  var encoded = tokenizer.Encode(tokens.Count(), sentence);

  // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id)
  var bertInput = new BertInput()
  {
      InputIds = encoded.Select(t => t.InputIds).ToArray(),
      AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
      TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
  };

  //Console.WriteLine(String.Join(", ", bertInput.InputIds));

  // Get path to model to create inference session.
  var modelPath = @"C:\code\bert-nlp-csharp\BertNlpTest\BertNlpTest\bert-large-uncased-finetuned-qa.onnx";

  // create input tensor

  var input_ids = ConvertToTensor(bertInput.InputIds, bertInput.InputIds.Length);
  var attention_mask = ConvertToTensor(bertInput.AttentionMask, bertInput.InputIds.Length);
  var token_type_ids = ConvertToTensor(bertInput.TypeIds, bertInput.InputIds.Length);


  // Create input data for session.
  var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids), 
                                         NamedOnnxValue.CreateFromTensor("input_mask", attention_mask), 
                                         NamedOnnxValue.CreateFromTensor("segment_ids", token_type_ids) };

  // Create an InferenceSession from the Model Path.
  var session = new InferenceSession(modelPath);

  // Run session and send input data in to get inference output. Call ToList then get the Last item. Then usethe AsEnumerable extension method to return the Value result as an Enumerable of NamedOnnxValue.
  var output = session.Run(input);
  //var startLogits = output.ToList().First().Value;
  List<float> startLogits = (output.ToList().First().Value as IEnumerable<float>).ToList();
  List<float> endLogits = (output.ToList().Last().Value as IEnumerable<float>).ToList();

  //argmax results
  var startIndex = startLogits.ToList().IndexOf(startLogits.Max()); 
  var endIndex = endLogits.ToList().IndexOf(endLogits.Max());

  //from list of tokens
  //get token from the index of the highest value from start to end
  var predictedTokens =tokens
              .Skip(startIndex)
              .Take(endIndex + 1 - startIndex)
              .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
              .ToList();

  Console.WriteLine(String.Join(" ", predictedTokens));
```

- Create the `ConvertToTensor` function

```csharp
        public static Tensor<long> ConvertToTensor(long[] inputArray, int inputDimension)
        {
            Tensor<long> input = new DenseTensor<long>(new[] { 1, inputDimension });
            for (var i = 0; i < inputArray.Length; i++)
            {
                input[0,i] = inputArray[i];
            }
            return input;
        }
```

## Deploy with Azure Web App

In this example we created a simple console app however this could easily be implmented in something like a C# Web App. Check out the docs on how to [Quickstart: Deploy an ASP.NET web app](https://docs.microsoft.com/en-us/azure/app-service/quickstart-dotnetcore?tabs=net60&pivots=development-environment-vs).

## More Resources

