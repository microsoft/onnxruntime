---
title: Migrate
description: Learn how to migrate from one version of ONNX Runtime generate() API when there are breaking API changes
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 5
---

# Migrate ONNX Runtime generateA() API from 0.5.2 to 0.6.0

Learn how to migrate from ONNX Runtime generate() version 0.5.2 to version 0.6.0. 

Version 0.6.0 adds support for "chat mode", also known as _continuation_, _continous decoding_, and _interactive decoding_. The introduction of chat mode necessitated a change to the API, which breaks the previous API.

In summary, the new API adds support for `AppendTokens`, which allows turn taking in the conversation. Previously, there was a simple API to `SetInputs`.

Calling `AddTokens` outside of the loop also adds support for system prompt caching.

Note: chat mode and system prompt caching is only supported when running on CPU, NVIDIA GPUs with the CUDA EP, and all GPUs with the Web GPU native EP. It is not supported on NPU or GPUs running with the DirecML EP. For Q&A mode, the migrations described below *are* required.

## Python

### Migrate Python question and answer (single turn) code to 0.6.0

1. Replace calls to `params.input_ids = input_tokens` with `generator.append_tokens(input_tokens)` after the generator object has been created.
2. Remove calls to `generator.compute_logits()`

### Add system prompt caching

1. Create and tokenize the system prompt and call `generator.append_tokens(system_tokens)`. This call can be done before the user is asked for their prompt.

   ```python
   system_tokens = tokenizer.encode(system_prompt)
   generator.append_tokens(system_tokens)
   ```

### Add chat mode

1. Create a loop in your application, and call `generator.append_tokens(prompt)` every time the user provides new input:
   
   ```python
   while True:
       user_input = input("Input: ")
       input_tokens = tokenizer.encode(user_input)
       generator.append_tokens(input_tokens)

       while not generator.is_done():
           generator.generate_next_token()

           new_token = generator.get_next_tokens()[0]
           print(tokenizer_stream.decode(new_token), end='', flush=True)
        except KeyboardInterrupt:
        print()
    ```

## C/C++ 

### Migrate C/C++ question and answer (single turn) code to 0.6.0

1. Replace calls to `params->SetInputSequences(*sequences)` with `generator->AppendTokenSequences(*sequences)`
2. Remove calls to `generator->ComputeLogits()`

## C#

### Migrate C# question and answer (single turn) code to 0.6.0

1. Replace calls to `generatorParams.SetInputSequences(sequences)` with generator.AppendTokenSequences(sequences)`
2. Remove calls to `generator.ComputeLogits()`

## Java

### Migrate Java question and answer (single turn) code to 0.6.0

1. Replace calls to `GeneratorParams::setInput(sequences)` with `Generator::AppendTokenSequences`
2. Remove calls to `Generator::ComputeLogits`