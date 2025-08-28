---
title: Run with LoRA adapters
description: Use Olive and ONNX Runtime to generate and run fine-tuned LoRA adapters
has_children: false
parent: Tutorials
grand_parent: Generate API (Preview)
nav_order: 4
---

# Generate and run fine-tuned models with LoRA adapters
{: .no_toc }

Learn how to generate models and adapters in formats suitable for executing with ONNX Runtime.

LoRA stands for Low Rank Adaptation. It is a popular method of fine-tuning that freezes some layers in a graph and provides the values of the weights of the variable layers in an artifact called an adapter. 

Multi LoRA uses multiple adapters at runtime to run different fine-tunings of the same model. The adapter could be per-scenario, per-tenant/customer, or per-user i.e. there could be just a few adapters to many hundreds or thousands.

Olive generates models and adapters in ONNX format. These models and adapters can then be run with ONNX Runtime.

## Setup

1. Install Olive
   
   This installs Olive from main. Replace with version 0.8.0 when it is released.

   ```bash
   pip install git+https://github.com/microsoft/olive
   ```

2. Install ONNX Runtime generate()

   ```
   pip install onnxruntime-genai
   ```

3. Install other dependencies

   ```bash
   pip install optimum peft
   ```

4. Downgrade torch and transformers

   TODO: There is an export bug with torch 2.5.0 and an incompatibility with transformers>=4.45.0

   ```bash
   pip uninstall torch
   pip install torch==2.4
   pip uninstall transformers
   pip install transformers==4.44
   ```
   
5. Choose a model

   You can use a model from HuggingFace, or your own model. The model must be a PyTorch model.
   
6. Decide whether you are fine-tuning your model, or using a pre-existing adapter

   There are many pre-existing adapters on HuggingFace. If you are using multiple different adapters, these must all use the same fine-tuned layers of the original model.

## Generate model and adapters in ONNX format

1. If fine-tuning, run Olive to fine-tune your model

   Note: this operations requires a system with an NVIDIA GPU, with CUDA installed

   Use the `olive fine-tune` command: https://microsoft.github.io/Olive/how-to/cli/cli-finetune.html

   Here is an example usage of the command:

   ```bash
   olive finetune --method qlora -m meta-llama/Meta-Llama-3-8B -d nampdn-ai/tiny-codes --train_split "train[:4096]" --eval_split "train[4096:4224]" --text_template "### Language: {programming_language} \n### Question: {prompt} \n### Answer: {response}" --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --max_steps 150 --logging_steps 50 -o adapters\tiny-codes
   ```

2. Optionally, quantize your model

   Use the `olive quantize` command: https://microsoft.github.io/Olive/how-to/cli/cli-quantize.html


3. Generate the ONNX model and adapter using the quantized model

   Use the `olive auto-opt` command for this step: https://microsoft.github.io/Olive/how-to/cli/cli-auto-opt.html

   The `--adapter path` can either be a HuggingFace adapter reference, or a path to the adapter you fine-tuned above.

   The `--provider` argument can be an ONNX Runtime execution provider.

   ```bash
   olive auto-opt -m <path to your model folder> --adapter_path <path to your adapter> -o <output model folder> --device cpu\|gpu --provider <provider> 
   ```

4. Convert adapters to `.onnx_adapter` format

   Run this step once for each adapter that you have generated.

   ```bash
   olive convert-adapters --adapter_path <path to your fine-tuned adapter --output_path <path to .onnx_adapter location --dtype float32
   ```

## Write your application

This example is shown in Python, but you can also use the C/C++ API, the C# API, and the Java API (_coming soon!_)

```python
import onnxruntime_genai as og
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Application to load and switch ONNX LoRA adapters')
parser.add_argument('-m', '--model', type=str, help='The ONNX base model')
parser.add_argument('-a', '--adapters', nargs='+', type=str, help='List of adapters in .onnx_adapters format')
parser.add_argument('-t', '--template', type=str, help='The template with which to format the prompt')
parser.add_argument('-s', '--system', type=str, help='The system prompt to pass to the model')
parser.add_argument('-p', '--prompt', type=str, help='The user prompt to pass to the model')
args = parser.parse_args()

model = og.Model(args.model)
if args.adapters:
    adapters = og.Adapters(model)
    for adapter in args.adapters:
        adapters.load(adapter, adapter)

tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

prompt = args.template.format(system=args.system, input=args.prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=2048, past_present_share_buffer=False)
# This input is generated for transformers versions > 4.45
#params.set_model_input("onnx::Neg_67", np.array(0, dtype=np.int64))
params.input_ids = tokenizer.encode(prompt)

generator = og.Generator(model, params)

if args.adapters:
   for adapter in args.adapters:
      print(f"[{adapter}]: {prompt}")
      generator.set_active_adapter(adapters, adapter)

      while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)
else:
    print(f"[Base]: {prompt}")

    while not generator.is_done():
       generator.compute_logits()
       generator.generate_next_token()
```

## Call the application

```bash
python app.py -m <model folder> -a <.onnx_adapter files> -t <prompt template> -s <system prompt> -p <prompt>
```

## References

* [Python API docs](../api/python.md)
* [Olive CLI docs](https://microsoft.github.io/Olive/how-to/index.html#working-with-the-cli)
