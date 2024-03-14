---
title: Python phi-2 tutorial
description: Learn how to write a language generation application with ONNX Runtime GenAI in Python using the phi-2 model
has_children: false
parent: Tutorials
grand_parent: Generative AI (Preview)
nav_order: 1
---

# Language generation in Python with phi-2

## Setup and installation

Install the ONNX Runtime GenAI Python package using the [installation instructions](../howto/install.md).

## Build phi-2 ONNX model

The onnxruntime-genai package contains a model builder that generates the phi-2 ONNX model using the weights and config on Huggingface. The tools also allows you to load locally stored weights, or convert from GGUF format. For more details, see [how to build models](../howto/build-models.md)

If using the `-m` option shown here, which downloads from HuggingFace, you will need to login into HuggingFace.

```bash
pip install huggingface-hub`
huggingface-cli --login
```

You can build the model in different precisions. This command uses int4 as it produces the smallest model and can run on a CPU.

```bash
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o ./example-models/phi2-int4-cpu
```
You can replace the name of the output folder specified with the `-o` option with a folder of your choice.

After you run the script, you will see a series of files generated in this folder. They include the HuggingFace configs for your reference, as well as the following generated files used by ONNX Runtime GenAI.

- `model.onnx`: the phi-2 ONNX model
- `model.onnx.data`: the phi-2 ONNX model weights
- `genai_config.json`: the configuration used by ONNX Runtime GenAI

You can view and change the values in the `genai_config.json` file. The model section should not be updated unless you have brought your own model and it has different parameters. 

The search parameters can be changed. For example, you might want to generate with a different temperature value. These values can also be set via the `set_search_options` method shown below.

## Run the model with a sample prompt

Run the model with the following Python script. You can change the prompt and other parameters as needed.

```python
import onnxruntime_genai as og

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

model=og.Model(f'example-models/phi2-int4-cpu')

tokenizer = model.create_tokenizer()

tokens = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":200})
params.input_ids = tokens

output_tokens=model.generate(params)[0]

text = tokenizer.decode(output_tokens)

print(text)
```

## Run batches of prompts

You can also run batches of prompts through the model.

```python
prompts = [
    "This is a test.",
    "Rats are awesome pets!",
    "The quick brown fox jumps over the lazy dog.",
    ]

inputs = tokenizer.encode_batch(prompts)

params=og.GeneratorParams(model)
params.input_ids = tokens

outputs = model.generate(params)[0]

text = tokenizer.decode(output_tokens)
```

## Stream the output of the tokenizer

If you are developing an application that requires tokens to be output to the user interface one at a time, you can use the streaming tokenizer.

```python
generator=og.Generator(model, params)
tokenizer_stream=tokenizer.create_stream()

print(prompt, end='', flush=True)

while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token_top_p(0.7, 0.6)
    print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end='', flush=True)
```
