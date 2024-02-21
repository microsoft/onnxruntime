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

Install the ONNX Runtime GenAI Python package using the [installation instructions](../install.md).

## Build phi-2 ONNX model

The onnxruntime-genai package contains a model builder that generates the phi-2 ONNX model using the weights and config on Huggingface. The tools also allows you to load locally stored weights, or convert from GGUF format. For more details, see [how to build models](../how-to/build-models.md)

If using the `-m` option shown here, which downloads from HuggingFace, you will need to login into HuggingFace.

```bash
pip install huggingface-hub`
huggingface-cli --login
```

You can build the model in different precisions. This command uses int4 as it produces the smallest model and can run on a CPU.

```python
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o ./example-models/phi2-int4-cpu
```
You can replace the name of the output folder specified with the `-o` option with a folder of your choice.

After you run the script, you will see a series of files generated in this folder. They include the HuggingFace configs for your reference, as well as the following generated files used by ONNX Runtime GenAI.

`genai_config.json`: the configuration used by ONNX Runtime GenAI
`model.onnx`: the phi-2 ONNX model
`model.onnx.data`: the phi-2 ONNX model weights

## Run the model with a sample prompt

Run the model with the following Python script. You can change the prompt and other parameters as needed.

```python
import onnxruntime_genai as og

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

model=og.Model(f'example-models/phi2-int4-cpu', og.DeviceType.CPU)

tokenizer = model.create_tokenizer()

tokens = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":200})
params.input_ids = tokens

output_tokens=model.generate(params)[0]

text = tokenizer.decode(output_tokens)

print(text)
```

## 