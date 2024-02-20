---
title: Build models
description: How to build models with ONNX Runtime GenAI
has_children: false
parent: How to
grand_parent: Generative AI
nav_order: 2
---

# Generate models using Model Builder
{: .no_toc }

* TOC placeholder
{:toc}

The model builder greatly accelerates creating optimized and quantized ONNX models that run with ONNX Runtime GenAI.

## Current Support
The tool currently supports the following model architectures.

- Gemma
- LLaMA
- Mistral
- Phi

## Usage

### Full Usage
For all available options, please use the `-h/--help` flag.
```
# From wheel:
python3 -m onnxruntime_genai.models.builder --help

# From source:
python3 builder.py --help
```

### Original Model From Hugging Face

This scenario is where your PyTorch model is not downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk).

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_to_save_hf_files

# From source:
python3 builder.py -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_to_save_hf_files
```

### Original Model From Disk

This scenario is where your PyTorch model is already downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk).

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved

# From source:
python3 builder.py -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved
```

### Customized or Finetuned Model

This scenario is where your PyTorch model has been customized or finetuned for one of the currently supported model architectures and your model can be loaded in Hugging Face.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m path_to_local_folder_on_disk -o /path/to/output/folder -p precision -e execution_provider

# From source:
python3 builder.py -m path_to_local_folder_on_disk -o /path/to/output/folder -p precision -e execution_provider
```

### Extra Options

This scenario is for when you want to have control over some specific settings. The below example shows how you can pass key-value arguments to `--extra_options`.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_to_save_hf_files --extra_options filename=decoder.onnx

# From source:
python3 builder.py -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_to_save_hf_files --extra_options filename=decoder.onnx
```

To see all available options through `--extra_options`, please use the `help` commands in the `Full Usage` section above.

### Unit Testing Models

This scenario is where your PyTorch model is already downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk). If it is not already downloaded locally, here is an example of how you can download it.

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
cache_dir = "cache_dir_to_save_hf_files"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
model.save_pretrained(cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.save_pretrained(cache_dir)
```

#### Option 1: Use the model builder tool directly

This option is the simplest but it will download another copy of the PyTorch model onto disk to accommodate the change in the number of hidden layers.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o /path/to/output/folder -p precision -e execution_provider --extra_options num_hidden_layers=4

# From source:
python3 builder.py -m model_name -o /path/to/output/folder -p precision -e execution_provider --extra_options num_hidden_layers=4
```

#### Option 2: Edit the config.json file on disk and then run the model builder tool

1. Navigate to where the PyTorch model and its associated files are saved on disk.
2. Modify `num_hidden_layers` in `config.json` to your desired target (e.g. 4 layers).
3. Run the below command for the model builder tool.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved

# From source:
python3 builder.py -m model_name -o /path/to/output/folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved
```

