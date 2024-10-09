---
title: Troubleshoot
description: How to troubleshoot common problems
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 4
---

# Troubleshoot issues with ONNX Runtime generate() API
{: .no_toc }

* TOC placeholder
{:toc}

## Installation issues

### Windows Conda import error

```
ImportError: DLL load failed while importing onnxruntime_genai: A dynamic link library (DLL) initialization routine failed.
```

If you see this issue in a Conda environment on Windows, you need to upgrade the `C++ runtime for Visual Studio`. In the conda environment, run the following command:

```bash
conda install conda-forge::vs2015_runtime
```

The onnxruntime-genai Python package should run without error after this extra step.

### Windows CUDA import error

```
DLL load failed while importing onnxruntime_genai
```

After CUDA toolkit installation completed on windows, ensure that the `CUDA_PATH` system environment variable has been set to the path where the toolkit was installed. This variable will be used when importing the onnxruntime_genai python module on Windows. Unset or incorrectly set `CUDA_PATH` variable may lead to a `DLL load failed while importing onnxruntime_genai`.

### Transformers / Tokenizers incompatibility with ONNX Runtime generate()

```
RuntimeError: [json.exception.type_error.302] type must be string, but is array
```

Only occurs when you generate models with the Model Builder, not with downloaded models.

There was a change in the HuggingFace transformers version 4.45.0 that caused an incompatibility with onnxruntime-genai versions 0.4.0 and earlier, resolved in 0.5.0. There are two alternative workarounds that you can employ to fix this issue:

- Option 1: downgrade your transformers version to lower than v4.45.0 (the version in which the above [change](https://github.com/huggingface/transformers/pull/32535) was introduced)
- Option 2: build onnxruntime-genai from source, using these instructions [build from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)
