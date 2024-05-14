---
title: Setup CUDA env
description: Instructions to setup the CUDA environtment to run onnxruntime-genai-cuda
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 4
---

# Setup the CUDA Environment
{: .no_toc }

* TOC placeholder
{:toc}

## Install the CUDA Toolkit

On a CUDA capable machine, install the CUDA toolkit. onnxruntime-genai-cuda is built and packaged with CUDA-11.8.

The CUDA toolkit can be downloaded from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

## Special Instructions on Windows

After CUDA toolkit installation completed on windows, ensure that the `CUDA_PATH` system environment variable has been set to the path where the toolkit was installed. This variable will be used when importing the onnxruntime_genai python module on windows. Unset or incorrectly set `CUDA_PATH` variable may lead to a `DLL load failed while importing onnxruntime_genai`.
