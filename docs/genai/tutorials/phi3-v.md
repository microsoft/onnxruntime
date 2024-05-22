---
title: Phi-3 vision tutorial
description: Small and mighty useful. Run Phi-3 vision with ONNX Runtime.
has_children: false
parent: Tutorials
grand_parent: Generate API (Preview)
nav_order: 1
---

# Run the Phi-3 vision model with the ONNX Runtime generate() API
{: .no_toc }

The Phi-3 vision model is a small, but powerful multi modal model that allows you to use both image and text to output text. It is used in scenarios such as describing the content of images in detail.

The Phi-3 vision model is supported by versions of onnxruntime-genai 0.3.0 and later.

You can download the models here:

* https://microsoft/Phi-3-vision-128k-instruct-onnx-cpu
* https://microsoft/Phi-3-vision-128k-instruct-onnx-cuda

Support for DirectML is coming soon!

* TOC placeholder
{:toc}

## Setup

1. Install the git large file system extension

  HuggingFace uses `git` for version control. To download the ONNX models you need `git lfs` to be installed, if you do not already have it.

  * Windows: `winget install -e --id GitHub.GitLFS` (If you don't have winget, download and run the `exe` from the [official source](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows))
  * Linux: `apt-get install git-lfs`
  * MacOS: `brew install git-lfs`

  Then run `git lfs install`

2. Install the HuggingFace CLI

   ```bash
   pip install huggingface-hub[cli]
   ```

## Choose your platform

If you have an NVIDIA GPU, that will give the best performance right now.

The models will also run on CPU, but they will be slower.

Support for Windows machines with GPUs other than NVIDIA ones is coming soon!
 
**Note: Only one package and model is required based on your hardware. That is, only execute the steps for one of the following sections**

## Run with NVIDIA CUDA

1. Download the model

   ```bash
   huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx-cuda --include cuda-int4-rtn-block-32/* --local-dir .
   ```
   This command downloads the model into a folder called `cuda-int4-rtn-block-32`.

2. Install the generate() API

   ```
   pip install numpy
   pip install --pre onnxruntime-genai-cuda --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
   ```

3. Run the model

   Run the model with [phi3v.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py).

   ```bash
   curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3v.py -o phi3v.py
   python phi3v.py -m cuda-int4-rtn-block-32 
   ```

   You enter the path to an image file and a prompt, and the model uses the image and prompt to give you an answer.

   For example: `Describe the image`

   ![Sushi](sushi.png) 

   ```
   The image shows a package of California Salad Roll SP from Metropolitan Market. The package is transparent, allowing the view of the salad rolls inside. The label on the package indicates the net weight as 340 Calories and the price as $7.49. The package is marked 'Sell By 05/05'
   ```

## Run on CPU

1. Download the model

   ```bash
   huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx-cpu --include cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
   ```

   This command downloads the model into a folder called `cpu-int4-rtn-block-32-acc-level-4`

2. Install the generate() API for CPU
   
   ```
   pip install numpy
   pip install --pre onnxruntime-genai
   ```

3. Run the model

   Run the model with [phi3v.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py).

   ```bash
   curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3v.py -o phi3v.py
   python phi3v.py -m cpu-int4-rtn-block-32-acc-level-4
   ```

   You enter the path to an image file and a prompt, and the model uses the image and prompt to give you an answer.

   For example: `Where was the photograph taken?`

   ![Market](market.png)

   ```
   The photograph was taken at the Public Market Center, as indicated by the signage in the image.</s>
   ```
