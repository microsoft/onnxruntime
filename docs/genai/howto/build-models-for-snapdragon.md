---
title: Build models for Snapdragon
description: How to build model assets for Snapdragon devices
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 4
---

# How to build model assets for Snapdragon NPU devices

These instructions demonstrate generating the Llama 3.2 3B model. You can use the same instructions to generate the Phi-3.5 mini instruct model.

## Setup and pre-requisites

1. Sign up for [Qualcomm AI Hub](https://aihub.qualcomm.com/) access

   Once signed up, configure your Qualcomm AI hub API token

   Follow instructions shown here: [https://app.aihub.qualcomm.com/docs/hub/getting_started.html#getting-started]

2. Install the [Qualcomm AI Engine Direct SDK](https://softwarecenter.qualcomm.com/#/catalog/item/qualcomm_neural_processing_sdk_public)

3. Sign up to get access to HuggingFace weights for [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
   
   This step is only required for models that require signing a license agreement.

4. Setup a Linux environment
   
   There are steps in this process that can only be run on Linux. A WSL environment suffices.

   Install libc++=dev in the Linux environment.

   ```bash
   sudo apt get libc++-dev
   ```

## Generate Qualcomm context binaries

1. Install model from Qualcomm AI hub

   ```bash
   python -m pip install -U qai_hub_models[llama-v3-2-3b-chat-quantized]
   ```

2. Generate QNN context binaries

   Generate the QNN binaries. This step downloads and uploads the model and binaries to and from the Qualcomm AI Hub and depending on your upload speed can take several hours.

   ```bash
   python -m qai_hub_models.models.llama_v3_2_3b_chat_quantized.export --device "Snapdragon X Elite CRD" --skip-inferencing --skip-profiling --output-dir .
   ```

   More information on this step can be found at: [https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie].

## Generate ONNX wrapper models


1. Download the following script from the onnxruntime repo

   ```bash
   curl -LO https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/onnxruntime/python/tools/qnn/gen_qnn_ctx_onnx_model.py
   ```

2. Install the `onnx` package

   ```bash
   pip install onnx
   ```

3. Extract the QNN graph information from QNN context binary file, once for each model (.bin file)

   Note: this script only runs on Linux with libc++-dev installed (from the setup section)

   ```bash
   for bin_file in *.bin; do $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility --context_binary="$bin_file" --json_file="${bin_file%.bin}.json"; done
   ```

4. Generate the ONNX wrapper models

   Run the following command below to generate the ONNX wrapper models

   On Linux with bash:

   ```bash
   for bin_file in *.bin; do python gen_qnn_ctx_onnx_model.py -b "$bin_file" -q "${bin_file%.bin}.json" --quantized_IO --disable_embed_mode; done
   ```

   On Windows with PowerShell:

   ```powershell
   Get-ChildItem -Filter "*.bin" | ForEach-Object {
     $binFile = $_.Name
     $jsonFile = "$($binFile -replace '\.bin$', '.json')"
     python gen_qnn_ctx_onnx_model.py -b $binFile -q $jsonFile --quantized_IO --disable_embed_mode
   }
   ```

## Add other assets

Download assets from [https://huggingface.co/onnx-community/Llama-3.2-3B-instruct-hexagon-npu-assets] 

## Check model assets

Once the above instructions are complete you should have the following model assets

* `genai_config.json`
* `tokenizer.json`
* `tokenizer_config.json`
* `special_tokens_map.json`
* `quantizer.onnx`
* `dequantizer.onnx`
* `position-processor.onnx`
* a set of transformer model binaries
  * Qualcomm context binaries (`*.bin`)
  * Context binary meta data (`*.json`)
  * ONNX wrapper models (`*.onnx`)

