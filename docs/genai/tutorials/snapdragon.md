---
title: Run on Snapdragon devices 
description: Learn how to run Phi-3.5 and Llama 3.2 ONNX models on Snapdragon devices 
has_children: false
parent: Tutorials
grand_parent: Generate API (Preview)
nav_order: 6
---


# Run models on Snapdragon devices with NPUs

Learn how to run SLMs on Snapragon devices with ONNX Runtime.

## Models
Devices with Snapdragon NPUs requires models in a specific size and format.

Models supported currently are:
* [Phi-3.5 mini instruct](https://github.com/microsoft/ort_npu_samples/releases/tag/v73-phi-3.5-2.31)
* [Llama 3.2 3B](https://github.com/microsoft/ort_npu_samples)

Due to Meta licensing restrictions, the Llama model cannot be pre-published. Instructions to generate the Llama model can be found in the link.


## Python application

If your device has Python installed, you can run a simple question and answering script to query the model.

### Install the runtime

```powershell
pip install onnxruntime-genai
```

### Download the script

```powershell
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/examples/python/model-qa.py -o model-qa.py
```

### Run the script

```powershell
python .\model-qa.py -e cpu -g -v --system_prompt "You are a helpful assistant. Be brief and concise." --chat_template "<|user|>\n{input} <|end|>\n<|assistant|>" -m ..\..\models\microsoft\phi-3.5-mini-instruct-npu-qnn-2.31-v2
```

### A look inside the Python script


## C++ Application

To run the models on snadragon NPU within a C++ application, use the code from here: https://github.com/microsoft/onnxruntime-genai/tree/main/examples/c.

Building and running this application requires a Windows PC with a Snadragon NPU, as well as:
* cmake
* Visual Studio 2022


1. Clone the repo

   ```powershell
   git clone https://github.com/microsoft/onnxruntime-genai
   cd examples\c
   ```

2. Install onnxruntime

   Currently requires the nightly build of onnxruntime, as there are up to the minute changes to QNN support for language models. 
   
   Download the nightly version of the ONNX Runtime QNN binaries from [here](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/NuGet/Microsoft.ML.OnnxRuntime.QNN/overview/1.22.0-dev-20250225-0548-e46c0d8)


   ```powershell
   mkdir onnxruntime-win-arm64-qnn
   move Microsoft.ML.OnnxRuntime.QNN.1.22.0-dev-20250225-0548-e46c0d8.nupkg onnxruntime-win-arm64-qnn
   cd onnxruntime-win-arm64-qnn
   tar xvzf Microsoft.ML.OnnxRuntime.QNN.1.22.0-dev-20250225-0548-e46c0d8.nupkg
   copy runtimes\win-arm64\native\* ..\..\..\lib
   cd ..
   ```


3. Install onnxruntime-genai

   ```powershell
   curl https://github.com/microsoft/onnxruntime-genai/releases/download/v0.6.0/onnxruntime-genai-0.6.0-win-arm64.zip -o onnxruntime-genai-win-arm64.zip
   tar xvf onnxruntime-genai-win-arm64.zip
   cd onnxruntime-genai-0.6.0-win-arm64
   copy include\* ..\include
   copy lib\* ..\lib
   ```

4. Build the sample

   ```powershell
   cmake -A arm64 -S . -B build -DPHI3-QA=ON
   cd build
   cmake --build . --config Release
   ```

5. Run the sample

   ```
   cd Release
   .\phi3_qa.exe <path_to_model>
   ```











