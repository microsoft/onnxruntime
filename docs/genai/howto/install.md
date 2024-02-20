---
title: Install ONNX Runtime GenAI
description: Instructions to install ONNX Runtime GenAI on your target platform in your environment
has_children: false
parent: How to
grand_parent: Generative AI
nav_order: 1
---

# Install ONNX Runtime GenAI

## Python package

(Coming soon) `pip install onnxruntime-genai`

(Temporary)
1. Build from source

   Follow the instructions in [build-from-source.md]

2. Install wheel

   ```bash
   cd build/wheel
   pip install onnxruntime-genai*.whl
   ```

## C# package

(Coming soon) `dotnet add package Microsoft.ML.OnnxRuntime.GenAI`

(Temporary)
1. Build from source

   Follow the instructions in [build-from-source.md]

2. Build nuget package

   ```cmd
   nuget.exe pack Microsoft.ML.OnnxRuntimeGenAI.nuspec -Prop version=0.1.0 -Prop id="Microsoft.ML.OnnxRuntimeGenAI.Gpu"
   ```

3. Install the nuget package

   ```cmd
   dotnet add package .. local instructions
   ```


## C artifacts

(Coming soon) Download release archive

Unzip archive

(Temporary)
1. Build from source

   Follow the instructions in [build-from-source.md]

   
2. Use the following include locations to build your C application

   * 

3. Use the following library locations to build your C application

   * 

   

