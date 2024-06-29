---
title: Troubleshoot
description: How to troubleshoot common problems
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 1
---

# Troubleshoot issues with ONNX Runtime generate() API
{: .no_toc }

* TOC placeholder
{:toc}

## Installation issues

```
ImportError: DLL load failed while importing onnxruntime_genai: A dynamic link library (DLL) initialization routine failed.
```

If you see this issue in a Conda environment on Windows, you need to upgrade the `C++ runtime for Visual Studio`. In the conda environment, run the following command:

```bash
conda install conda-forge::vs2015_runtime
```

The onnxruntime-genai Python package should run without error after this extra step.
