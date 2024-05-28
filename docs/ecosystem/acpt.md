---
title: Azure Container for PyTorch (ACPT)
description: Learn more about Azure Container for PyTorch (ACPT) and how it utilizes ONNX Runtime
nav_order: 9
redirect_from: /docs/tutorials/ecosystem/acpt
---
# Azure Container for PyTorch (ACPT)
{: .no_toc }

Azure Container for PyTorch (ACPT) is a lightweight, standalone environment that includes needed components to effectively run optimized training for large models. It helps with reducing preparation costs and faster deployment time. ACPT can be used to quickly get started with various deep learning tasks with PyTorch on Azure.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Why should I use ACPT?
* **Flexibility:** Use as-is with preinstalled packages or build on top of the curated environment.
* **Ease of use:** All components are installed and validated against dozens of Microsoft workloads to reduce setup costs and accelerate time to value.
* **Efficiency:** Avoid unnecessary image builds and only have required dependencies that are accessible right in the image/container.
* **Optimized training framework:** Set up, develop, and accelerate PyTorch models on large workloads, and improve training and deployment success rate.
* **Up-to-date stack:** Access the latest compatible versions of Ubuntu, Python, PyTorch, CUDA/RocM, etc.
* **Latest training optimization technologies:** Make use of ONNX Runtime, DeepSpeed, MSCCL, and more.

## Supported configurations for Azure Container for PyTorch (ACPT)
The following configurations are supported in the Microsoft Container Registry (MCR): [ptca_image_list.md](https://aiinfra.visualstudio.com/PyTorch/_git/ptebic?path=/.azure_pipelines/pipeline_config/ptca_image_list.md&_a=preview).

Most recent image releases (last updated 4/25/2024):

| Environment Name            | OS             | GPU Version | Python Version | PyTorch Version | ORT-training Version | DeepSpeed Version | torch-ort Version | Nebula Version |
|-----------------------------|----------------|-------------|----------------|-----------------|---------------------|-------------------|-------------------|----------------|
| acpt-pytorch-2.2-cuda12.1   | Ubuntu 20.04   | cu121       | 3.10           | 2.2.2           | 1.17.3                 | 0.13.1            | 1.17.3            | 0.16.11        |
| acpt-pytorch-2.1-cuda12.1   | Ubuntu 20.04   | cu121       | 3.10           | 2.1.2           | 1.17.3                 | 0.13.1            | 1.17.3            | 0.16.11        |
| acpt-pytorch-2.0-cuda11.7   | Ubuntu 20.04   | cu117       | 3.10           | 2.0.1           | 1.17.3                 | 0.13.1            | 1.17.3            | 0.16.11        |
| acpt-pytorch-1.13-cuda11.7  | Ubuntu 20.04   | cu117       | 3.10           | 1.13.1          | 1.17.3                 | 0.13.1            | 1.17.3            | 0.16.11        |

Other packages like fairscale, horovod, msccl, protobuf, pyspark, pytest, pytorch-lightning, tensorboard, NebulaML, torchvision, and torchmetrics are provided to support all training needs.

## Support
Version updates for supported environments, including the base images they reference, are released every two weeks to address vulnerabilities no older than 30 days. Based on usage, some environments may be deprecated (hidden from the product but usable) to support more common machine learning scenarios.
