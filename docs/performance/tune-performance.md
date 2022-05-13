---
title: Tune performance
parent: Performance
nav_order: 1
description: ONNX Runtime performance tuning considerations across range of hardware, execution providers, and multi-programming languages. ONNX Runtime performance tuning tools, tips, faqs, troubleshooting checklist, and other design considerations are given.
redirect_from: /docs/how-to/tune-performance
---

# ONNX Runtime Performance Tuning

ONNX Runtime provides high performance across a range of hardware options through its [Execution Providers interface](../execution-providers) for different  environments.

Along with this flexibility comes decisions for tuning and usage. For each model running with different execution providers, there are a few settings that can be tuned (thread number, wait policy, and so on) to improve performance.

This document covers basic tools and troubleshooting checklists that can be leveraged to optimize your ONNX Runtime (ORT) model and hardware.

Refer to a simple demo of [deploying and optimizing a distilled BERT model](https://youtu.be/W_lUGPMW_Eg) to inference on device in the browser.


<iframe height="315" class="table-wrapper py px" src="https://www.youtube.com/embed/W_lUGPMW_Eg?rel=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen=""></iframe>


Here are some additional topics to explore for more information on performance tuning ONNX Runtime.

#### [1. Performance Tuning Tools](./performance-tuning-tools.md)
#### [2. Choosing the Execution Provider for best performance](./choosing-execution-providers.md)
#### [3. Tips for Tuning Performance](./tips-to-tune-performance.md)
#### [4. Troubleshooting Performance Issues](./troubleshooting-performance-issues.md)
#### [5. Mobile Performance Tuning](./mobile-performance-tuning.md)


<p><a href="#" id="back-to-top">Back to top</a></p>

