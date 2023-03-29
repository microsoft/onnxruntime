---
title: ORT model format runtime optimization
grand_parent: Performance
parent: Model optimizations
nav_order: 5
redirect_from: 
- /docs/performance/ort-format-model-runtime-optimization
- /docs/reference/mobile/ort-format-model-runtime-optimization
---
{::options toc_levels="2" /}

# ORT Format Model Runtime Optimization

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Background

The full ONNX Runtime build supports [graph optimizations](./graph-optimizations.md) at runtime for ONNX models.

The ORT format model was designed to be used with ONNX Runtime [minimal builds](../../build/custom.md#minimal-build) for environments where smaller binary size is important. To reduce the binary size, some or all of the graph optimizer code is excluded from a minimal build. As such, ONNX models and ORT format models do not share the same graph optimization process.

In ONNX Runtime **1.11 and later**, there is limited support for graph optimizations at runtime for ORT format models. This only applies to extended minimal builds or full builds.

In ONNX Runtime **1.10 and earlier**, there is **no support** for graph optimizations at runtime for ORT format models. Any graph optimizations must be done at model conversion time. See [this page](./../mobile-performance-tuning.md) for guidance using older ORT versions. 

As a rule, [basic graph optimizations](./graph-optimizations.md#basic-graph-optimizations) are semantics-preserving and result in a valid ONNX graph. The basic optimizations can and generally should be baked in to the converted ORT format model at conversion time - this is the default behavior of the conversion script. In fact, any runtime optimization support for ORT format models will not include basic optimizations at all.

## Types of runtime optimization

These only apply to extended minimal builds or full builds.

### Saved runtime optimizations

Some graph optimizers support additional modes to save and load information about potential graph optimizations to and from the ORT format model. These potential optimizations are known as saved runtime optimizations.

Saved runtime optimizations are only applied at runtime if they are still applicable. For example, a CPU Execution Provider (EP)-specific optimization for some nodes is only applicable if those nodes are assigned to the CPU EP at runtime.

When converting from ONNX to ORT format, the potential optimizations are identified (1) and their effects are saved alongside the graph (without those optimizations applied) in the ORT format model. Later, when loading the ORT format model with saved runtime optimizations, the effects of potential optimizations are applied (2) if the potential optimizations are still applicable.

In an extended minimal build, only enough implementation to support (2) is included, reducing the binary size.

### Graph optimizers

Some graph optimizers are also fully enabled in an extended minimal build and can be directly applied to an ORT format model. One example is the NHWC transformer.

## Choosing whether to use runtime optimizations

The use of runtime optimizations is optional. Generally, the choice is between using an ORT format model with saved runtime optimizations and using a fully optimized ORT format model.

A fully optimized model will have the full set of ONNX Runtime optimizations (at the extended level or higher) available but will be fully optimized for the configuration at model conversion time.

A model with saved runtime optimizations has fewer optimizations available but has more flexibility at runtime. For example, at runtime, a compiling EP like the NNAPI EP can claim the set of nodes it can handle and the remaining nodes can be further optimized with saved runtime optimizations.

You can compare the performance of:
- A fully optimized model run with only the CPU EP enabled
- A model with saved runtime optimizations run with additional EPs enabled

The [model usability checker](../../tutorials/mobile/helpers/model-usability-checker.md) will provide guidance for a particular model.
