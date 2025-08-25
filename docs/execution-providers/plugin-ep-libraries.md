---
title: Plugin EP libraries
description: Plugin EP libraries
parent: Execution Providers
nav_order: 17
redirect_from: /docs/reference/execution-providers/Plugin-EP-Libraries
---

# Plugin Execution Provider Libraries
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Background
An ONNX Runtime Execution Provider (EP) executes model operations on one or more hardware accelerators (e.g., GPU, NPU, etc.). ONNX Runtime provides a variety of built-in EPs, such as the default CPU EP. To enable further extensibility, ONNX Runtime supports user-defined plugin EP libraries that an application can register with ONNX Runtime for use in an ONNX Runtime inference session.<br/>

This page provides a reference for the APIs necessary to develop and use plugin EP libraries with ONNX Runtime.

## Creating a plugin EP library

## Using a plugin EP library

## API reference
API header files:
 - [onnxruntime_ep_c_api.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_ep_c_api.h)
   - Defines interfaces implemented by plugin EP and EP factory instances.
   - Provides APIs utilized by plugin EP and EP factory instances.
 - [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h)
   - Provides APIs used to traverse an input model graph.

### Data Types
**OrtHardwareDeviceType**

Enumerates classes of hardware devices.

**OrtHardwareDevice**

Opaque type that represents a specific hardware device (e.g., CPU, GPU, NPU) on the machine.

**OrtExecutionProviderDevicePolicy**

Enumerates the default EP selection policies available to users of ORT's automatic EP selection.

**OrtEpDevice**

Opaque type that represents a pairing of an EP and hardware device that can run a model or model subgraph.

**OrtNodeFusionOptions**

Struct that contains options for fusing nodes supported by an EP.

**OrtNodeComputeContext**

Opaque type that contains a compiled/fused node's name and host memory allocation functions. ONNX Runtime provides an instance of `OrtNodeComputeContext` as an argument to `OrtNodeComputeInfo::CreateState()`.

**OrtNodeComputeInfo**

Struct that contains the computation function for a compiled `OrtGraph` instance. Initialized by an `OrtEp` instance.

**OrtEpGraphSupportInfo**

Opaque type that contains information on the nodes supported by an EP. An instance of `OrtEpGraphSupportInfo` is passed to `OrtEp::GetCapability()` and the EP populates the `OrtEpGraphSupportInfo` instance with information on the nodes that it supports.

**OrtEpDataLayout**

Enumerates the operator data layouts that could be preferred by an EP. By default, ONNX models use a "channel-first" layout (e.g., NCHW) but some EPs may prefer a "channel-last" layout (e.g., NHWC).

**OrtMemoryDevice**

**OrtDataTransferImpl**

**OrtSyncNotificationImpl**

**OrtSyncStreamImpl**

**OrtEpFactory**

A plugin EP library provides ORT with one or more instances of `OrtEpFactory`. An `OrtEpFactory` implements functions that are used by ORT to query device support, create allocators, create data transfer objects, and create instances of an EP (i.e., an `OrtEp` instance).<br/>

An `OrtEpFactory` may support more than one hardware device (`OrtHardwareDevice`). If more than one hardware device is supported by the factory, an EP instance created by the factory is expected to internally partition any graph nodes assigned to the EP among its supported hardware devices.<br/>

Alternatively, if an EP library author needs ONNX Runtime to partition the graph nodes among different hardware devices supported by the EP library, then the EP library must provide multiple `OrtEpFactory` instances. Each `OrtEpFactory` instance must support one hardware device and must create an EP instance with a unique name (e.g., MyEP_CPU, MyEP_GPU, MyEP_NPU).

**OrtEp**

An instance of an Ep that can execute model nodes on one or more hardware devices (`OrtHardwareDevice`). An `OrtEp` implements functions that are used by ORT to query graph node support, compile supported nodes, query preferred data layout, set run options, etc. An `OrtEpFactory` creates an `OrtEp` instance via the `OrtEpFactory::CreateEp()` function.

**OrtRunOptions**

**OrtGraph**

**OrtValueInfo**

**OrtExternalInitializerInfo**

**OrtTypeInfo**

**OrtTensorTypeAndShapeInfo**

**OrtNode**

**OrtOpAttr**

### Plugin EP Library Registration APIs
**RegisterExecutionProviderLibrary**

**UnregisterExecutionProviderLibrary**

**GetEpDevices**

**SessionOptionsAppendExecutionProvider_V2**

**SessionOptionsSetEpSelectionPolicy**

**SessionOptionsSetEpSelectionPolicyDelegate**


### Plugin EP Library Exported Symbols
**CreateEpFactories**

**ReleaseEpFactory**


