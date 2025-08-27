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
A plugin EP is built as a dynamic/shared library that exports the functions `CreateEpFactories()` and `ReleaseEpFactory()`. ONNX Runtime calls `CreateEpFactories()` to obtain one or more instances of `OrtEpFactory`. An `OrtEpFactory` creates `OrtEp` instances and specifies the hardware devices supported by the EPs it creates. A plugin EP library provides ONNX Runtime with custom implementations of `OrtEpFactory` and `OrtEp`.

The ONNX Runtime repository includes a [sample plugin EP library](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/autoep/library), which is referenced in the following sections.

### Defining an OrtEp
An `OrtEp` represents an instance of an EP that is used by an ONNX Runtime session to determine the model operations supported by the EP and run the supported model operations.

The following table lists the required varibles and functions that an implementor must define for an `OrtEp`.

<table>
<tr>
<th>Field</th>
<th>Signature</th>
<th>Summary</th>
<th>Sample EP reference</th>
</tr>

<tr>
<td>ort_version_supported</td>
<td><pre><code>uint32_t ort_version_supported;</code></pre></td>
<td>The ONNX Runtime version with which the EP was compiled. Implementation should set to <code>ORT_API_VERSION</code>.</td>
<td><a href="https://github.com/microsoft/onnxruntime/blob/16ae99ede405d3d6c59d7cce80c53f5f7055aeed/onnxruntime/test/autoep/library/ep.cc#L160">ExampleEp constructor</a></td>
</tr>

<tr>
<td>GetName</td>
<td><pre><code>const char* GetName(OrtEp* this_ptr);</code></pre></td>
<td>Get the execution provider name. The returned string should be a null-terminated, UTF-8 encoded string. ORT will copy the string.</td>
<td><a href="https://github.com/microsoft/onnxruntime/blob/16ae99ede405d3d6c59d7cce80c53f5f7055aeed/onnxruntime/test/autoep/library/ep.cc#L181">ExampleEp::GetNameImpl()</a></td>
</tr>

<tr>
<td>GetCapability</td>
<td><pre><code>
OrtStatus* GetCapability(OrtEp* this_ptr,
                         const OrtGraph* graph,
                         OrtEpGraphSupportInfo* graph_support_info);
</code></pre></td>
<td>Get information about the nodes/subgraphs supported by the <code>OrtEp</code> instance.</td>
<td><a href="https://github.com/microsoft/onnxruntime/blob/16ae99ede405d3d6c59d7cce80c53f5f7055aeed/onnxruntime/test/autoep/library/ep.cc#L231">ExampleEp::GetCapabilityImpl()</a></td>
</tr>

<tr>
<td>Compile</td>
<td><pre><code>
OrtStatus* Compile(OrtEp* this_ptr,
                   const OrtGraph** graphs,
                   const OrtNode** fused_nodes,
                   size_t count,
                   OrtNodeComputeInfo** node_compute_infos,
                   OrtNode** ep_context_nodes);
</code></pre></td>
<td>
Compile <code>OrtGraph</code> instances assigned to the <code>OrtEp</code>. Implementation must set a <code>OrtNodeComputeInfo</code> instance for each <code>OrtGraph</code> in order to define its computation function.<br/>
If the session is configured to generate a pre-compiled model, the execution provider must return <code>count</code> number of EPContext nodes.
</td>
<td><a href="https://github.com/microsoft/onnxruntime/blob/16ae99ede405d3d6c59d7cce80c53f5f7055aeed/onnxruntime/test/autoep/library/ep.cc#L293">ExampleEp::CompileImpl()</a></td>
</tr>

<tr>
<td>ReleaseNodeComputeInfos</td>
<td><pre><code>
void ReleaseNodeComputeInfos(OrtEp* this_ptr,
                             OrtNodeComputeInfo** node_compute_infos,
                             size_t num_node_compute_infos);
</code></pre></td>
<td>
Release <code>OrtNodeComputeInfo</code> instances.
</td>
<td><a href="https://github.com/microsoft/onnxruntime/blob/16ae99ede405d3d6c59d7cce80c53f5f7055aeed/onnxruntime/test/autoep/library/ep.cc#L364">ExampleEp::ReleaseNodeComputeInfosImpl()</a></td>
</tr>

</table>


### Defining an OrtEpFactory
### Exporting functions to create and release factories

## Using a plugin EP library

## API reference
API header files:
 - [onnxruntime_ep_c_api.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_ep_c_api.h)
   - Defines interfaces implemented by plugin EP and EP factory instances.
   - Provides APIs utilized by plugin EP and EP factory instances.
 - [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h)
   - Provides APIs used to traverse an input model graph.


### Data Types

<!-- Use an HTML table to allow embedding a code block in a cell -->
<table>
<tr>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td>
<a href="#ort-hardware-device-type">OrtHardwareDeviceType</a>
</td>
<td>
Enumerates classes of hardware devices:<br>
<ul>
<li>OrtHardwareDeviceType_CPU</li>
<li>OrtHardwareDeviceType_GPU</li>
<li>OrtHardwareDeviceType_NPU</li>
</ul>
</td>
</tr>

<tr>
<td>
OrtHardwareDevice
</td>
<td>
Opaque type that represents a physical hardware device.
</td>
</tr>

<tr>
<td>
OrtExecutionProviderDevicePolicy
</td>
<td>
Enumerates the default EP selection policies available to users of ORT's automatic EP selection.
</td>
</tr>

<tr>
<td>
OrtEpDevice
</td>
<td>
Opaque type that represents a pairing of an EP and hardware device that can run a model or model subgraph.
</td>
</tr>

<tr>
<td>
OrtNodeFusionOptions
</td>
<td>
Struct that contains options for fusing nodes supported by an EP.
</td>
</tr>

<tr>
<td>
OrtNodeComputeContext
</td>
<td>
Opaque type that contains a compiled/fused node's name and host memory allocation functions. ONNX Runtime provides an instance of <code>OrtNodeComputeContext</code> as an argument to <code>OrtNodeComputeInfo::CreateState()</code>.
</td>
</tr>

<tr>
<td>
OrtNodeComputeInfo
</td>
<td>
Struct that contains the computation function for a compiled `OrtGraph` instance. Initialized by an `OrtEp` instance.
</td>
</tr>

<tr>
<td>
OrtEpGraphSupportInfo
</td>
<td>
Opaque type that contains information on the nodes supported by an EP. An instance of `OrtEpGraphSupportInfo` is passed to `OrtEp::GetCapability()` and the EP populates the `OrtEpGraphSupportInfo` instance with information on the nodes that it supports.
</td>
</tr>

<tr>
<td>
OrtEpDataLayout
</td>
<td>
Enumerates the operator data layouts that could be preferred by an EP. By default, ONNX models use a "channel-first" layout (e.g., NCHW) but some EPs may prefer a "channel-last" layout (e.g., NHWC).
</td>
</tr>

<tr>
<td>
OrtMemoryDevice
</td>
<td>
Opaque type that represents a combination of a physical device and memory type. A memory allocation and allocator are associated with a specific `OrtMemoryDevice`, and this information is used to determine when data transfer is required.
</td>
</tr>

<tr>
<td>
<a href="#ort-data-transfer-impl">OrtDataTransferImpl</a>
</td>
<td>
Struct of functions that an EP implements to copy data between the devices that the EP uses and CPU.
</td>
</tr>

<tr>
<td>
<a href="#ort-sync-notification-impl">OrtSyncNotificationImpl</a>
</td>
<td>
Struct of functions that an EP implements for Stream notifications.
</td>
</tr>

<tr>
<td>
<a href="#ort-sync-stream-impl">OrtSyncStreamImpl</a>
</td>
<td>
Struct of functions that an EP implements if it needs to support Streams.
</td>
</tr>

<tr>
<td>
<a href="#ort-ep-factory">OrtEpFactory</a>
</td>
<td>
A plugin EP library provides ORT with one or more instances of `OrtEpFactory`. An `OrtEpFactory` implements functions that are used by ORT to query device support, create allocators, create data transfer objects, and create instances of an EP (i.e., an `OrtEp` instance).<br/>

An `OrtEpFactory` may support more than one hardware device (`OrtHardwareDevice`). If more than one hardware device is supported by the factory, an EP instance created by the factory is expected to internally partition any graph nodes assigned to the EP among its supported hardware devices.<br/>

Alternatively, if an EP library author needs ONNX Runtime to partition the graph nodes among different hardware devices supported by the EP library, then the EP library must provide multiple `OrtEpFactory` instances. Each `OrtEpFactory` instance must support one hardware device and must create an EP instance with a unique name (e.g., MyEP_CPU, MyEP_GPU, MyEP_NPU).
</td>
</tr>

<tr>
<td>
<a href="#ort-ep">OrtEp</a>
</td>
<td>
An instance of an Ep that can execute model nodes on one or more hardware devices (`OrtHardwareDevice`). An `OrtEp` implements functions that are used by ORT to query graph node support, compile supported nodes, query preferred data layout, set run options, etc. An `OrtEpFactory` creates an `OrtEp` instance via the `OrtEpFactory::CreateEp()` function.
</td>
</tr>

<tr>
<td>
OrtRunOptions
</td>
<td>
Opaque object containing options passed to the `OrtApi::Run()` function, which runs a model.
</td>
</tr>

<tr>
<td>
OrtGraph
</td>
<td>
Opaque type that represents a graph. Provided to `OrtEp` instances in calls to `OrtEp::GetCapability()` and `OrtEp::Compile()`.
</td>
</tr>

<tr>
<td>
OrtValueInfo
</td>
<td>
Opaque type that contains information for a value in a graph. A graph value can be a graph input, graph output, graph initializer, node input, or node output. An `OrtValueInfo` instance has the following information.<br/>
<ul>
<li>Type and shape (e.g., `OrtTypeInfo`)</li>
<li>`OrtNode` consumers</li>
<li>`OrtNode` producer</li>
<li>Information that classifies the value as a graph input, graph output, initializer, etc.</li>
</ul>
</td>
</tr>

<tr>
<td>
OrtExternalInitializerInfo
</td>
<td>
Opaque type that contains information for an initializer stored in an external file. An `OrtExternalInitializerInfo` instance contains the file path, file offset, and byte size for the initializer. Can be obtained from an `OrtValueInfo` via the function `ValueInfo_GetExternalInitializerInfo()`.
</td>
</tr>

<tr>
<td>
OrtTypeInfo
</td>
<td>
Opaque type that contains the element type and shape information for ONNX tensors, sequences, maps, sparse tensors, etc.
</td>
</tr>

<tr>
<td>
OrtTensorTypeAndShapeInfo
</td>
<td>
Opaque type that contains the element type and shape information for an ONNX tensor.
</td>
</tr>

<tr>
<td>
OrtNode
</td>
<td>
Opaque type that represents a node in a graph.
</td>
</tr>

<tr>
<td>
<a href="#ort-op-attr-type">OrtOpAttrType</a>
</td>
<td>
Enumerates attribute types.
</td>
</tr>

<tr>
<td>
OrtOpAttr
</td>
<td>
Opaque type that represents an ONNX operator attribute.
</td>
</tr>

</table>


### Plugin EP Library Registration APIs
The following table lists the API functions used for registration of a plugin EP library.

<table>
<tr>
<th>
Function
</th>
<th>
Description
</th>
</tr>

<tr>
<td>
RegisterExecutionProviderLibrary
</td>
<td>
</td>
</tr>

<tr>
<td>
UnregisterExecutionProviderLibrary
</td>
<td>
</td>
</tr>

<tr>
<td>
GetEpDevices
</td>
<td>
</td>
</tr>

<tr>
<td>
SessionOptionsAppendExecutionProvider_V2
</td>
<td>
</td>
</tr>

<tr>
<td>
SessionOptionsSetEpSelectionPolicy
</td>
<td>
</td>
</tr>

<tr>
<td>
SessionOptionsSetEpSelectionPolicyDelegate
</td>
<td>
</td>
</tr>

</table>

### Plugin EP Library Exported Symbols
The following table lists the functions that have to be exported from the plugin EP library.

<table>
<tr>
<th>
Function
</th>
<th>
Description
</th>
</tr>

<tr>
<td>
CreateEpFactories
</td>
<td>
</td>
</tr>

<tr>
<td>
ReleaseEpFactory
</td>
<td>
</td>
</tr>

</table>

