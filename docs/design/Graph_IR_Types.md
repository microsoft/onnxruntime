# Public Graph IR Types: Model Editor API and EP API

`OrtGraph`, `OrtNode`, and `OrtValueInfo` are opaque public C types used by two unrelated APIs:

- The [Model Editor API](Model_Editor_API.md) (`OrtModelEditorApi`). An application creates these objects to *build*
  a model.
- The plugin EP API (`OrtEp` callbacks plus the graph accessors in `OrtApi`). ORT creates these objects so an
  execution provider can *inspect* a model, for example in `OrtEp::GetCapability` and `OrtEp::Compile`.

The C type is the same in both cases, but the object behind the pointer is a different implementation. This document
explains that split, which operations work on which kind of object, and why many getters return `ORT_NOT_IMPLEMENTED`
for objects created by the Model Editor API.

## Two implementations of the same types

| Public type | EP API implementation | Model Editor API implementation |
|---|---|---|
| `OrtGraph` | `onnxruntime::EpGraph` | `onnxruntime::ModelEditorGraph` |
| `OrtNode` | `onnxruntime::EpNode` | `onnxruntime::ModelEditorNode` |
| `OrtValueInfo` | `onnxruntime::EpValueInfo` | `onnxruntime::ModelEditorValueInfo` |
| Source | `onnxruntime/core/graph/ep_api_types.h` | `onnxruntime/core/graph/model_editor_api_types.h` |

Other related types have only one implementation and behave the same in both APIs:

- `OrtOpAttr` is an ONNX `AttributeProto`. See [OrtOpAttr](#ortopattr-one-representation-for-both-apis).
- `OrtTypeInfo` describes a value's type and shape.
- `OrtModel` exists only in the Model Editor API.

### OrtOpAttr: one representation for both APIs

Both APIs use `OrtOpAttr` heavily:

- **EP API.** EPs get attributes from `Node_GetAttributes` and `Node_GetAttributeByName`, and from
  `ShapeInferContext_GetAttribute`. They read them with `OpAttr_GetName`, `OpAttr_GetType`, `ReadOpAttr`, and
  `OpAttr_GetTensorAttributeAsOrtValue`.
- **Model Editor API.** Applications create attributes with `CreateOpAttr` and pass them to `CreateNode`. So do EPs,
  when they build EPContext nodes.

`OrtOpAttr` is not polymorphic, because an attribute is self-contained data. Its name, type, and value don't depend on
a resolved graph, so an attribute built by the caller and one read from a model node can answer the same questions.
There is nothing variant-specific to dispatch on.

The internal representation is simply `ONNX_NAMESPACE::AttributeProto`. `abi_graph_types.h` declares
`struct OrtOpAttr { AttributeProto attr_proto; }`, but ORT never constructs an `OrtOpAttr`. Instead, every producer
casts an `AttributeProto*` to `OrtOpAttr*`, and every consumer casts it back:

| Producer | Storage owner and lifetime |
|---|---|
| `CreateOpAttr` (`standalone_op_invoker.cc`) | A new heap `AttributeProto`. The caller owns it until `CreateNode` takes it, or releases it with `ReleaseOpAttr`, which deletes it as an `AttributeProto`. |
| `EpNode::Create` (`ep_api_types.cc`) | A copy of each internal `Node` attribute, owned by the `EpNode`. Valid as long as the node is. |
| `ShapeInferContext_GetAttribute` (`custom_ops.cc`) | Points into the node being inferred. Valid only during the shape inference call. |

Consumers either cast back to `AttributeProto` (`ReadOpAttr`, `CreateNode`) or use the `attr_proto` member
(`OpAttr_GetName`, `OpAttr_GetType`). Both work because the member is at offset zero and `OrtOpAttr` adds no other
state. Keep `OrtOpAttr` exactly that shape: adding a field, a base class, or a virtual function would break every
cast.

Graph-valued attributes are a special case. `OpAttr_GetType` can report `ORT_OP_ATTR_GRAPH`, but EPs read nested
graphs through `Node_GetSubgraphs`, which returns `EpGraph` objects. `CreateOpAttr` can't create graph or tensor
attributes, which is why the Model Editor API can't build control-flow nodes.

### Why the implementations differ

The two APIs work on different things.

**EP API objects are read-only views of a resolved graph.** `EpGraph` wraps an internal `GraphViewer`, and each
`EpNode` wraps an internal `Node`. The graph has been resolved, so ORT can answer structural questions: a value's
producer and consumers, a node's operator `since_version`, the model's opsets, initializer data, and nested subgraphs.
ORT owns these objects. They are valid only for the duration of the callback that received them. The one exception is
the graph returned by `Graph_GetGraphView`, which the caller owns and releases with `ReleaseGraph`.

**Model Editor objects are write-only builders.** They hold only what the caller supplied: names, `OrtTypeInfo`
instances, attributes, and initializer `OrtValue`s. Nodes refer to their inputs and outputs *by name*. No internal
`Graph` exists until the `OrtModel` is loaded into a session, which resolves names into edges and runs shape
inference. Before that, "who produces this value?" or "what is this node's `since_version`?" has no answer. The caller
owns these objects until ownership moves into a graph, model, or session.

Because a Model Editor object has no resolved graph behind it, getters that need one return `ORT_NOT_IMPLEMENTED`
instead of returning an incomplete or misleading answer.

## How the variant is identified

The base classes in `onnxruntime/core/graph/abi_graph_types.h` are abstract. Each declares the full set of accessors as
pure virtual functions and stores a tag:

```cpp
enum class OrtGraphIrApi {
  kInvalid = 0,
  kModelEditorApi,
  kEpApi,
};

struct OrtNode {
  explicit OrtNode(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtNode() = default;
  virtual const std::string& GetName() const = 0;
  virtual onnxruntime::Status GetSinceVersion(int& since_version) const = 0;
  // ...
  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};
```

Each derived type passes its tag to the base constructor. It also uses `DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS`
to generate:

- `ToExternal()`, which upcasts to the public type.
- `ToInternal(external)`, a checked downcast that returns `nullptr` if the tag doesn't match.

The check compares tags rather than using `dynamic_cast`, so it still works in builds with `onnxruntime_DISABLE_RTTI`.

This gives the implementation three ways to handle a call:

1. **Virtual dispatch through the base type.** Most `OrtApi` accessors call the virtual function directly, for
   example `Graph_GetNumNodes` calls `graph->GetNumNodes()`. Each variant supplies its own answer. `EpGraph` queries
   the `GraphViewer`, and `ModelEditorGraph` returns the size of its node list or `NOT_IMPLEMENTED`.
2. **An explicit variant check in the C API function.** Operations that only make sense for one variant call
   `ToInternal` and return `ORT_INVALID_ARGUMENT` on a mismatch. For example, `Graph_GetGraphView`,
   `Node_GetAttributeByName`, `Node_GetEpName`, and `EpGraphSupportInfo_AddNodesToFuse` require EP API objects.
   Every mutating `OrtModelEditorApi` function requires Model Editor objects.
3. **Release.** `ReleaseGraph`, `ReleaseNode`, and `ReleaseValueInfo` call `delete` on the base pointer. The virtual
   destructor runs the correct derived destructor.

## Accessor support by variant

"Yes" means the accessor returns a real answer. "Not impl." means it returns `ORT_NOT_IMPLEMENTED` through the virtual
function. "Invalid arg." means the C API function checks the variant and returns `ORT_INVALID_ARGUMENT`.

### `OrtValueInfo`

| `OrtApi` function | EP API | Model Editor |
|---|---|---|
| `GetValueInfoName`, `GetValueInfoTypeInfo` | Yes | Yes |
| `ValueInfo_GetValueProducer`, `ValueInfo_GetValueNumConsumers`, `ValueInfo_GetValueConsumers` | Yes¹ | Not impl. |
| `ValueInfo_GetInitializerValue`, `ValueInfo_GetExternalInitializerInfo` | Yes¹ | Not impl. |
| `ValueInfo_IsRequiredGraphInput`, `ValueInfo_IsOptionalGraphInput`, `ValueInfo_IsGraphOutput` | Yes | Not impl. |
| `ValueInfo_IsConstantInitializer`, `ValueInfo_IsFromOuterScope` | Yes | Not impl. |

¹ These need the owning `EpGraph`. `EpValueInfo` instances that have no owning graph return an error. This applies to
the inputs and outputs of the fused nodes that ORT passes to `OrtEp::Compile`.

### `OrtNode`

| `OrtApi` function | EP API | Model Editor |
|---|---|---|
| `Node_GetId` | Yes | Yes² |
| `Node_GetName`, `Node_GetOperatorType`, `Node_GetDomain` | Yes | Yes |
| `Node_GetNumInputs`, `Node_GetNumOutputs`, `Node_GetNumAttributes` | Yes | Yes |
| `Node_GetSinceVersion` | Yes | Not impl. |
| `Node_GetInputs`, `Node_GetOutputs` | Yes | Not impl. |
| `Node_GetNumImplicitInputs`, `Node_GetImplicitInputs` | Yes | Not impl. |
| `Node_GetAttributes` | Yes | Not impl. |
| `Node_GetNumSubgraphs`, `Node_GetSubgraphs`, `Node_GetGraph` | Yes | Not impl. |
| `Node_GetAttributeByName`, `Node_GetEpName` | Yes | Invalid arg. |

² A Model Editor node's ID is its position in the graph's node list. The ID is assigned by `AddNodeToGraph` and is
`0` before that call.

### `OrtGraph`

| `OrtApi` function | EP API | Model Editor |
|---|---|---|
| `Graph_GetName` | Yes | Yes³ |
| `Graph_GetModelMetadata`, `Graph_GetModelPath`, `Graph_GetOnnxIRVersion` | Yes | Yes³ |
| `Graph_GetNumInputs`, `Graph_GetNumOutputs`, `Graph_GetNumInitializers`, `Graph_GetNumNodes` | Yes | Yes |
| `Graph_GetNumOperatorSets`, `Graph_GetOperatorSets` | Yes | Not impl. |
| `Graph_GetInputs`, `Graph_GetOutputs`, `Graph_GetInitializers`, `Graph_GetNodes` | Yes | Not impl. |
| `Graph_GetParentNode` | Yes | Not impl. |
| `Graph_GetGraphView` | Yes | Invalid arg. |

³ These return placeholder values: the name `ModelEditorGraph`, empty metadata, an empty model path, and ORT's
current ONNX IR version. The operator sets live on the `OrtModel`, not the `OrtGraph`, which is why
`Graph_GetOperatorSets` is not implemented.

In minimal builds that are not extended minimal builds, some EP API accessors (such as producer/consumer and
`since_version`) also return `ORT_NOT_IMPLEMENTED`. Model Editor objects exist only in full builds.

## Where the two APIs meet

An EP author uses both APIs:

- **Reading.** `OrtEp::GetCapability` and `OrtEp::Compile` receive EP API objects. Use the `OrtApi` graph accessors
  to inspect them. Don't release them, and don't keep them after the callback returns. Nodes passed back to ORT, for
  example with `EpGraphSupportInfo_AddNodesToFuse`, must be the EP API objects that ORT provided.
- **Writing EPContext nodes.** When `OrtEp::Compile` generates an EPContext model, the EP creates each EPContext node
  with `OrtModelEditorApi::CreateNode` and attributes with `OrtApi::CreateOpAttr`. ORT takes ownership of the returned
  nodes. `ConvertEpContextNodes` in `onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc` uses
  `ModelEditorNode::ToInternal` to read their names and attributes, and rejects nodes of any other variant.
- **Building test or input models.** An EP test can build an `OrtModel` with the Model Editor API and pass it to
  `CreateSessionFromModel` or to the Compile API with `ModelCompilationOptions_SetInputModel`.

You can't convert one variant into the other. To derive a new model from an EP-provided graph, read it with the
`OrtApi` accessors and create new Model Editor objects from the names, types, and attributes you read.

### C++ wrappers

The C++ API mirrors the split:

- `Ort::ConstGraph`, `Ort::ConstNode`, and `Ort::ConstValueInfo` are non-owning views, typically of EP API objects.
- `Ort::Graph`, `Ort::Node`, and `Ort::ValueInfo` are owning wrappers, typically of Model Editor objects.

The owning wrappers inherit the const getters, so a call such as `Ort::Node::GetInputs()` compiles. At run time it
throws an `Ort::Exception` carrying the `ORT_NOT_IMPLEMENTED` status for a Model Editor node.

## Guidance for ORT developers

- **Adding an accessor to a public graph type.**
  1. Add a pure virtual function to the base class in `abi_graph_types.h`.
  2. Implement it in `ep_api_types.h`/`.cc`.
  3. Implement it in `model_editor_api_types.h`. Return `NOT_IMPLEMENTED` with a message naming the operation if the
     builder state can't answer it.
  4. Implement the `OrtApi` function in `onnxruntime_c_api.cc` by calling the virtual function.
  5. Update the tables in this document.
- **Operations that are meaningful for only one variant.** Check the variant in the C API function with `ToInternal`
  and return `ORT_INVALID_ARGUMENT` with a clear message. Don't add a virtual function that every other variant must
  stub out. Always check the `ToInternal` result before using it.
- **Accepting public graph objects as input.** Any API that takes an `OrtGraph*`, `OrtNode*`, or `OrtValueInfo*`
  must validate the variant, because callers can pass either kind.
- **A new variant.** Adding a third implementation requires a new `OrtGraphIrApi` value. Also check every existing
  `ToInternal` call site, because each one assumes the only other variant is the one it rejects.
