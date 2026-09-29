# Model Editor API

The Model Editor API constructs ONNX models directly with the ONNX Runtime C or C++ API. It has two usage
scenarios:

1. Create a complete model in memory and create an inference session from it.
2. Load an existing model and augment it with nodes and initializers before session initialization.

The API was introduced in ONNX Runtime 1.22. In C, obtain the function table with
`OrtApi::GetModelEditorApi()`; it returns `nullptr` in a minimal build. In C++, `Ort::GetModelEditorApi()` returns a
reference and throws `Ort::Exception` when the API is unavailable. The API requires a full build.

The API cannot remove or replace existing nodes or initializers, and it does not support control-flow nodes
(`If`, `Loop`, `Scan`) because graph-valued attributes cannot be created.

## Core objects and workflow

The API builds a small hierarchy of objects:

```text
OrtTypeInfo -> OrtValueInfo --\
OrtOpAttr  -> OrtNode --------> OrtGraph -> OrtModel -> OrtSession
OrtValue   (initializer) -----/
```

For a new model, the usual sequence is:

1. Create type information and `OrtValueInfo` instances for graph inputs and outputs.
2. Create nodes and any attributes they need.
3. Create initializer tensors.
4. Add the inputs, outputs, nodes, and initializers to an `OrtGraph`.
5. Create an `OrtModel` with every required domain/opset pair and add the graph to it.
6. Create an `OrtSession` from the model. Session creation resolves and validates the graph, runs graph
   optimizations, and initializes the execution providers.

Names connect the graph. A node input name must match a graph input, initializer, or another node's output. Node order
does not define connectivity. Types are declared only for graph inputs and outputs; intermediate value types come from
ONNX shape inference when the graph is resolved.

## Example: create a complete model

The following C++ example builds `Z = Gemm(X, Y)` with a model input `X`, an initializer `Y`, and a model output `Z`.
The C++ API is a resource-safe wrapper over the same `OrtModelEditorApi` C functions.

```cpp
#include <numeric>
#include <vector>

#include <onnxruntime_cxx_api.h>

Ort::Session BuildSession(Ort::Env& env) {
  Ort::Graph graph;

  // X: float[3, 4]
  Ort::TensorTypeAndShapeInfo input_tensor_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 4});
  auto input_type_info = Ort::TypeInfo::CreateTensorInfo(input_tensor_info.GetConst());
  std::vector<Ort::ValueInfo> graph_inputs;
  graph_inputs.emplace_back("X", input_type_info.GetConst());

  // Z: float[3, 8]
  Ort::TensorTypeAndShapeInfo output_tensor_info(
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{3, 8});
  auto output_type_info = Ort::TypeInfo::CreateTensorInfo(output_tensor_info.GetConst());
  std::vector<Ort::ValueInfo> graph_outputs;
  graph_outputs.emplace_back("Z", output_type_info.GetConst());

  graph.SetInputs(graph_inputs);
  graph.SetOutputs(graph_outputs);

  float alpha = 2.0f;
  std::vector<Ort::OpAttr> attributes;
  attributes.emplace_back("alpha", &alpha, 1, ORT_OP_ATTR_FLOAT);
  Ort::Node gemm("Gemm", "", "Gemm1", {"X", "Y"}, {"Z"}, attributes);
  graph.AddNode(gemm);

  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<int64_t> weight_shape{4, 8};
  auto initializer = Ort::Value::CreateTensor<float>(allocator, weight_shape.data(), weight_shape.size());
  float* weight_data = initializer.GetTensorMutableData<float>();
  std::iota(weight_data, weight_data + 4 * 8, 1.0f);
  graph.AddInitializer("Y", initializer, false);

  std::vector<Ort::Model::DomainOpsetPair> opsets{{"", 18}};
  Ort::Model model(opsets);
  model.AddGraph(graph);

  Ort::SessionOptions options;
  Ort::Session session(env, model, options);

  // The session no longer needs the Ort::Model.
  return session;
}
```

For large initializers, an application can avoid the copy by using externally owned storage and passing `true` for
`data_is_external`. That storage must outlive every session created from the model.
`CreateTensorWithDataAndDeleterAsOrtValue` lets the C API transfer responsibility for releasing externally allocated
memory to ORT.

The direct C API follows the same sequence:

| C++ wrapper | C API |
|---|---|
| `Ort::TypeInfo::CreateTensorInfo` | `CreateTensorTypeInfo` |
| `Ort::Graph()` | `CreateGraph` |
| `Ort::ValueInfo(...)` | `CreateValueInfo` |
| `Ort::Node(...)` | `CreateNode` |
| `Graph::SetInputs/SetOutputs` | `SetGraphInputs/SetGraphOutputs` |
| `Graph::AddInitializer` | `AddInitializerToGraph` |
| `Graph::AddNode` | `AddNodeToGraph` |
| `Ort::Model(opsets)` | `CreateModel` |
| `Model::AddGraph` | `AddGraphToModel` |
| `Ort::Session(env, model, options)` | `CreateSessionFromModel` |
| Destructors | `OrtApi::ReleaseValueInfo`, `ReleaseNode`, `ReleaseGraph`, `ReleaseModel`, `ReleaseOpAttr` |

Attributes are created with `OrtApi::CreateOpAttr` (`Ort::OpAttr`). `CreateSparseTensorTypeInfo`,
`CreateMapTypeInfo`, `CreateSequenceTypeInfo`, and `CreateOptionalTypeInfo` exist in the table, but their results
cannot currently be used for graph inputs or outputs.

See `ModelEditorAPITest.Basic_CApi` and `ModelEditorAPITest.Basic_CxxApi` in
`onnxruntime/test/shared_lib/test_model_builder_api.cc` for complete, executable versions.

## Example: augment an existing model

An editor session loads the existing model internally but deliberately postpones session initialization. Create an
`OrtModel` that describes the augmentation, based on the existing model's opsets and graph inputs or outputs as
needed. It contains only the new nodes and initializers plus any replacement graph inputs or outputs. Apply it to the
editor session and then finalize the session.

This example prepends a `Cast` so an existing float input can be supplied as `int64`:

```cpp
Ort::SessionOptions options;
Ort::Session session = Ort::Session::CreateModelEditorSession(env, ORT_TSTR("model.onnx"), options);

auto inputs = session.GetInputs();
const std::string old_input_name = inputs.at(0).GetName();
const auto input_shape = inputs.at(0).TypeInfo().GetTensorTypeAndShapeInfo().GetShape();

int64_t cast_to = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
std::vector<Ort::OpAttr> attributes;
attributes.emplace_back("to", &cast_to, 1, ORT_OP_ATTR_INT);
Ort::Node cast("Cast", "", "CastInput", {"NewInt64Input"}, {old_input_name}, attributes);

Ort::TensorTypeAndShapeInfo tensor_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, input_shape);
auto type_info = Ort::TypeInfo::CreateTensorInfo(tensor_info.GetConst());
inputs.at(0) = Ort::ValueInfo("NewInt64Input", type_info.GetConst());

Ort::Graph additions;
additions.AddNode(cast);
additions.SetInputs(inputs);  // Inputs for the complete augmented graph.

std::vector<Ort::Model::DomainOpsetPair> additional_opsets;  // Cast uses the existing ONNX opset.
Ort::Model update(additional_opsets);
update.AddGraph(additions);
session.FinalizeModelEditorSession(update, options);
```

New nodes in an existing domain must be valid at the model's existing opset for that domain. Query it with
`session.GetOpset(domain)` (`SessionGetOpsetForDomain`) before choosing operators or attributes.

To append a node, use an existing graph output name as the new node's input and give the node a new output name. Call
`SetOutputs` with `OrtValueInfo` entries for the new overall graph outputs.

Graph inputs and outputs in the update are all-or-nothing:

- If the update graph never calls `SetInputs` (or `SetOutputs`), the existing graph inputs (or outputs) are kept.
- If it does, the supplied list replaces the complete set. Include every input (or output) of the augmented graph,
  not only the ones that changed. `session.GetInputs()` and `session.GetOutputs()` provide the current lists.

The equivalent C workflow is:

1. `CreateModelEditorSession` or `CreateModelEditorSessionFromArray`.
2. `SessionGetOpsetForDomain` for every existing domain used by new nodes.
3. Build an `OrtModel` containing the additions.
4. `ApplyModelToModelEditorSession`.
5. `FinalizeModelEditorSession`.

The C++ `FinalizeModelEditorSession(model, options)` wrapper performs steps 4 and 5. It also accepts an optional
`OrtPrepackedWeightsContainer`. After finalization, use the session normally. The session must not be used for
inference before it is finalized.

`ApplyModelToModelEditorSession` is not transactional. If it fails, some domains, initializers, or nodes may already
have been merged into the session's graph, so release the session instead of retrying on it.

See `ModelEditorAPITest.BasicModelEdit_CxxApi` in `onnxruntime/test/shared_lib/test_model_builder_api.cc` for the full
executable example.

## Compile an OrtModel

An `OrtModel` can be the input to the Compile API instead of a file or buffer. This lets an application or EP test
build a model and produce an EPContext model without serializing an ONNX protobuf first:

```cpp
Ort::ModelCompilationOptions compile_options(env, session_options);
compile_options.SetInputModel(static_cast<const OrtModel*>(model));
compile_options.SetOutputModelPath(ORT_TSTR("model_ctx.onnx"));
Ort::Status status = Ort::CompileModel(env, compile_options);
if (!status.IsOK()) {
  // status.GetErrorMessage() describes the failure.
}
```

The C function is `OrtCompileApi::ModelCompilationOptions_SetInputModel` (since 1.24). The model is borrowed: it must
remain valid until `CompileModel` returns and can be reused afterwards. Because an `OrtModel` has no model path, an
explicit output location is required when generating an EPContext model. See the `ModelEditorCompileAPITest` tests in
`onnxruntime/test/shared_lib/test_model_builder_api.cc`.

## Relationship to the EP graph API

`OrtGraph`, `OrtNode`, and `OrtValueInfo` are the same opaque C types that plugin EPs receive through the graph
accessors in `OrtApi` (for example `Graph_GetNodes` and `Node_GetInputs`). Internally they are different variants:

- Objects created by the Model Editor API are write-only builders. Most `OrtApi` graph accessors, such as
  `Graph_GetInputs`, `Node_GetInputs`, and `ValueInfo_GetValueProducer`, return `ORT_NOT_IMPLEMENTED` for them. Simple
  queries such as names, operator type, domain, and counts work.
- Graphs and nodes that ORT gives to an EP cannot be passed to Model Editor functions. Those calls fail with
  `Invalid OrtGraph variant for use in the OrtModelEditorApi` (or the `OrtNode`/`OrtValueInfo` equivalent). To build a
  new model from an EP-provided graph, read it with the `OrtApi` graph accessors and create new Model Editor objects.

See [Public Graph IR Types](Graph_IR_Types.md) for the implementation details and a per-accessor support table.

## Important constraints

- Opsets:
  - The ONNX domain opset must always be explicit in `CreateModel`. The ONNX domain is the empty string; `ai.onnx` is
    accepted as an alias.
  - Other domains known to ORT's schema registry, such as `com.microsoft`, are added automatically at the registry's
    latest version if they are not declared. Declare them explicitly to pin a version.
  - Registered custom-op domains are added automatically at the registry's latest version; declare them explicitly to pin a version.
  - When augmenting, an update model may add new domains or repeat an existing domain at the same version. Changing an
    existing domain's version is rejected.
- Graph inputs and outputs created with `CreateValueInfo` must currently be dense tensors. Sparse tensor, map,
  sequence, and optional types are rejected. A dimension of `-1` is dynamic; give it a name through
  `SetSymbolicDimensions` or the C++ `symbolic_dims` constructor argument, or leave it unnamed for an unknown
  dimension.
- `CreateOpAttr` supports only `INT`, `INTS`, `FLOAT`, `FLOATS`, `STRING`, and `STRINGS` attributes. `TENSOR` and
  `GRAPH` attributes are unavailable, so control-flow nodes cannot be built.
- An ONNX `Constant` node is converted to an initializer. It must have no inputs, one output, and exactly one attribute.
  Because tensor attributes are unavailable, use `value_float`, `value_floats`, `value_int`, `value_ints`,
  `value_string`, or `value_strings`, or add an initializer instead.
- Initializer tensors must be allocated and CPU based.
- With `data_is_external == false`, initializer data is copied into the model. With `data_is_external == true`, the
  buffer is not copied and must remain valid for the lifetime of every session created from the model. External
  initializer data must be at least 128 bytes; small values may be needed by ONNX shape inference and must be copied
  instead.
- `SetGraphInputs`, `SetGraphOutputs`, `CreateNode`, `AddInitializerToGraph`, `AddNodeToGraph`, and `AddGraphToModel`
  transfer ownership on success. `CreateNode` takes ownership of its `OrtOpAttr` inputs. The C API nulls array entries
  where applicable. On failure, ownership remains with
  the caller. Do not release or reuse an object after ownership has transferred. The C++ wrappers encode this by
  resetting the source object after a successful call.
- Each `OrtModel` accepts one graph. Duplicate pointers, duplicate initializer names, null attribute entries, and a
  second graph are rejected.
- To save the result, set `SetOptimizedModelFilePath` on the session options before session creation/finalization. See
  [Graph optimizations: Offline mode](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#offline-mode)
  for details.

## Common errors

| Error text | Cause |
|---|---|
| `The opset for the ONNX domain must be explicitly specified.` | `CreateModel` did not include the ONNX domain. |
| `Node input '<name>' is not a graph input, initializer, or output of a previous node.` | A node input name is not connected. |
| `No opset import for domain '<domain>'` | A node uses a domain that the model does not declare. |
| `Domain version can not be changed for '<domain>'.` | An update model declares a different version for an existing domain. |
| `data_is_external=true requires the tensor to be larger than 127 bytes ...` | A small initializer was marked external. |
| `Only tensor types are supported currently` | `CreateValueInfo` received a non-tensor `OrtTypeInfo`. |
| `Invalid OrtGraph variant for use in the OrtModelEditorApi` | An EP-provided graph object was passed to a Model Editor function. |

## Where to make changes

- Public C declarations and API documentation: `include/onnxruntime/core/session/onnxruntime_c_api.h`, in
  `OrtModelEditorApi`. This is an append-only ABI table.
- C API implementation and function table: `onnxruntime/core/session/model_editor_c_api.cc`.
- Internal temporary object types and ownership: `onnxruntime/core/graph/model_editor_api_types.h`. These types derive
  from the shared `OrtGraph`, `OrtNode`, and `OrtValueInfo` bases in `onnxruntime/core/graph/abi_graph_types.h`. A new
  virtual method added there for the EP graph API must also be implemented, or return `NOT_IMPLEMENTED`, in the
  Model Editor types.
- Conversion of `OrtModel` additions into the runtime graph: `onnxruntime/core/graph/graph.cc` and
  `onnxruntime/core/graph/model.cc`.
- Model-level opset merging: `Model::LoadFromModelEditorApiModel` in `model.cc`. Update-time domain checks:
  `Graph::UpdateUsingModelEditorApiModel` in `graph.cc`.
- Loading, applying updates, and finalizing: `InferenceSession::Load(const OrtModel&)` and
  `InferenceSession::ApplyUpdates` in `onnxruntime/core/session/inference_session.cc`. Session helpers are in
  `onnxruntime/core/session/utils.cc`, including the `OrtModel` load path used by the Compile API.
- C++ declarations and wrappers: `include/onnxruntime/core/session/onnxruntime_cxx_api.h` and
  `include/onnxruntime/core/session/onnxruntime_cxx_inline.h`.
- Tests and canonical examples: `onnxruntime/test/shared_lib/test_model_builder_api.cc`.

When adding a public function, append it to `OrtModelEditorApi` and to the initializer in `model_editor_c_api.cc`; never
remove or reorder existing entries. The C# bindings also depend on the table layout. Keep declarations inside the
header's `#if !defined(ORT_MINIMAL_BUILD)` block, and add a `\since` tag. Release-boundary markers and `static_assert`
slot checks are added during release preparation, not with the function; see
[Versioning](Versioning.md) and `.github/instructions/c-api.instructions.md`. Add the C++ wrapper when appropriate and
update the shared-library tests.

## Design history

- [#23223](https://github.com/microsoft/onnxruntime/pull/23223) introduced complete model creation and augmentation.
- [#26015](https://github.com/microsoft/onnxruntime/pull/26015) made created and saved models include required internal
  ORT domain opset imports.
- [#28758](https://github.com/microsoft/onnxruntime/pull/28758) added validation for duplicate and null node attributes.
- [#28800](https://github.com/microsoft/onnxruntime/pull/28800) clarified ownership transfer and made mutating C and C++
  calls strongly exception safe.
