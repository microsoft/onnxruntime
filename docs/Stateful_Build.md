# Stateful build boundaries

The stateful engine is built as the private `onnxruntime_stateful` target. It
depends on the existing session and framework targets; those targets do not link
to the stateful engine or include its private headers. A composition bridge
connects the stable C API tables to the engine in the final ORT product. Generic
execution-provider registration remains a core contract, so an independently
built provider can register with either configuration.

Configuration checks the core targets' CMake link dependency graph for stateful
targets. A source-boundary test also rejects direct core includes of private
stateful headers. The feature definition is limited to the composition bridge,
bindings, plugin registration, and availability tests rather than all core
targets. The API-table bridge is a composition contract, not evidence that the
session archive is independently linkable without the final product. The engine
still uses concrete core C++ session types; it is not yet isolated behind a
mockable interface.

The `onnxruntime_ENABLE_STATEFUL` CMake option defaults to `ON` in full builds
and is disabled in minimal and extended-minimal builds. To omit the engine from
a full build, pass `--cmake_extra_defines onnxruntime_ENABLE_STATEFUL=OFF` to
the ORT build script. The C API remains ABI-compatible: `GetStatefulApi`
returns null, and Model Package chain creation returns `ORT_NOT_IMPLEMENTED`
without changing its output. In full builds, ordinary sessions and Model
Package APIs remain available. Python configuration objects remain importable;
stateful execution reports an unavailable-feature error.

Consumers should link the `onnxruntime` product target rather than the private
stateful or bridge targets. This separation provides an optional build boundary,
not independent plug-in interfaces for cache policy, samplers, or tokenizers.
