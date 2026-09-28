# Workstream `node-migration`: Node Plugin Migration

Status: Working plan

[WebGPU EP extraction overview](../webgpu_ep_extraction.md)

## Objective

Replace the WebGPU EP currently bundled in `onnxruntime-node` with an explicitly consumable plugin while preserving a
supported migration path for existing Node WebGPU users.

This is a required extraction workstream, not optional new consumer enablement. Built-in Node WebGPU support must not
be removed until the replacement package and loading model are tested on the agreed launch platforms.

## Current state

`onnxruntime-node` currently:

- Includes WebGPU in several prebuilt platform binaries.
- Recognizes the `"webgpu"` execution provider through compile-time provider-specific code.
- Accepts WebGPU-specific provider options.
- Supports WebGPU buffer interoperability.
- Uses a singleton native `Ort::Env`; JavaScript callers do not create independent ORT environment instances.

Removing the provider from these binaries changes an existing, although experimental, capability and requires an
explicit compatibility and package transition.

## Desired end state

- The Node ORT host does not compile, bundle, or depend on the WebGPU implementation.
- Node can register optional plugin EP libraries through a generic API.
- A separately installable WebGPU package supplies supported platform-specific plugin artifacts.
- Existing WebGPU provider options and buffer interoperability continue to work.
- Loading errors identify missing packages, incompatible versions, unsupported platforms, and late registration.
- The mechanism supports other plugin EPs without adding provider-specific Node binding code.
- Existing users have a documented package and code migration path.

## Generic Node plugin loading

The Node binding should expose an API to register a plugin library before creating sessions that use it.

The current binding owns a singleton `Ort::Env`, so the initial API should register plugins with that singleton.
If Node later exposes user-created ORT environment instances, registration can be extended to those instances.

The API should define:

- Whether registration is explicit or may also be triggered by a package helper.
- Required ordering relative to ORT initialization and session creation.
- Library lifetime and cleanup across Node worker environments.
- Duplicate registration behavior.
- Error handling for incompatible plugin and ORT versions.
- How provider names and options become visible to session creation.
- How a package safely resolves its platform-specific native library.

## WebGPU npm package

The WebGPU repository should publish an npm package that:

- Contains or installs the appropriate native WebGPU plugin artifact for each supported platform and architecture.
- Exposes a small helper that resolves and registers the artifact through the generic Node API.
- Does not require WebGPU-specific code in the Node ORT binding.
- Declares compatibility independently from the ORT package version.
- Produces clear unsupported-platform and compatibility errors.
- Follows the same signing, provenance, and release requirements as comparable ORT packages.

The exact package name is open.

## Core Node package transition

Two primary package strategies remain under consideration:

| Strategy | Advantages | Costs and risks |
| --- | --- | --- |
| Keep `onnxruntime-node` as the core host and remove bundled WebGPU in a major-version transition | Preserves the established package name and avoids maintaining two core packages | Existing WebGPU users must install and register a second package; capability removal requires prominent migration guidance |
| Introduce a core-only package such as `onnxruntime-node-core` | Allows the existing `onnxruntime-node` contract to remain stable during migration and makes the optional boundary explicit | Creates a new ecosystem package, duplicates support or requires a later consolidation, and may confuse which package applications should choose |

An extended compatibility period may accompany either strategy, but indefinitely bundling WebGPU is not the target
state. The decision should account for the experimental status of current WebGPU support, semantic-versioning policy,
download size, other built-in EPs, and maintenance cost.

## Compatibility requirements

The migration must preserve:

- Session creation using the `"webgpu"` provider name or a clearly documented replacement.
- Existing provider options.
- GPU-buffer input and output behavior on supported platforms.
- Node worker behavior and native library lifetime safety.
- Current supported platform coverage unless a reduction is explicitly approved.
- Clear detection of ORT/plugin version incompatibility.

## Tests

Required coverage includes:

- Installation of the core Node package without WebGPU.
- Installation and registration of the WebGPU plugin package.
- Session creation and inference with CPU fallback disabled.
- Existing WebGPU provider options.
- GPU-buffer interoperability.
- Missing-plugin, unsupported-platform, duplicate-registration, and version-mismatch failures.
- Node worker initialization and cleanup.
- Upgrade tests for the selected package transition.

## Work packages

1. **API design:** define generic plugin registration for the singleton Node ORT environment.
2. **Binding implementation:** load and register arbitrary plugin EP libraries.
3. **Package prototype:** package WebGPU native artifacts and registration helper.
4. **Compatibility validation:** preserve options, buffers, workers, and diagnostics.
5. **Package transition decision:** select naming, versioning, and deprecation policy.
6. **Release migration:** publish packages, documentation, and upgrade tests before removing bundled WebGPU.

The API and package prototypes can proceed in parallel once the native plugin artifact contract is known.

## Interfaces with other workstreams

### Plugin boundary and Web/Wasm integration

- Reuses ORT's generic dynamic plugin registration and compatibility behavior.
- Does not depend on the static WebAssembly registration path.

### Provider isolation and repository migration

- Consumes versioned native shared WebGPU plugin artifacts.
- Coordinates platform naming, signing, compatibility metadata, and release timing.

### Test ownership and operator conformance

- Supplies Node host and package integration tests.
- Reuses portable operator cases where practical to verify the loaded provider executes correctly.

## Completion criteria

- A package naming and compatibility strategy is approved.
- The Node binding registers plugin EPs without WebGPU-specific compile-time code.
- A separately installable WebGPU package exists for the agreed launch platforms.
- Current WebGPU options and buffer interop pass against the plugin.
- Package installation, worker, compatibility, and failure-mode tests are blocking.
- Existing users have migration documentation and a supported transition window.
- Bundled WebGPU is removed only after the replacement is released and validated.

## Open questions

- Should the core host remain `onnxruntime-node` or move to a name such as `onnxruntime-node-core`?
- What should the WebGPU npm package be named?
- Should plugin registration be an explicit application call, a package helper side effect, or both?
- Which Node platforms and architectures are required at first release?
- How long should any compatibility or deprecation period last?
- How should plugin loading behave across Node worker environments?
