# Release Process

This document describes the release conventions and process for the WebGPU plugin EP.

## Versioning

The plugin follows [Semantic Versioning](https://semver.org/):

- **MAJOR** — incompatible API/ABI changes.
- **MINOR** — backwards-compatible feature additions.
- **PATCH** — backwards-compatible bug and security fixes.

The current version is tracked in [VERSION_NUMBER](VERSION_NUMBER).

## Branch and tag naming

All release refs are namespaced under `plugin-ep-webgpu/` so they group together in `git branch` / `git tag`
listings and don't collide with the main ONNX Runtime release refs.

- **Release branch:** `plugin-ep-webgpu/rel-X.Y`
  - One branch per minor version line (e.g. `plugin-ep-webgpu/rel-1.0`).
  - Holds all patch releases for that minor line (1.0.0, 1.0.1, 1.0.2, ...).
  - Forked from `main` at the point of the first release on that line.
- **Release tag:** `plugin-ep-webgpu/vX.Y.Z`
  - One tag per shipped release (e.g. `plugin-ep-webgpu/v1.0.0`).
  - Tags are immutable and are the source of truth for "what shipped."
- **Pre-release tag:** `plugin-ep-webgpu/vX.Y.Z-rc.N` (semver-style)
  - Used for release candidates and other pre-release artifacts.
  - Note: this convention is forward-looking as we don't have release candidates in the release process yet.

The `rel-` prefix on branches and the `v` prefix on tags ensure branches and tags are never ambiguous at the ref
level.

### Difference from the main ONNX Runtime convention

The main ORT repo uses **per-patch** release branches of the form `rel-X.Y.Z` (e.g. `rel-1.20.0`, `rel-1.20.1`).
This plugin deliberately uses **per-minor** branches (`rel-X.Y`) instead.

The per-minor model is simpler: one long-lived branch per supported minor line, with each patch release marked by a
tag on that branch. Tags are the immutable record of what shipped; the branch is just where the next patch is staged.
For a component of this size and release cadence, that is sufficient and avoids the branch sprawl of the per-patch
model.

The per-minor model is also the broader open-source convention (Linux, LLVM, Python, Node, Kubernetes), so
contributors coming from outside the ORT ecosystem will find it familiar. The namespaced ref prefix
(`plugin-ep-webgpu/`) keeps the plugin's release refs cleanly separated from the main ORT release refs.

## Release workflow

### Step 1: Prepare release branch

Follow whichever sub-section matches the release type.

#### New minor or major release
1. Create release branch `plugin-ep-webgpu/rel-X.Y` from `main`. `main`'s `VERSION_NUMBER` should already be `X.Y.0`,
   reflecting the release that is about to be cut.
2. Bump `VERSION_NUMBER` on `main` to the next development version (e.g. if cutting `1.0.0`, bump `main` to `1.1.0`).

#### Patch release
1. Bump `VERSION_NUMBER` on existing release branch `plugin-ep-webgpu/rel-X.Y` to `X.Y.Z`.

### Step 2: Integrate fixes, build, and validate packages

1. Integrate any fixes into the release branch. These may be cherry-picked from `main` or made directly in the release
   branch. The latter should be re-integrated into `main` unless the fix is specific to the release branch.
2. Run the [packaging pipeline](https://aiinfra.visualstudio.com/Lotus/_build?definitionId=2186) at the release branch
   tip.
3. Verify that the subsequent [packaging test pipeline](https://aiinfra.visualstudio.com/Lotus/_build?definitionId=2212)
   run succeeds. A successful packaging pipeline run will trigger a packaging test pipeline run.
4. Optionally, run the publishing pipelines to publish a test version of the packages:
   - [NuGet publishing](https://aiinfra.visualstudio.com/Lotus/_build?definitionId=2229). Set the `PublishLocation`
     parameter to "nugettest" (int.nugettest.org) or "ado" (ADO nightly feed). **⚠ Caution:** this pipeline is also
     used to publish to nuget.org — double-check the `PublishLocation` value before running.
   - [Test Python publishing](https://aiinfra.visualstudio.com/Lotus/_build?definitionId=2225). Publishes to ADO
     nightly feed.
5. Run any other manual validation as needed.

Repeat the steps in this section as many times as needed.

### Step 3: Release

1. Run the packaging pipeline and set the `Package Version` parameter to "release". This run produces the release
   build.
2. Run the publishing pipelines to publish the packages. Take care to select the release build from item 1 as the
   source packaging pipeline run.
   - [NuGet publishing](https://aiinfra.visualstudio.com/Lotus/_build?definitionId=2229). Set the `PublishLocation`
     parameter to "nuget".
   - [Python publishing](https://aiinfra.visualstudio.com/Lotus/_build?definitionId=2228).
3. Tag the release branch at the same commit that the release build was created from with tag
   `plugin-ep-webgpu/vX.Y.Z`.
4. Create a new GitHub release at that tag.
