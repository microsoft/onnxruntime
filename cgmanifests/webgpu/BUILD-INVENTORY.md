# WebGPU Component Governance inventory

WebGPU plugin packaging generates a `metadata/cgmanifest.json` artifact after
building on Windows, Linux, and macOS, then explicitly scans that directory.
The regular pipeline scan excludes `Build.BinariesDirectory`, where Dawn's
fetched sources normally live.

`generate_webgpu_cgmanifest.py` uses the **same dependency list** as the fetched
Dawn revision's `tools/fetch_dawn_dependencies.py`. That upstream script fetches
sources but does not generate a Component Governance manifest. ORT reads its
literal `required_submodules` list and the corresponding `DEPS` entries without
executing either Python file. It verifies each fetched Git checkout's actual
commit against the pin. Missing or stale checkouts fail generation.

The generator does not maintain a second list of Dawn dependency names, paths, or commits.
The separately maintained `cgmanifest.webgpu.json` records reviewed classification metadata
for the [Dawn upgrade workflow](README.md); it is not an input to this generator.
New dependencies selected by Dawn's fetcher are included automatically; entries
not selected by the fetcher are excluded. If upstream changes the metadata
syntax beyond what the reader supports, generation fails rather than emitting
an incomplete inventory.

## Inputs and scope

- **Dawn:** the archive URL, version, and SHA1 from `cmake/deps.txt`. FetchContent
  archives have no root Git metadata, so the manifest records an `other`
  archive component instead of requiring a separately maintained tag-to-commit
  mapping or a network lookup.
- **Fetched dependencies:** public repository URLs from Dawn's DEPS and verified
  checkout commits. GitHub mirror URLs are normalized to GitHub; authenticated
  CI Git redirects are not read or emitted.
- **Redistributed DXC (Windows only):** a separate archive component derived
  from the packaging pipeline's existing URL and SHA256. This is distinct from
  the DXC source checkout used during the build.

This is a fetched **build-input inventory**, including fetched test/build tools.
It is not a link analysis or a declaration that every fetched component ships
in every package. It does not automatically mark components as development-only
or omit unfamiliar components. ORT may also substitute its own Abseil/Protobuf
inputs for Dawn's fetched copies; those remain covered by ORT's dependency
inventory. Package license/NOTICE contents still require review against what is
actually redistributed. This change does not regenerate `ThirdPartyNotices.txt`.

## Local use

Run after configuring a fresh WebGPU build with the matching ORT source pin:

```powershell
python cgmanifests/webgpu/generate_webgpu_cgmanifest.py --dawn-source build/Windows/Release/_deps/dawn-src --output build/webgpu-metadata/cgmanifest.json
```

For the Windows plugin package, additionally pass:

```text
--dxc-pipeline tools/ci_build/github/azure-pipelines/stages/plugin-win-webgpu-stage.yml
```

`--deps-file` can select another `cmake/deps.txt` when inspecting a build from a
different revision. It must describe the Dawn archive that populated the source
directory. `--output` can be repeated to write identical copies. Custom or external
Dawn builds must supply matching metadata and fetched checkouts; arbitrary
prebuilt Dawn packages are outside this generator's scope.

Run the regression tests with:

```powershell
python cgmanifests/webgpu/test_generate_webgpu_cgmanifest.py
```

The platform packaging templates run these tests before inventory generation.
