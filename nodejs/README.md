# ONNX Runtime Node.js API

## Requirements
Node.js version 12.x is required.


### Supported Platforms

- Windows x64 CPU NAPI_v3
- Linux x64 CPU NAPI_v3
- MacOS x64 CPU NAPI_v3

## Usage
Samples can be found [here](../samples/nodejs).

Install the latest stable version: `npm install onnxruntime`

Install the latest dev version: `npm install onnxruntime@dev`

## Building
1. Refer to the main project's [build instructions](../BUILD.md). Requires the `build_shared_lib` option.

2.  In current folder, run `npm install` to pull dev dependencies.

3.  Run `npm run build` to build binding.

4.  Run `npm test` to run tests.

To consume the local built Node.js binding in a Node.js project: `npm install <onnxruntime-repo-root-folder>/nodejs`

### Publish
Publishing a NPM package with addon requires 2 steps: publish NPM package itself, and publish prebuild binaries.

#### Publish NPM package
*NOTE: Packaged needs to be published from a clean build, otherwise extra files may be packed.*

* To publish a release: `npm publish`
* To publish a @dev release: `npm publish --tag dev`
* To create a npm package (for local use or debug purpose): `npm pack`

#### Publish prebuild binaries
Prebuilds are currently uploaded manually.
