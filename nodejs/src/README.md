# ONNX Runtime Node.js API

This directory contains the Node.js binding for the ONNX runtime.

## Installation

TBD: Use the following command to install:

```
npm install onnxruntime
```

### Building

TBD: Use the main project's [build instructions](../BUILD.md) with the `--build_nodejs` option.

Currently it takes 4 steps to build Node.js binding:

1.  Build ONNX Runtime with flag `--build_shared` in repo root.

2.  In current folder, Install NPM packages:

    ```
    npm install
    ```

    This will pull dev dependencies.

3.  Run `npm run build` to build binding.

4.  Run `npm test` run tests.

### Publish

Currently we can use command `npm pack` to pack the whole binding project into one `.tar.gz` file. This file can be used in `npm install <file path>` to install the package locally, with the same behavior of installing from a NPM registery.

TODO: publish to official NPM registery.

## Examples

Check [examples](../examples) for the proposed API usage.
