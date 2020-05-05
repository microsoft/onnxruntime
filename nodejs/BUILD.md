## Building Node.js binding

### Pre-Requisites

1.  Node.js 12.x

### Build Instructions

Currently it takes 4 steps to build Node.js binding:

1.  Build ONNX Runtime with flag `--build_shared` in repo root. See [Build](../BUILD.md) for more info.

2.  In current folder, run `npm install`. This will pull dev dependencies.

3.  Run `npm run build` to build binding.

4.  Run `npm test` run tests.

To consume the local built Node.js binding in a Node.js project:

```
npm install <onnxruntime-repo-root-folder>/nodejs
```

### Publish

Publishing a NPM package with addon requires 2 steps: publish NPM package itself, and publish prebuild binaries.

#### Publish NPM package

To publish a release:

```
npm publish
```

To publish a @dev release:

```
npm publish --tag dev
```

To create a npm package (for local use or debug purpose):

```
npm pack
```

NOTE: Need to publish the package from a clean build, otherwise extra files may be packed.

#### Publish prebuild binaries

Currently, prebuild binaries only support 3 platforms on x64: win32/linux/macos.

Prebuilds are currently uploaded manually.
