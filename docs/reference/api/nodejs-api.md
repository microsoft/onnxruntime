---
title: Node.js API
parent: API docs
grand_parent: Reference
nav_order: 4
---

# ONNX Runtime Node.js API
{: .no_toc }

ONNX Runtime Node.js binding enables Node.js applications to run ONNX model inference.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Supported Versions
Node.js v12.x+ or Electron v5.x+

## API Reference
See [Typescript declarations](https://github.com/microsoft/onnxruntime/blob/master/nodejs/lib/inference-session.ts) and refer to [samples](#samples) for reference.

## Builds
Builds are published to **npm** and can be installed using `npm install`

| Artifact      | Description | Supported Platforms |
|-----------    |-------------|---------------------|
|[onnxruntime](https://www.npmjs.com/package/onnxruntime)|CPU (Release)| Windows x64 CPU NAPI_v3, Linux x64 CPU NAPI_v3, MacOS x64 CPU NAPI_v3|
|onnxruntime@dev| CPU (Dev)|Windows x64 CPU NAPI_v3, Linux x64 CPU NAPI_v3, MacOS x64 CPU NAPI_v3|

To use on platforms without pre-built binaries, you can [build Node.js binding from source]((../../how-to/build.md#apis-and-language-bindings)) and consume using `npm install <onnxruntime_repo_root>/nodejs/`. 

## Samples
See [Tutorials: Basics - NodeJS](../../tutorials/inferencing/api-basics.md#nodejs)


