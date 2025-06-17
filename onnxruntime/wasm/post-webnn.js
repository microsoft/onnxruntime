// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This file contains the post-run code for the ORT WebAssembly module. The code in this file will be injected into the
// final module using Emscripten's `--post-js` option.
//
// This file will only be used in build with flag `--use_webnn`.

/**
 * This function is called only once when initializing the WebNN backend.
 *
 * @param params WebNN initialization parameters.
 */
Module["webnnInit"] = (params) => {
  // Functions called from EM_ASM need to be assigned in a way that can be minified.
  // Functions called via emscripten::val::module_property need to be assigned by name so that the minifier doesn't
  // change the name.

  const backend = params[0];
  [
    Module.webnnReserveTensorId,
    Module.webnnReleaseTensorId,
    Module["webnnEnsureTensor"],
    Module.webnnUploadTensor,
    Module["webnnDownloadTensor"],
    Module.webnnRegisterMLContext,
    Module["webnnEnableTraceEvent"],
  ] = params.slice(1);

  // This function is called from both JS and an EM_ASM block, it needs both a minifiable name and an explicit name.
  Module["webnnReleaseTensorId"] = Module.webnnReleaseTensorId;
  Module["webnnUploadTensor"] = Module.webnnUploadTensor;
  Module["webnnRegisterMLContext"] = Module.webnnRegisterMLContext;

  // Functions called from JS also need to have explicit names.
  Module["webnnOnRunStart"] = (sessionId) => {
    return backend["onRunStart"](sessionId);
  };
  Module["webnnOnRunEnd"] = backend["onRunEnd"].bind(backend);
  Module["webnnOnReleaseSession"] = (sessionId) => {
    backend["onReleaseSession"](sessionId);
  };
  Module["webnnCreateMLTensorDownloader"] = (tensorId, type) => {
    return backend["createMLTensorDownloader"](tensorId, type);
  };
  Module["webnnRegisterMLTensor"] = (sessionId, tensor, dataType, shape) => {
    return backend["registerMLTensor"](sessionId, tensor, dataType, shape);
  };
  Module["webnnCreateMLContext"] = (optionsOrGpuDevice) => {
    return backend["createMLContext"](optionsOrGpuDevice);
  };
  Module["webnnRegisterMLConstant"] = (
    externalFilePath,
    dataOffset,
    dataLength,
    builder,
    desc,
    shouldConvertInt64ToInt32
  ) => {
    return backend["registerMLConstant"](
      externalFilePath,
      dataOffset,
      dataLength,
      builder,
      desc,
      Module.MountedFiles,
      shouldConvertInt64ToInt32
    );
  };
  Module["webnnRegisterGraphInput"] =
    backend["registerGraphInput"].bind(backend);
  Module["webnnIsGraphInput"] = backend["isGraphInput"].bind(backend);
  Module["webnnRegisterGraphOutput"] =
    backend["registerGraphOutput"].bind(backend);
  Module["webnnIsGraphOutput"] = backend["isGraphOutput"].bind(backend);

  Module["webnnCreateTemporaryTensor"] =
    backend["createTemporaryTensor"].bind(backend);
  Module["webnnIsGraphInputOutputTypeSupported"] =
    backend["isGraphInputOutputTypeSupported"].bind(backend);
};
