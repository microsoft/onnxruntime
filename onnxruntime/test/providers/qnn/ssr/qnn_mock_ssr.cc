// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <vector>
#include <thread>
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "qnn_mock_ssr_controller.h"
#include "rpcmem_utils.h"

extern "C" QnnMockSSRController* GetQnnMockSSRController() { return &QnnMockSSRController::Instance(); }

const QnnInterface_t** real_providerList{nullptr};
uint32_t real_numProviders{0};

namespace {
#if defined(_WIN32)
// Register a free_qnn_htp_fn to ensure we release QnnHtp.dll before
// destructing QnnMockSSR.dll
auto free_qnn_htp_fn = [](HMODULE m) {
  if (m) FreeLibrary(m);
};

std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(free_qnn_htp_fn)> qnn_htp(
    LoadLibraryW(L"QnnHtp.dll"), free_qnn_htp_fn);

FARPROC addr = GetProcAddress(qnn_htp.get(), "QnnInterface_getProviders");
typedef Qnn_ErrorHandle_t (*QnnApiFnType_t)(const QnnInterface_t***, uint32_t*);
QnnApiFnType_t real_QnnInterface_getProviders = reinterpret_cast<QnnApiFnType_t>(addr);
auto res = real_QnnInterface_getProviders((const QnnInterface_t***)&real_providerList, &real_numProviders);
#endif  // defined(_WIN32)
}  // namespace

#if defined(_WIN32)

QNN_API
Qnn_ErrorHandle_t QnnGraph_create(Qnn_ContextHandle_t contextHandle,
                                  const char* graphName,
                                  const QnnGraph_Config_t** config,
                                  Qnn_GraphHandle_t* graphHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphCreate(contextHandle, graphName, config, graphHandle);
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_getBuildId(const char** id) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.backendGetBuildId(id);
}

QNN_API
Qnn_ErrorHandle_t QnnLog_create(QnnLog_Callback_t callback,
                                QnnLog_Level_t maxLogLevel,
                                Qnn_LogHandle_t* logger) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.logCreate(callback, maxLogLevel, logger);
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_create(Qnn_LogHandle_t logHandle,
                                    const QnnBackend_Config_t** config,
                                    Qnn_BackendHandle_t* backend) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.backendCreate(logHandle, config, backend);
}

QNN_API
Qnn_ErrorHandle_t QnnContext_create(Qnn_BackendHandle_t backend,
                                    Qnn_DeviceHandle_t device,
                                    const QnnContext_Config_t** config,
                                    Qnn_ContextHandle_t* context) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.contextCreate(backend, device, config, context);
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_validateOpConfig(Qnn_BackendHandle_t backend,
                                              Qnn_OpConfig_t opConfig) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.backendValidateOpConfig(backend, opConfig);
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_addNode(Qnn_GraphHandle_t graph, Qnn_OpConfig_t opConfig) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphAddNode(
      graph, opConfig);
}

QNN_API
Qnn_ErrorHandle_t QnnTensor_createGraphTensor(Qnn_GraphHandle_t graph, Qnn_Tensor_t* tensor) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, tensor);
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_retrieve(Qnn_ContextHandle_t contextHandle,
                                    const char* graphName,
                                    Qnn_GraphHandle_t* graphHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphRetrieve(contextHandle, graphName, graphHandle);
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_finalize(Qnn_GraphHandle_t graph,
                                    Qnn_ProfileHandle_t profileHandle,
                                    Qnn_SignalHandle_t signalHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphFinalize(graph, profileHandle, signalHandle);
}

QNN_API
Qnn_ErrorHandle_t QnnContext_getBinarySize(Qnn_ContextHandle_t context,
                                           Qnn_ContextBinarySize_t* binaryBufferSize) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.contextGetBinarySize(context, binaryBufferSize);
}

QNN_API
Qnn_ErrorHandle_t QnnContext_getBinary(Qnn_ContextHandle_t context,
                                       void* binaryBuffer,
                                       Qnn_ContextBinarySize_t binaryBufferSize,
                                       Qnn_ContextBinarySize_t* writtenBufferSize) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.contextGetBinary(
      context, binaryBuffer, binaryBufferSize, writtenBufferSize);
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_execute(Qnn_GraphHandle_t graphHandle,
                                   const Qnn_Tensor_t* inputs,
                                   uint32_t numInputs,
                                   Qnn_Tensor_t* outputs,
                                   uint32_t numOutputs,
                                   Qnn_ProfileHandle_t profileHandle,
                                   Qnn_SignalHandle_t signalHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphExecute(graphHandle,
                                                                   inputs, numInputs, outputs, numOutputs, profileHandle, signalHandle);
}
#endif  // defined(_WIN32)

extern "C" Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList,
                                                       uint32_t* numProviders) {
  static QnnInterface_t interface;
  interface.backendId = 0;
  interface.providerName = "MockSSR";
#if defined(_WIN32)
  interface.apiVersion = real_providerList[0]->apiVersion;
  interface.QNN_INTERFACE_VER_NAME = real_providerList[0]->QNN_INTERFACE_VER_NAME;
  switch (QnnMockSSRController::Instance().GetTiming()) {
    case QnnMockSSRController::Timing::BackendGetBuildId:
      interface.QNN_INTERFACE_VER_NAME.backendGetBuildId = QnnBackend_getBuildId;
      break;
    case QnnMockSSRController::Timing::BackendCreate:
      interface.QNN_INTERFACE_VER_NAME.backendCreate = QnnBackend_create;
      break;
    case QnnMockSSRController::Timing::ContextCreate:
      interface.QNN_INTERFACE_VER_NAME.contextCreate = QnnContext_create;
      break;
    case QnnMockSSRController::Timing::BackendValidateOpConfig:
      interface.QNN_INTERFACE_VER_NAME.backendValidateOpConfig = QnnBackend_validateOpConfig;
      break;
    case QnnMockSSRController::Timing::LogCreate:
      interface.QNN_INTERFACE_VER_NAME.logCreate = QnnLog_create;
      break;
    case QnnMockSSRController::Timing::ContextGetBinarySize:
      interface.QNN_INTERFACE_VER_NAME.contextGetBinarySize = QnnContext_getBinarySize;
      break;
    case QnnMockSSRController::Timing::ContextGetBinary:
      interface.QNN_INTERFACE_VER_NAME.contextGetBinary = QnnContext_getBinary;
      break;
    case QnnMockSSRController::Timing::TensorCreateGraphTensor:
      interface.QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor = QnnTensor_createGraphTensor;
      break;
    case QnnMockSSRController::Timing::GraphCreate:
      interface.QNN_INTERFACE_VER_NAME.graphCreate = QnnGraph_create;
      break;
    case QnnMockSSRController::Timing::GraphRetrieve:
      interface.QNN_INTERFACE_VER_NAME.graphRetrieve = QnnGraph_retrieve;
      break;
    case QnnMockSSRController::Timing::GraphAddNode:
      interface.QNN_INTERFACE_VER_NAME.graphAddNode = QnnGraph_addNode;
      break;
    case QnnMockSSRController::Timing::GraphFinalize:
      interface.QNN_INTERFACE_VER_NAME.graphFinalize = QnnGraph_finalize;
      break;
    case QnnMockSSRController::Timing::GraphExecute:
      interface.QNN_INTERFACE_VER_NAME.graphExecute = QnnGraph_execute;
      break;
    default:
      break;
  }
#endif  // defined(_WIN32)
  static std::vector<const QnnInterface_t*> m_providerPtrs = {&interface};
  *providerList = m_providerPtrs.data(),
  *numProviders = 1;
  return QNN_SUCCESS;
}
