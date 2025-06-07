#include "ep.h"
#include "ep_factory.h"
#include "ep_stream_support.h"

ExampleEp::ExampleEp(ExampleEpFactory& factory, const std::string& name,
                     const OrtSessionOptions& session_options, const OrtLogger& logger)
    : ApiPtrs(static_cast<const ApiPtrs&>(factory)),
      factory_{factory},
      name_{name},
      session_options_{session_options},
      logger_{logger} {
  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  ReleaseDataTransfer = ReleaseDataTransferImpl;

  auto status = ort_api.Logger_LogMessage(&logger_,
                                          OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                          ("ExampleEp has been created with name " + name_).c_str(),
                                          ORT_FILE, __LINE__, __FUNCTION__);
  // ignore status for now
  (void)status;
}

/*static*/
const char* ExampleEp ::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const ExampleEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ExampleEp::CreateSyncStreamForDeviceImpl(OrtEp* this_ptr, /*const OrtSession* session,*/
                                                    const OrtMemoryDevice* memory_device,
                                                    OrtSyncStreamImpl** stream) noexcept {
  auto& ep = *static_cast<ExampleEp*>(this_ptr);
  *stream = nullptr;

  // we only need stream synchronization on the device stream
  if (ep.ep_api.OrtMemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_DEFAULT) {
    auto sync_stream = std::make_unique<StreamImpl>(ep);
    *stream = sync_stream.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEp::CreateDataTransferImpl(OrtEp* this_ptr,
                                                          OrtDataTransferImpl** data_transfer) noexcept {
  auto& ep = *static_cast<ExampleEp*>(this_ptr);

  // ExampleEpFactory has a single shared instance of OrtDataTransferImpl and returns it as a const pointer to make
  // the ownership semantics clear.
  // ORT can't assume the instance is shared and needs a non-const instance to call ReleaseDataTransfer with.
  // due to this we const_cast here and implement ReleaseDataTransferImpl as a no-op.
  *data_transfer = const_cast<OrtDataTransferImpl*>(ep.factory_.GetDataTransfer());
  return nullptr;
}
