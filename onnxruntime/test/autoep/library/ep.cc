#include "ep.h"
#include "ep_stream_support.h"

/*static*/
const char* ExampleEp ::GetNameImpl(const OrtEp* this_ptr) {
  const auto* ep = static_cast<const ExampleEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ExampleEp::CreateSyncStreamForDeviceImpl(OrtEp* this_ptr, /*const OrtSession* session,*/
                                                    const OrtMemoryDevice* memory_device,
                                                    OrtSyncStream** stream) {
  auto& ep = *static_cast<ExampleEp*>(this_ptr);

  auto sync_stream = std::make_unique<StreamImpl>(ep);
  return ep.ep_api.CreateSyncStream(memory_device, sync_stream.get(), stream);
}
