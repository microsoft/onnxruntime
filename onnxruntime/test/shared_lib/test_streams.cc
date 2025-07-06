// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#include <cuda_runtime.h>
#endif

/*
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
int32_t FENCE_STATE_UPLOAD_DONE = 1;
int32_t FENCE_STATE_KERNEL_DONE = 1;

TEST(StreamsTest, UseStreamForAsyncInputCopy) {
  OrtSession* session;
  OrtEnv* env = ort_env.get()->operator OrtEnv*();
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  size_t num_devices = 0;
  const OrtEpDevice* const* ep_devices = nullptr;
  api->GetEpDevices(*ort_env, &ep_devices, &num_devices);

  const OrtEpDevice* ep_device = ep_devices[0];  // Select EP

  // Create Stream for session input/output.
  // OrtSyncStream can provide a native handle (e.g. cudaStream_t) to the user.
  OrtSyncStream* stream = nullptr;
  api->CreateSyncStreamForDevice(ep_device, &stream);

  // Update the internal ORT Stream class behind OrtSyncStream to add something like this:
  //   WaitOnInput
  //   SignalInputAvailable
  //   WaitOnOutput
  //   SignalOutputAvailable
  //
  // Ideally this is limited to the derived class used for plugin EPs, as that's the only one that would be
  // accessible to a user via OrtSyncStream.

  // add the ability to bind a stream to an external synchronization primitive
  void* sharedFenceHandle = nullptr;  // get from D3D12
  void* wait_info = reinterpret_cast<void*>(FENCE_STATE_UPLOAD_DONE);
  void* signal_info = reinterpret_cast<void*>(FENCE_STATE_KERNEL_DONE);

  // add the info necessary to setup the external sync to the Stream internals
  // implementation should return failure if it doesn't support the external sync primitive
  api->BindSyncStreamToExternalSync(stream, ExternalSyncPrimitive_D3D12Fence,
                                    sharedFenceHandle, wait_info, signal_info);

  // for sync between two ORT Stream instances with no interop we can have an onnxruntime::sync::Notification
  // for input and one for output in the Stream instance. these can be added in CreateSyncStreamForDevice.

  // User provides the OrtSyncStream for input/output sync in a new OrtSession Run or RunAsync API (TBD - probably needs to be RunAsync)

  // ORT calls Stream::WaitOnInput inside InferenceSession::Run prior to running anything that uses the OrtDevice
  // in the OrtSyncStream the user provided
  //   - if the stream in bound to external sync primitive the EP implementation is called to do the wait
  //     - this would wait using the wait_info from the BindSyncStreamToExternalSync
  //   - if there is an onnxruntime::sync::Notification for input in the Stream instance we wait on that

  // User signals input is available
  //   - for interop sync, this happens outside of the ORT APIs
  //     - e.g. user code calls `pCommandQueue->Signal(pFence, FENCE_STATE_UPLOAD_DONE);`
  //   - otherwise user calls OrtApi SignalSyncStream which activates the input Notification in the Stream
  //     by calling SignalInputAvailable
  api->SignalSyncStream(stream);

  // User waits on output
  //  - for interop sync this happens outside of the ORT APIs
  //    - e.g. user code calls `pCommandQueue->Wait(pFence, FENCE_STATE_KERNEL_DONE);`
  //  - otherwise user calls WaitOnSyncStream which waits on the output Notification member in the Stream
  //    by calling WaitOnOutput
  api->WaitOnSyncStream(stream);

  // model runs

  // at end of model run signal output is available by calling SignalOutputAvailable
  //   - for interop sync signal the external sync primitive using signal_info
  //   - otherwise activate the output Notification in the Stream that WaitOnSyncStream is waiting on
}
}  // namespace test
}  // namespace onnxruntime
*/
