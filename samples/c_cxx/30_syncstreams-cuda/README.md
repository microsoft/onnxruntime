# SyncStream

The latest revision of ORT provides a mechanism to sync between multiple streams. Consider an example where upload work happens on a stream and inference happens on a different stream, we want the inference stream to wait for the upload stream completion. This is now possible with the newly introduced ORT APIs.

ORT introduces SyncStreams and SyncNotifications created from SyncStreams  to sync between streams. Activating a notification object is similar to Signal call and it also provides WaitOnDevice and WaitOnHost APIs for waiting on other streams.  
The following example highlights the synchronisation between streams:

```c
const OrtSyncStreamImpl* uploadStreamImpl;
OrtSyncNotificationImpl* uploadNotification;
OrtEpApi ortEpApi = *ortApi.GetEpApi();
uploadStreamImpl = ortEpApi.SyncStream_GetImpl(upload_stream);
uploadStreamImpl->CreateNotification(const_cast<OrtSyncStreamImpl*>(uploadStreamImpl), &uploadNotification);

// This should now be a truly asynchronous copy because the source (cpuInputFloat) is pinned memory.
std::vector<const OrtValue*> cpu_src_ptrs = { full_cpu_tensor };
std::vector<OrtValue*> gpu_dst_ptrs = { full_gpu_tensor };
ortApi.CopyTensors(ortEnvironment, cpu_src_ptrs.data(), gpu_dst_ptrs.data(), upload_stream, cpu_src_ptrs.size());

uploadNotification->Activate(uploadNotification);
uploadNotification->WaitOnDevice(uploadNotification, stream);

// work on the inference stream
void* cuda_compute_stream_handle = ortApi.SyncStream_GetHandle(stream);
cudaMemcpy2DAsync(..., static_cast<cudaStream_t>(cuda_compute_stream_handle));

input_tensors.push_back(std::move(inference_gpu_input_tensor));
output_tensors.push_back(std::move(inference_gpu_output_tensor));
Ort::IoBinding iobinding(session);
iobinding.BindInput(InputTensorName.get(), input_tensors[0]);
iobinding.BindOutput(OutputTensorName.get(), output_tensors[0]);

std::vector<const char*> input_names = { "input" };
std::vector<const char*> output_names = { "output" };
session.Run(Ort::RunOptions{}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_tensors.data(), output_tensors.size());

```

It must be noted Syncstream currently is only tested to work with CUDA streams (so CUDA EP and TRT RTX EP only). We have used TRT RTX EP in our sample application.

## Dependencies

This sample vendors a copy of https://github.com/lvandeve/lodepng (Zlib license)
