// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents the platform-specific options for creating an InferenceSession.
    /// </summary>
    /// <see cref="https://onnxruntime.ai"/>
    /// <remarks>Exposes only those APIs supported on Android.</remarks>
    public partial class SessionOptions
    {
        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="nnapi_flags">nnapi specific flags</param>
        public void AppendExecutionProvider_Nnapi(NNAPIFlags nnapi_flags = NNAPIFlags.NNAPI_FLAG_USE_NONE)
            => NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(handle, (uint)nnapi_flags));

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="nnapi_flags">nnapi specific flag mask</param>
        public void AppendExecutionProvider_Nnapi(uint nnapi_flags = 0)
            => NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(handle, nnapi_flags));

        /// <summary>
        /// A helper method to construct a SessionOptions object for CUDA execution.
        /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider(int deviceId = 0)
            => throw new NotSupportedException($"The {nameof(MakeSessionOptionWithCudaProvider)} method is not supported in this build");

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(int deviceId = 0)
            => throw new NotSupportedException($"The {nameof(MakeSessionOptionWithTensorrtProvider)} method is not supported in this build");

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution provider.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trtProviderOptions">TensorRT EP provider options</param>
        /// <returns>A SessionsOptions() object configured for execution on provider options</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(OrtTensorRTProviderOptions trtProviderOptions)
            => throw new NotSupportedException($"The {nameof(MakeSessionOptionWithTensorrtProvider)} method is not supported in this build");

        /// <summary>
        /// A helper method to construct a SessionOptions object for Nuphar execution.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">settings string, comprises of comma separated key:value pairs. default is empty</param>
        /// <returns>A SessionsOptions() object configured for execution with Nuphar</returns>
        public static SessionOptions MakeSessionOptionWithNupharProvider(String settings = "")
            => throw new NotSupportedException($"The {nameof(MakeSessionOptionWithNupharProvider)} method is not supported in this build");

        /// <summary>
        /// A helper method to construct a SessionOptions object for ROCM execution.
        /// Use only if ROCM is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">Device Id</param>
        /// <param name="gpuMemLimit">GPU memory limit. Defaults to no limit.</param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithRocmProvider(int deviceId = 0, UIntPtr gpuMemLimit = default)
            => throw new NotSupportedException($"The {nameof(MakeSessionOptionWithRocmProvider)} method is not supported in this build");

        /// <summary>
        /// (Deprecated) Loads a DLL named 'libraryPath' and looks for this entry point:
        /// OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
        /// It then passes in the provided session options to this function along with the api base.
        /// Deprecated in favor of RegisterCustomOpLibraryV2() because it provides users with the library handle 
        /// to release when all sessions relying on it are destroyed
        /// </summary>
        /// <param name="libraryPath">path to the custom op library</param>
        [ObsoleteAttribute("RegisterCustomOpLibrary(...) is obsolete. Use RegisterCustomOpLibraryV2(...) instead.", false)]
        public void RegisterCustomOpLibrary(string libraryPath)
            => throw new NotSupportedException($"The {nameof(RegisterCustomOpLibrary)} method is not supported in this build");

        /// <summary>
        /// Loads a DLL named 'libraryPath' and looks for this entry point:
        /// OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
        /// It then passes in the provided session options to this function along with the api base.
        /// The handle to the loaded library is returned in 'libraryHandle'.
        /// It can be unloaded by the caller after all sessions using the passed in
        /// session options are destroyed, or if an error occurs and it is non null.
        /// Hint: .NET Core 3.1 has a 'NativeLibrary' class that can be used to free the library handle
        /// </summary>
        /// <param name="libraryPath">Custom op library path</param>
        /// <param name="libraryHandle">out parameter, library handle</param>
        public void RegisterCustomOpLibraryV2(string libraryPath, out IntPtr libraryHandle)
            => throw new NotSupportedException($"The {nameof(RegisterCustomOpLibraryV2)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="useArena">1 - use allocation arena, 0 - otherwise</param>
        public void AppendExecutionProvider_Dnnl(int useArena = 1)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_Dnnl)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">integer device ID</param>
        public void AppendExecutionProvider_CUDA(int deviceId = 0)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_CUDA)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_DML(int deviceId = 0)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_DML)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification, default empty string</param>
        public void AppendExecutionProvider_OpenVINO(string deviceId = "")
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_OpenVINO)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_Tensorrt(int deviceId = 0)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_Tensorrt)} method is not supported in this build");

        /// <summary>
        /// Append a TensorRT EP instance (based on specified configuration) to the SessionOptions instance.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trtProviderOptions">TensorRT EP provider options</param>
        public void AppendExecutionProvider_Tensorrt(OrtTensorRTProviderOptions trtProviderOptions)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_Tensorrt)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">Device Id</param>
        /// <param name="gpuMemLimit">GPU memory limit. Defaults to no limit.</param>
        public void AppendExecutionProvider_ROCM(int deviceId = 0, UIntPtr gpuMemLimit = default)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_ROCM)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_MIGraphX(int deviceId = 0)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_MIGraphX)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">string with Nuphar specific settings</param>
        public void AppendExecutionProvider_Nuphar(string settings = "")
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_Nuphar)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="coreml_flags">CoreML specific flags</param>
        public void AppendExecutionProvider_CoreML(COREMLFlags coreml_flags = COREMLFlags.COREML_FLAG_USE_NONE)
             => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_CoreML)} method is not supported in this build");

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="coreml_flags">CoreML specific flag mask</param>
        public void AppendExecutionProvider_CoreML(uint coreml_flags = 0)
            => throw new NotSupportedException($"The {nameof(AppendExecutionProvider_CoreML)} method is not supported in this build");
    }
}