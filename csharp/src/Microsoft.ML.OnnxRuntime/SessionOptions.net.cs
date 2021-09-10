// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents the platform-specific options for creating an InferenceSession.
    /// </summary>
    /// <see cref="https://onnxruntime.ai"/>
    /// <remarks>Until .net6.0, all options will continue to be available or use in netcoreapp targets including those not supported across platforms.</remarks>
    public partial class SessionOptions
    {
        // Delay-loaded CUDA or cuDNN DLLs. Currently, delayload is disabled. See cmake/CMakeLists.txt for more information.
        private static string[] cudaDelayLoadedLibs = { };
        private static string[] trtDelayLoadedLibs = { };

        #region NetCoreApp Constructor and Factory methods

        /// <summary>
        /// A helper method to construct a SessionOptions object for CUDA execution.
        /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider(int deviceId = 0)
        {
            CheckCudaExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options.Handle, deviceId));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
            return options;
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(int deviceId = 0)
        {
            CheckTensorrtExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Tensorrt(options.Handle, deviceId));
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options.Handle, deviceId));
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
                return options;
            }
            catch (Exception e)
            {
                options.Dispose();
                throw e;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution provider.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trtProviderOptions">TensorRT EP provider options</param>
        /// <returns>A SessionsOptions() object configured for execution on provider options</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(OrtTensorRTProviderOptions trtProviderOptions)
        {
            CheckTensorrtExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            try
            {
                // Make sure that CUDA EP uses the same device id as TensorRT EP.
                int deviceId = trtProviderOptions.GetDeviceId();

                NativeApiStatus.VerifySuccess(NativeMethods.SessionOptionsAppendExecutionProvider_TensorRT(options.Handle, trtProviderOptions.Handle));
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options.Handle, deviceId));
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
                return options;
            }
            catch (Exception e)
            {
                options.Dispose();
                throw e;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for Nuphar execution.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">settings string, comprises of comma separated key:value pairs. default is empty</param>
        /// <returns>A SessionsOptions() object configured for execution with Nuphar</returns>
        public static SessionOptions MakeSessionOptionWithNupharProvider(String settings = "")
        {
            SessionOptions options = new SessionOptions();

            var settingsPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(settings), GCHandleType.Pinned);
            using (var pinnedSettingsName = new PinnedGCHandle(settingsPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nuphar(options.Handle, 1, pinnedSettingsName.Pointer));
            }

            return options;
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for ROCM execution.
        /// Use only if ROCM is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">Device Id</param>
        /// <param name="gpuMemLimit">GPU memory limit. Defaults to no limit.</param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithRocmProvider(int deviceId = 0, UIntPtr gpuMemLimit = default)
        {
            SessionOptions options = new SessionOptions();
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtSessionOptionsAppendExecutionProvider_ROCM(options.Handle, deviceId, gpuMemLimit));
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options.Handle, 1));
            return options;
        }

        #endregion

        #region Public Methods

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
        {
            IntPtr libraryHandle = IntPtr.Zero;
            var libraryPathPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(libraryPath), GCHandleType.Pinned);
            using (var pinnedlibraryPath = new PinnedGCHandle(libraryPathPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRegisterCustomOpsLibrary(handle, pinnedlibraryPath.Pointer, out libraryHandle));
            }
        }

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
        {
            var libraryPathPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(libraryPath), GCHandleType.Pinned);
            using (var pinnedlibraryPath = new PinnedGCHandle(libraryPathPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRegisterCustomOpsLibrary(handle, pinnedlibraryPath.Pointer, out libraryHandle));
            }
        }

        #endregion


        #region Platform-specific ExecutionProviderAppends

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="useArena">1 - use allocation arena, 0 - otherwise</param>
        public void AppendExecutionProvider_Dnnl(int useArena = 1)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Dnnl(handle, useArena));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">integer device ID</param>
        public void AppendExecutionProvider_CUDA(int deviceId = 0)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(handle, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_DML(int deviceId = 0)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_DML(handle, deviceId));
        }


        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification, default empty string</param>
        public void AppendExecutionProvider_OpenVINO(string deviceId = "")
        {
            var deviceIdPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(deviceId), GCHandleType.Pinned);
            using (var pinnedDeviceIdName = new PinnedGCHandle(deviceIdPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_OpenVINO(handle, pinnedDeviceIdName.Pointer));
            }
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_Tensorrt(int deviceId = 0)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Tensorrt(handle, deviceId));
        }

        /// <summary>
        /// Append a TensorRT EP instance (based on specified configuration) to the SessionOptions instance.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trtProviderOptions">TensorRT EP provider options</param>
        public void AppendExecutionProvider_Tensorrt(OrtTensorRTProviderOptions trtProviderOptions)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.SessionOptionsAppendExecutionProvider_TensorRT(handle, trtProviderOptions.Handle));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">Device Id</param>
        /// <param name="gpuMemLimit">GPU memory limit. Defaults to no limit.</param>
        public void AppendExecutionProvider_ROCM(int deviceId = 0, UIntPtr gpuMemLimit = default)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtSessionOptionsAppendExecutionProvider_ROCM(handle, deviceId, gpuMemLimit));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_MIGraphX(int deviceId = 0)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_MIGraphX(handle, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">string with Nuphar specific settings</param>
        public void AppendExecutionProvider_Nuphar(string settings = "")
        {
            var settingsPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(settings), GCHandleType.Pinned);
            using (var pinnedSettingsName = new PinnedGCHandle(settingsPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nuphar(handle, 1, pinnedSettingsName.Pointer));
            }
        }

        #endregion

        #region Private Methods

        // Declared, but called only if OS = Windows.
        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport("kernel32.dll")]
        static extern uint GetSystemDirectory([Out] StringBuilder lpBuffer, uint uSize);
        private static bool CheckCudaExecutionProviderDLLs()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var dll in cudaDelayLoadedLibs)
                {
                    IntPtr handle = LoadLibrary(dll);
                    if (handle != IntPtr.Zero)
                        continue;
                    var sysdir = new StringBuilder(String.Empty, 2048);
                    GetSystemDirectory(sysdir, (uint)sysdir.Capacity);
                    throw new OnnxRuntimeException(
                        ErrorCode.NoSuchFile,
                        $"kernel32.LoadLibrary():'{dll}' not found. CUDA is required for GPU execution. " +
                        $". Verify it is available in the system directory={sysdir}. Else copy it to the output folder."
                        );
                }
            }
            return true;
        }

        private static bool CheckTensorrtExecutionProviderDLLs()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var dll in trtDelayLoadedLibs)
                {
                    IntPtr handle = LoadLibrary(dll);
                    if (handle != IntPtr.Zero)
                        continue;
                    var sysdir = new StringBuilder(String.Empty, 2048);
                    GetSystemDirectory(sysdir, (uint)sysdir.Capacity);
                    throw new OnnxRuntimeException(
                        ErrorCode.NoSuchFile,
                        $"kernel32.LoadLibrary():'{dll}' not found. TensorRT/CUDA are required for GPU execution. " +
                        $". Verify it is available in the system directory={sysdir}. Else copy it to the output folder."
                        );
                }
            }
            return true;
        }
        #endregion
    }
}