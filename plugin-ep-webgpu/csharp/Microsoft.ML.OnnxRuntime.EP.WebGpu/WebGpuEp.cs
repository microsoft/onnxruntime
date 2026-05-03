using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime.EP.WebGpu
{
    /// <summary>
    /// Provides helper methods to locate the WebGPU plugin EP native library
    /// and retrieve the EP name for registration with ONNX Runtime.
    /// </summary>
    public static class WebGpuEp
    {
        /// <summary>
        /// Returns the path to the WebGPU plugin EP native library contained by this package.
        /// Can be passed to <c>OrtEnv.RegisterExecutionProviderLibrary()</c>.
        /// </summary>
        /// <returns>Full path to the EP native library.</returns>
        /// <exception cref="FileNotFoundException">If the native library file does not exist at the expected path.</exception>
        public static string GetLibraryPath()
        {
            string rootDir = GetNativeDirectory();
            string rid = GetRuntimeIdentifier();
            string libraryName = GetLibraryName();
            string epLibPath = Path.GetFullPath(Path.Combine(rootDir,
                                                             "runtimes", rid,
                                                             "native",
                                                             libraryName));

            if (!File.Exists(epLibPath))
            {
                throw new FileNotFoundException(
                    $"Did not find WebGPU EP library file: {epLibPath}");
            }

            return epLibPath;
        }

        /// <summary>
        /// Returns the names of the EPs created by the WebGPU plugin EP library.
        /// Can be used to select an <c>OrtEpDevice</c> from those returned by <c>OrtEnv.GetEpDevices()</c>.
        /// </summary>
        /// <returns>Array of EP names.</returns>
        public static string[] GetEpNames()
        {
            return new[] { "WebGpuExecutionProvider" };
        }

        /// <summary>
        /// Returns the name of the one EP supported by this plugin EP library.
        /// Convenience method for plugin EP packages that expose a single EP.
        /// </summary>
        /// <returns>The EP name string.</returns>
        public static string GetEpName()
        {
            return GetEpNames()[0];
        }

        private static string GetNativeDirectory()
        {
            var assemblyDir = Path.GetDirectoryName(typeof(WebGpuEp).Assembly.Location);

            if (!string.IsNullOrEmpty(assemblyDir) && Directory.Exists(assemblyDir))
                return assemblyDir;

            return AppContext.BaseDirectory;
        }

        private static string GetRuntimeIdentifier()
        {
            return GetOSTag() + "-" + GetArchTag();
        }

        private static string GetLibraryName()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return "onnxruntime_providers_webgpu.dll";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return "libonnxruntime_providers_webgpu.so";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return "libonnxruntime_providers_webgpu.dylib";

            return "onnxruntime_providers_webgpu";
        }

        private static string GetOSTag()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return "win";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return "linux";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return "osx";
            return "unknown";
        }

        private static string GetArchTag()
        {
            return RuntimeInformation.OSArchitecture == Architecture.X64 ? "x64"
                 : RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "arm64"
                 : "unknown";
        }
    }
}
