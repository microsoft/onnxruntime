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

            // Probe the standard NuGet runtimes/<rid>/native/ layout first, then fall back
            // to the base directory for single-file/published layouts where native assets
            // can land directly next to the managed assembly.
            string[] candidates =
            {
                Path.Combine(rootDir, "runtimes", rid, "native", libraryName),
                Path.Combine(rootDir, libraryName),
            };

            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                    return Path.GetFullPath(candidate);
            }

            throw new FileNotFoundException(
                $"Did not find WebGPU EP library file. Probed: {string.Join(", ", candidates)}");
        }

        /// <summary>
        /// Returns the names of the EPs created by the WebGPU plugin EP library.
        /// Can be used to select an <c>OrtEpDevice</c> from those returned by <c>OrtEnv.GetEpDevices()</c>.
        /// </summary>
        /// <returns>Array of EP names.</returns>
        public static string[] GetEpNames()
        {
            return new[] { GetEpName() };
        }

        /// <summary>
        /// Returns the name of the one EP supported by this plugin EP library.
        /// Convenience method for plugin EP packages that expose a single EP.
        /// </summary>
        /// <returns>The EP name string.</returns>
        public static string GetEpName()
        {
            return "WebGpuExecutionProvider";
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

            throw new PlatformNotSupportedException(
                $"WebGPU plugin EP does not support OS platform: {RuntimeInformation.OSDescription}");
        }

        private static string GetOSTag()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return "win";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return "linux";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return "osx";
            throw new PlatformNotSupportedException(
                $"WebGPU plugin EP does not support OS platform: {RuntimeInformation.OSDescription}");
        }

        private static string GetArchTag()
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X64 => "x64",
                Architecture.Arm64 => "arm64",
                _ => throw new PlatformNotSupportedException(
                    $"WebGPU plugin EP does not support process architecture: {RuntimeInformation.ProcessArchitecture}"),
            };
        }
    }
}
