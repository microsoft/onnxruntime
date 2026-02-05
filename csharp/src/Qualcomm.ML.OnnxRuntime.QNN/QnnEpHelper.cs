// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Qualcomm.ML.OnnxRuntime.QNN
{
    /// <summary>
    /// Helper class for QNN Execution Provider
    /// Provides static methods to query EP information
    /// </summary>
    public static class QnnEpHelper
    {
        private const string EpName = "QnnExecutionProvider";
        private static readonly string[] EpNames = { EpName };
        
        // Cache platform-specific values to avoid repeated checks
        private static readonly Lazy<string> CachedRuntimeIdentifier = new Lazy<string>(ComputeRuntimeIdentifier);
        private static readonly Lazy<bool> IsWindows = new Lazy<bool>(() => RuntimeInformation.IsOSPlatform(OSPlatform.Windows));
        private static readonly Lazy<bool> IsLinux = new Lazy<bool>(() => RuntimeInformation.IsOSPlatform(OSPlatform.Linux));
        private static readonly Lazy<string> AssemblyDirectory = new Lazy<string>(() => 
            Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location));

        /// <summary>
        /// Get the QNN execution provider name
        /// </summary>
        /// <returns>The EP name ("QnnExecutionProvider")</returns>
        public static string GetEpName() => EpName;

        /// <summary>
        /// Get all supported EP names
        /// </summary>
        /// <returns>Array of supported EP names</returns>
        public static string[] GetEpNames() => EpNames;

        /// <summary>
        /// Get the full path to the QNN EP library (onnxruntime_providers_qnn.dll or libonnxruntime_providers_qnn.so)
        /// </summary>
        /// <returns>Full path to the EP library</returns>
        /// <exception cref="FileNotFoundException">Thrown when EP library file is not found</exception>
        /// <exception cref="PlatformNotSupportedException">Thrown when platform is not supported</exception>
        public static string GetLibraryPath() => GetLibraryPathInternal(GetEpLibraryName(), "QNN EP");

        /// <summary>
        /// Get the full path to the QNN HTP library (QnnHtp.dll or libQnnHtp.so)
        /// </summary>
        /// <returns>Full path to the QNN HTP library</returns>
        /// <exception cref="FileNotFoundException">Thrown when QNN HTP library file is not found</exception>
        /// <exception cref="PlatformNotSupportedException">Thrown when platform is not supported</exception>
        public static string GetQnnHtpLibraryPath() => GetLibraryPathInternal(GetQnnHtpLibraryName(), "QNN HTP");

        /// <summary>
        /// Get the full path to the QNN CPU library (QnnCpu.dll or libQnnCpu.so)
        /// </summary>
        /// <returns>Full path to the QNN CPU library</returns>
        /// <exception cref="FileNotFoundException">Thrown when QNN CPU library file is not found</exception>
        /// <exception cref="PlatformNotSupportedException">Thrown when platform is not supported</exception>
        public static string GetQnnCpuLibraryPath() => GetLibraryPathInternal(GetQnnCpuLibraryName(), "QNN CPU");

        private static string GetLibraryPathInternal(string libraryName, string libraryDescription)
        {
            string epLibraryDirectory = GetEpLibraryDirectory(libraryName);
            string fullPath = Path.Combine(epLibraryDirectory, libraryName);
            
            if (!File.Exists(fullPath))
            {
                throw new FileNotFoundException(
                    $"{libraryDescription} library not found: {fullPath}. " +
                    $"Ensure the Qualcomm.ML.OnnxRuntime.QNN NuGet package is properly installed.",
                    fullPath);
            }
            
            return fullPath;
        }

        private static string GetEpLibraryName() => GetPlatformLibraryName("onnxruntime_providers_qnn");

        private static string GetQnnHtpLibraryName() => GetPlatformLibraryName("QnnHtp");

        private static string GetQnnCpuLibraryName() => GetPlatformLibraryName("QnnCpu");

        private static string GetPlatformLibraryName(string baseName)
        {
            if (IsWindows.Value)
            {
                return $"{baseName}.dll";
            }
            
            if (IsLinux.Value)
            {
                return $"lib{baseName}.so";
            }
            
            throw new PlatformNotSupportedException(
                $"Platform {RuntimeInformation.OSDescription} is not supported");
        }

        private static string GetEpLibraryDirectory(string epLibName)
        {
            string assemblyDir = AssemblyDirectory.Value;
            
            // Try current directory first
            if (File.Exists(Path.Combine(assemblyDir, epLibName)))
            {
                return assemblyDir;
            }
            
            // Try runtimes folder structure
            string rid = CachedRuntimeIdentifier.Value;
            string runtimesPath = Path.Combine(assemblyDir, "runtimes", rid, "native");
            if (File.Exists(Path.Combine(runtimesPath, epLibName)))
            {
                return runtimesPath;
            }
            
            // Try parent directory
            string parentDir = Directory.GetParent(assemblyDir)?.FullName;
            if (parentDir != null && File.Exists(Path.Combine(parentDir, epLibName)))
            {
                return parentDir;
            }
            
            // Try additional search paths
            string[] relativePaths = new[]
            {
                "native",
                Path.Combine("..", "native"),
                Path.Combine("..", "..", "native")
            };

            foreach (var relativePath in relativePaths)
            {
                string searchPath = Path.Combine(assemblyDir, relativePath);
                if (TryGetFullPath(searchPath, out string normalizedPath))
                {
                    string candidatePath = Path.Combine(normalizedPath, epLibName);
                    if (File.Exists(candidatePath))
                    {
                        return normalizedPath;
                    }
                }
            }
            
            // Fallback to assembly directory
            return assemblyDir;
        }

        private static bool TryGetFullPath(string path, out string fullPath)
        {
            try
            {
                fullPath = Path.GetFullPath(path);
                return Directory.Exists(fullPath);
            }
            catch (ArgumentException)
            {
                fullPath = null;
                return false;
            }
            catch (NotSupportedException)
            {
                fullPath = null;
                return false;
            }
            catch (PathTooLongException)
            {
                fullPath = null;
                return false;
            }
        }

        private static string ComputeRuntimeIdentifier()
        {
            if (IsWindows.Value)
            {
                return RuntimeInformation.ProcessArchitecture switch
                {
                    Architecture.X64 => "win-x64",
                    Architecture.Arm64 => "win-arm64",
                    Architecture.X86 => "win-x86",
                    _ => "win-x64" // Default fallback
                };
            }
            
            if (IsLinux.Value)
            {
                return RuntimeInformation.ProcessArchitecture switch
                {
                    Architecture.X64 => "linux-x64",
                    Architecture.Arm64 => "linux-arm64",
                    _ => "linux-x64" // Default fallback
                };
            }
            
            throw new PlatformNotSupportedException(
                $"Platform {RuntimeInformation.OSDescription} is not supported");
        }
    }
}
