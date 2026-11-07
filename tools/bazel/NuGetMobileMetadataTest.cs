// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;

internal static class NuGetMobileMetadataTest
{
    private const BindingFlags StaticNonPublic = BindingFlags.Static | BindingFlags.NonPublic;
    private const BindingFlags StaticAll = StaticNonPublic | BindingFlags.Public;

    private static void Assert(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static string GetAssemblyAttribute(Assembly assembly, string attributeName)
    {
        var attribute = assembly.CustomAttributes.SingleOrDefault(
            item => item.AttributeType.FullName == attributeName);
        Assert(attribute != null, $"{assembly.Location} is missing {attributeName}.");
        return (string)attribute!.ConstructorArguments[0].Value!;
    }

    private static void Validate(
        string path,
        string expectedPlatform,
        string expectedNativeLibrary,
        bool expectNnapi,
        bool expectCoreMl)
    {
        var context = new AssemblyLoadContext(expectedPlatform, isCollectible: true);
        try
        {
            var assembly = context.LoadFromAssemblyPath(Path.GetFullPath(path));
            Assert(
                GetAssemblyAttribute(assembly, "System.Runtime.Versioning.TargetFrameworkAttribute") ==
                    ".NETCoreApp,Version=v9.0",
                $"{path} must identify the .NET 9 target framework.");
            Assert(
                GetAssemblyAttribute(assembly, "System.Runtime.Versioning.TargetPlatformAttribute") ==
                    expectedPlatform,
                $"{path} must identify target platform {expectedPlatform}.");

            var nativeMethods = assembly.GetType("Microsoft.ML.OnnxRuntime.NativeMethods", throwOnError: true)!;
            var nativeLibrary = nativeMethods.GetNestedType("NativeLib", BindingFlags.NonPublic)!;
            var dllName = (string)nativeLibrary.GetField("DllName", StaticNonPublic)!.GetRawConstantValue()!;
            Assert(dllName == expectedNativeLibrary, $"{path} uses native library {dllName}, not {expectedNativeLibrary}.");

            Assert(
                nativeMethods.GetMethod("OrtSessionOptionsAppendExecutionProvider_Dnnl", StaticAll) == null,
                $"{path} contains a desktop-only DNNL entry point; __MOBILE__ was not applied.");
            Assert(
                (nativeMethods.GetMethod("OrtSessionOptionsAppendExecutionProvider_Nnapi", StaticAll) != null) ==
                    expectNnapi,
                $"{path} has an incorrect Android NNAPI surface; __ANDROID__ was not applied correctly.");
            Assert(
                (nativeMethods.GetMethod("OrtSessionOptionsAppendExecutionProvider_CoreML", StaticAll) != null) ==
                    expectCoreMl,
                $"{path} has an incorrect CoreML surface; __IOS__/__ENABLE_COREML__ were not applied correctly.");
        }
        finally
        {
            context.Unload();
        }
    }

    public static void Main(string[] args)
    {
        Assert(args.Length == 3, "Expected Android, iOS, and Mac Catalyst managed assembly paths.");
        var hashes = new HashSet<string>(
            args.Select(path => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)))),
            StringComparer.Ordinal);
        Assert(hashes.Count == args.Length, "Mobile managed assemblies must be distinct compiler outputs.");

        Validate(args[0], "Android35.0", "libonnxruntime.so", expectNnapi: true, expectCoreMl: false);
        Validate(args[1], "iOS18.0", "__Internal", expectNnapi: false, expectCoreMl: true);
        Validate(args[2], "MacCatalyst18.0", "__Internal", expectNnapi: false, expectCoreMl: true);
        Console.WriteLine("Distinct Android, iOS, and Mac Catalyst managed assemblies have correct platform metadata.");
    }
}
