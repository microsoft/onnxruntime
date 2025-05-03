// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// not supported on mobile platforms
#if !(ANDROID || IOS)

namespace Microsoft.ML.OnnxRuntime.Tests;

using System;
using System.Globalization;
using System.Runtime.InteropServices;
using Xunit;


public class CompileApiTests
{
    private OrtEnv ortEnvInstance = OrtEnv.Instance();


    [Fact]
    public void BasicUsage()
    {
        var so = new SessionOptions();
        using (var compileOptions = new OrtModelCompilationOptions(so))
        {
            // mainly checking these don't throw which ensures all the plumbing for the binding works.
            compileOptions.SetInputModelPath("model.onnx");
            compileOptions.SetOutputModelPath("compiled_model.onnx");

            compileOptions.SetOutputModelExternalInitializersFile("external_data.bin", 512);
            compileOptions.SetEpContextEmbedMode(true);

        }

        // setup a new instance as SetOutputModelExternalInitializersFile is incompatible with SetOutputModelBuffer
        using (var compileOptions = new OrtModelCompilationOptions(so))
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
            compileOptions.SetInputModelFromBuffer(model);

            // SetOutputModelBuffer updates the user provided IntPtr and size when it allocates data post-compile.
            // Due to that we need to allocate an IntPtr and UIntPtr here.
            IntPtr bytePtr = new IntPtr();
            UIntPtr bytesSize = new UIntPtr();
            var allocator = OrtAllocator.DefaultInstance;
            compileOptions.SetOutputModelBuffer(allocator, ref bytePtr, ref bytesSize);

            compileOptions.CompileModel();

            Assert.NotEqual(IntPtr.Zero, bytePtr);
            Assert.NotEqual(UIntPtr.Zero, bytesSize);

            byte[] compiledBytes = new byte[bytesSize.ToUInt64()];
            Marshal.Copy(bytePtr, compiledBytes, 0, (int)bytesSize.ToUInt32());

            // Check the compiled model is valid
            using (var session = new InferenceSession(compiledBytes, so))
            {
                Assert.NotNull(session);
            }

            allocator.FreeMemory(bytePtr);
        }
    }
}

#endif