﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// not supported on mobile platforms
#if !(ANDROID || IOS)

namespace Microsoft.ML.OnnxRuntime.Tests;

using System;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using Xunit;


public class CompileApiTests
{
    private OrtEnv ortEnvInstance = OrtEnv.Instance();


    [Fact]
    public void BasicUsage()
    {
        using (var sessionOptions = new SessionOptions())
        {
            using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
            {
                // mainly checking these don't throw which ensures all the plumbing for the binding works.
                compileOptions.SetInputModelPath("model.onnx");
                compileOptions.SetOutputModelPath("compiled_model.onnx");

                compileOptions.SetOutputModelExternalInitializersFile("external_data.bin", 512);
                compileOptions.SetEpContextEmbedMode(true);
                compileOptions.SetGraphOptimizationLevel(GraphOptimizationLevel.ORT_ENABLE_BASIC);

            }

            // setup a new instance as SetOutputModelExternalInitializersFile is incompatible with SetOutputModelBuffer
            using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
            {
                var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
                compileOptions.SetInputModelFromBuffer(model);

                // SetOutputModelBuffer updates the user provided IntPtr and size when it allocates data post-compile.
                // Due to that we need to allocate an IntPtr and UIntPtr here.
                IntPtr bytePtr = new IntPtr();
                UIntPtr bytesSize = new UIntPtr();
                var allocator = OrtAllocator.DefaultInstance;
                compileOptions.SetOutputModelBuffer(allocator, ref bytePtr, ref bytesSize);
                compileOptions.SetEpContextBinaryInformation("./", "squeezenet.onnx");

                compileOptions.CompileModel();

                Assert.NotEqual(IntPtr.Zero, bytePtr);
                Assert.NotEqual(UIntPtr.Zero, bytesSize);

                byte[] compiledBytes = new byte[bytesSize.ToUInt64()];
                Marshal.Copy(bytePtr, compiledBytes, 0, (int)bytesSize.ToUInt32());

                // Check the compiled model is valid
                using (var session = new InferenceSession(compiledBytes, sessionOptions))
                {
                    Assert.NotNull(session);
                }

                allocator.FreeMemory(bytePtr);
            }

            // Test using OrtCompileApiFlags.ERROR_NO_NODES_COMPILED. A model compiled with CPU EP will not generate
            // any compiled EPContext nodes, so expect an ORT_FAIL error.
            using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
            {
                var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
                var output_model_file = "should_not_generate.onnx";
                compileOptions.SetInputModelFromBuffer(model);
                compileOptions.SetOutputModelPath(output_model_file);
                compileOptions.SetFlags(OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED);

                // compile should fail
                try
                {
                    compileOptions.CompileModel();
                    Assert.Fail("CompileModel() should have thrown an exception");
                }
                catch (OnnxRuntimeException ex)
                {
                    Assert.Contains("Unable to compile any nodes", ex.Message);
                }

                Assert.False(File.Exists(output_model_file));  // Output file should not be generated.
            }

            // Test using OrtCompileApiFlags.ERROR_IF_OUTPUT_FILE_EXISTS.
            var outputModelFile = "squeezenet_ctx.onnx";
            try
            {
                using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
                {
                    var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

                    // Compile and generate an output model.
                    compileOptions.SetInputModelFromBuffer(model);
                    compileOptions.SetOutputModelPath(outputModelFile);
                    compileOptions.CompileModel();
                    Assert.True(File.Exists(outputModelFile));

                    // Try to compile again with flag that prevents replacing an existing file.
                    // Expect failure.
                    compileOptions.SetFlags(OrtCompileApiFlags.ERROR_IF_OUTPUT_FILE_EXISTS);

                    // compile should fail
                    try
                    {
                        compileOptions.CompileModel();
                        Assert.Fail("CompileModel() should have thrown an exception");
                    }
                    catch (OnnxRuntimeException ex)
                    {
                        Assert.Contains("exists already", ex.Message);
                    }
                }
            }
            finally
            {
                if (File.Exists(outputModelFile))
                {
                    // This file is created by ORT, so we delete it manually in finally block.
                    File.Delete(outputModelFile);
                }
            }
        }
    }

    [Fact]
    public void WriteOutModelWithDelegate()
    {
        var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
        var outputModelFilePath = "squeezenet_write_delegate_ctx.onnx";

        using (FileStream fs = new FileStream(outputModelFilePath, FileMode.Create, FileAccess.Write, FileShare.None,
            4096, FileOptions.DeleteOnClose))
        using (var sessionOptions = new SessionOptions())
        using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
        {
            void BasicWriteBufferDelegate(ReadOnlySpan<byte> buffer)
            {
                Assert.True(buffer.Length > 0);
                fs.Write(buffer.ToArray(), 0, buffer.Length);  // Write it out to a file
            }

            // Compile and generate an output model.
            compileOptions.SetInputModelFromBuffer(model);
            compileOptions.SetOutputModelWriteDelegate(BasicWriteBufferDelegate);
            compileOptions.CompileModel();
            Assert.True(File.Exists(outputModelFilePath));
        }
    }

    [Fact]
    public void BasicGetInitializerLocationDelegate()
    {
        var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
        var outputModelFilePath = "squeezenet_handle_initializer_delegate_ctx.onnx";
        var initializersFilePath = "squeezenet_handle_initializer_delegate_ctx.bin";

        try
        {
            using (FileStream fs = new FileStream(initializersFilePath, FileMode.Create, FileAccess.Write,
                FileShare.None, 4096, FileOptions.DeleteOnClose))
            using (var sessionOptions = new SessionOptions())
            using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
            {
                // Custom delegate that stores large initializers in a new file.
                OrtExternalInitializerInfo BasicHandleInitializer(
                    string initializerName, IReadOnlyOrtValue initializerValue,
                    IReadOnlyExternalInitializerInfo originalInitializerLocation)
                {
                    Assert.True(initializerName.Length > 0);

                    var byteSize = initializerValue.GetTensorSizeInBytes();
                    if (byteSize <= 64)
                    {
                        // Keep small initializers stored within model.
                        return null;
                    }

                    long byteOffset = fs.Position;
                    ReadOnlySpan<byte> dataSpan = initializerValue.GetTensorDataAsSpan<byte>();
                    fs.Write(dataSpan.ToArray(), 0, dataSpan.Length);  // Write it out to a file

                    // Return the data's new location.
                    return new OrtExternalInitializerInfo(initializersFilePath, byteOffset, byteSize);
                }

                // Compile and generate an output model.
                compileOptions.SetInputModelFromBuffer(model);
                compileOptions.SetOutputModelPath(outputModelFilePath);
                compileOptions.SetOutputModelGetInitializerLocationDelegate(BasicHandleInitializer);
                compileOptions.CompileModel();
                Assert.True(File.Exists(outputModelFilePath));
            }
        }
        finally
        {
            if (File.Exists(outputModelFilePath))
            {
                // This file is created by ORT, so we delete it manually in finally block.
                File.Delete(outputModelFilePath);
            }
        }
    }

    [Fact]
    public void GetInitializerLocationDelegateThatReusesExternalInitializers()
    {
        var model = TestDataLoader.LoadModelFromEmbeddedResource("conv_qdq_external_ini.onnx");
        var outputModelFilePath = "conv_qdq_external_ini.reuse.ctx.onnx";
        bool reusedExternalInitializers = false;

        try
        {
            using (var sessionOptions = new SessionOptions())
            using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
            {
                // Custom delegate that reuses the original external initializer file.
                OrtExternalInitializerInfo ReuseExternalInitializers(
                    string initializerName, IReadOnlyOrtValue initializerValue,
                    IReadOnlyExternalInitializerInfo originalInitializerLocation)
                {
                    Assert.True(initializerName.Length > 0);

                    if (originalInitializerLocation != null)
                    {
                        reusedExternalInitializers = true;  // For test assertion only
                        string originalFilePath = originalInitializerLocation.GetFilePath();
                        long originalFileOffset = originalInitializerLocation.GetFileOffset();
                        long originalByteSize = originalInitializerLocation.GetByteSize();

                        Assert.True(originalFilePath.Length > 0);
                        Assert.True(originalFileOffset >= 0);
                        Assert.True(originalByteSize > 0);

                        // This initializer comes from an external file. Reuse it for compiled model.
                        return new OrtExternalInitializerInfo(originalFilePath, originalFileOffset, originalByteSize);
                    }

                    // Otherwise, embed initializers that were not originally external.
                    return null;
                }

                // Compile and generate an output model.
                compileOptions.SetInputModelFromBuffer(model);
                compileOptions.SetOutputModelPath(outputModelFilePath);
                compileOptions.SetOutputModelGetInitializerLocationDelegate(ReuseExternalInitializers);
                compileOptions.CompileModel();

                Assert.True(File.Exists(outputModelFilePath));
                Assert.True(reusedExternalInitializers);
            }
        }
        finally
        {
            if (File.Exists(outputModelFilePath))
            {
                // This file is created by ORT, so we delete it manually in finally block.
                File.Delete(outputModelFilePath);
            }
        }
    }
}

#endif