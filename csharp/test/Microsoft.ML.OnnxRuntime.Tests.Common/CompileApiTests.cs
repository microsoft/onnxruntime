// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// not supported on mobile platforms
#if !(ANDROID || IOS)

namespace Microsoft.ML.OnnxRuntime.Tests;

using System;
using System.Globalization;
using System.IO;
using System.Linq;
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
    public void EpContextDataCallbackValidation()
    {
        Assert.Equal(IntPtr.Size * 426, Marshal.SizeOf<OrtApi>());
        Assert.Equal(IntPtr.Size * 17, Marshal.SizeOf<CompileApi.OrtCompileApi>());

        using var sessionOptions = new SessionOptions();
        Assert.Throws<ArgumentNullException>(() => sessionOptions.SetEpContextDataReadDelegate(null, 1024));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            sessionOptions.SetEpContextDataReadDelegate((_, _) => { }, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            sessionOptions.SetEpContextDataReadDelegate((_, _) => { }, ulong.MaxValue));
        if (UIntPtr.Size == 4)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                sessionOptions.SetEpContextDataReadDelegate((_, _) => { }, uint.MaxValue));
        }

        sessionOptions.SetEpContextDataReadDelegate((_, output) => output.Allocate(0), 1024);
        using var compileOptions = new OrtModelCompilationOptions(sessionOptions);
        sessionOptions.ClearEpContextDataReadDelegate();
        sessionOptions.Dispose();

        Assert.Throws<ArgumentNullException>(() => compileOptions.SetEpContextDataWriteDelegate(null));
        compileOptions.SetEpContextDataWriteDelegate((_, _) => { });
        compileOptions.ClearEpContextDataWriteDelegate();
    }

    [SkippableFact]
    public void ExternalEpContextDataUsesCallbacks()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Windows),
            "The example plugin EP integration artifact is available only in the Windows C# test build.");

        string libraryPath = Path.Combine(Directory.GetCurrentDirectory(), "example_plugin_ep.dll");
        Assert.True(File.Exists(libraryPath), $"Expected library {libraryPath} does not exist.");

        const string epName = "csharp_ep_context";
        const string modelName = "csharp_ep_context.onnx";
        byte[] inputModel = TestDataLoader.LoadModelFromEmbeddedResource("mul_1.onnx");
        byte[] contextPayload = null;
        string contextName = null;
        OrtEpContextData retainedWriteData = null;
        OrtEpContextDataBuffer retainedReadBuffer = null;
        int writeCount = 0;
        int readCount = 0;

        ortEnvInstance.RegisterExecutionProviderLibrary(epName, libraryPath);
        try
        {
            OrtEpDevice epDevice = ortEnvInstance.GetEpDevices().Single(device => device.EpName == epName);

            using (var failureSessionOptions = new SessionOptions())
            {
                failureSessionOptions.AppendExecutionProvider(ortEnvInstance, new[] { epDevice }, null);
                using var failureCompileOptions = new OrtModelCompilationOptions(failureSessionOptions);
                failureCompileOptions.SetInputModelFromBuffer(inputModel);
                failureCompileOptions.SetOutputModelPath(modelName);
                failureCompileOptions.SetEpContextEmbedMode(false);
                failureCompileOptions.SetEpContextBinaryInformation("./", modelName);
                failureCompileOptions.SetEpContextDataWriteDelegate((_, _) =>
                    throw new InvalidOperationException("synthetic C# EPContext write failure"));

                var exception = Assert.Throws<OnnxRuntimeException>(() => failureCompileOptions.CompileModel());
                Assert.Contains("synthetic C# EPContext write failure", exception.Message);
                Assert.False(File.Exists(modelName));
            }

            byte[] compiledModel;
            using (var compileSessionOptions = new SessionOptions())
            {
                compileSessionOptions.AppendExecutionProvider(ortEnvInstance, new[] { epDevice }, null);
                using var compileOptions = new OrtModelCompilationOptions(compileSessionOptions);
                compileOptions.SetInputModelFromBuffer(inputModel);
                compileOptions.SetEpContextEmbedMode(false);
                compileOptions.SetEpContextBinaryInformation("./", modelName);
                compileOptions.SetEpContextDataWriteDelegate((name, data) =>
                {
                    ++writeCount;
                    contextName = name;
                    retainedWriteData = data;
                    contextPayload = data.GetSpan().ToArray();
                });

                IntPtr modelBuffer = IntPtr.Zero;
                UIntPtr modelBufferSize = UIntPtr.Zero;
                OrtAllocator allocator = OrtAllocator.DefaultInstance;
                compileOptions.SetOutputModelBuffer(allocator, ref modelBuffer, ref modelBufferSize);
                try
                {
                    compileOptions.CompileModel();
                    compiledModel = new byte[checked((int)modelBufferSize.ToUInt64())];
                    Marshal.Copy(modelBuffer, compiledModel, 0, compiledModel.Length);
                }
                finally
                {
                    if (modelBuffer != IntPtr.Zero)
                    {
                        allocator.FreeMemory(modelBuffer);
                    }
                }
            }

            Assert.Equal(1, writeCount);
            Assert.False(string.IsNullOrEmpty(contextName));
            Assert.NotEmpty(contextPayload);
            Assert.False(File.Exists(contextName));
            Assert.Throws<ObjectDisposedException>(() => retainedWriteData.GetSpan());

            int compileReadCount = 0;
            var sourceCompileOptions = new SessionOptions();
            sourceCompileOptions.SetEpContextDataReadDelegate((name, output) =>
            {
                ++compileReadCount;
                Assert.Equal(contextName, name);
                contextPayload.CopyTo(output.Allocate(contextPayload.Length));
            }, (ulong)contextPayload.Length);
            sourceCompileOptions.AppendExecutionProvider(ortEnvInstance, new[] { epDevice }, null);
            using (var retainedCompileOptions = new OrtModelCompilationOptions(sourceCompileOptions))
            {
                sourceCompileOptions.ClearEpContextDataReadDelegate();
                sourceCompileOptions.Dispose();
                GC.Collect();
                GC.WaitForPendingFinalizers();

                retainedCompileOptions.SetInputModelFromBuffer(compiledModel);
                IntPtr recompiledModelBuffer = IntPtr.Zero;
                UIntPtr recompiledModelBufferSize = UIntPtr.Zero;
                OrtAllocator allocator = OrtAllocator.DefaultInstance;
                retainedCompileOptions.SetOutputModelBuffer(
                    allocator, ref recompiledModelBuffer, ref recompiledModelBufferSize);
                try
                {
                    retainedCompileOptions.CompileModel();
                    Assert.NotEqual(IntPtr.Zero, recompiledModelBuffer);
                    Assert.NotEqual(UIntPtr.Zero, recompiledModelBufferSize);
                }
                finally
                {
                    if (recompiledModelBuffer != IntPtr.Zero)
                    {
                        allocator.FreeMemory(recompiledModelBuffer);
                    }
                }
            }
            Assert.Equal(1, compileReadCount);
            Assert.False(File.Exists(contextName));

            using (var missingOutputLoadOptions = new SessionOptions())
            {
                missingOutputLoadOptions.SetEpContextDataReadDelegate((_, _) => { },
                    (ulong)contextPayload.Length);
                missingOutputLoadOptions.AppendExecutionProvider(ortEnvInstance, new[] { epDevice }, null);
                var exception = Assert.Throws<OnnxRuntimeException>(() =>
                    new InferenceSession(compiledModel, missingOutputLoadOptions));
                Assert.Contains("must allocate its output buffer", exception.Message);
                Assert.False(File.Exists(contextName));
            }

            using (var oversizedLoadOptions = new SessionOptions())
            {
                oversizedLoadOptions.SetEpContextDataReadDelegate((_, output) =>
                    output.Allocate(contextPayload.Length + 1),
                    (ulong)contextPayload.Length);
                oversizedLoadOptions.AppendExecutionProvider(ortEnvInstance, new[] { epDevice }, null);
                var exception = Assert.Throws<OnnxRuntimeException>(() =>
                    new InferenceSession(compiledModel, oversizedLoadOptions));
                Assert.Contains("[ErrorCode:InvalidArgument]", exception.Message);
                Assert.Contains("configured maximum size", exception.Message);
                Assert.False(File.Exists(contextName));
            }

            using (var failureLoadOptions = new SessionOptions())
            {
                failureLoadOptions.SetEpContextDataReadDelegate((_, _) =>
                    throw new InvalidOperationException("synthetic C# EPContext read failure"),
                    (ulong)contextPayload.Length);
                failureLoadOptions.AppendExecutionProvider(ortEnvInstance, new[] { epDevice }, null);
                var exception = Assert.Throws<OnnxRuntimeException>(() =>
                    new InferenceSession(compiledModel, failureLoadOptions));
                Assert.Contains("synthetic C# EPContext read failure", exception.Message);
                Assert.False(File.Exists(contextName));
            }

            var loadOptions = new SessionOptions();
            InferenceSession session = null;
            try
            {
                loadOptions.SetEpContextDataReadDelegate((name, output) =>
                {
                    ++readCount;
                    Assert.Equal(contextName, name);
                    retainedReadBuffer = output;
                    contextPayload.CopyTo(output.Allocate(contextPayload.Length));
                }, (ulong)contextPayload.Length);
                loadOptions.AppendExecutionProvider(ortEnvInstance, new[] { epDevice }, null);

                session = new InferenceSession(compiledModel, loadOptions);
                Assert.NotNull(session);
                Assert.Throws<ObjectDisposedException>(() => retainedReadBuffer.GetSpan());

                // The native session owns a snapshot of the callback state. Clearing and disposing the source
                // SessionOptions must not release that state before the native session is destroyed.
                loadOptions.ClearEpContextDataReadDelegate();
                loadOptions.Dispose();
                session.Dispose();
                session = null;
            }
            finally
            {
                session?.Dispose();
                loadOptions.Dispose();
            }

            Assert.Equal(1, readCount);
            Assert.False(File.Exists(contextName));
        }
        finally
        {
            if (contextName != null && File.Exists(contextName))
            {
                File.Delete(contextName);
            }
            if (File.Exists(modelName))
            {
                File.Delete(modelName);
            }
            ortEnvInstance.UnregisterExecutionProviderLibrary(epName);
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