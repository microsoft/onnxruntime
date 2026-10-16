// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if NET8_0_OR_GREATER && !(ANDROID || IOS)
namespace Microsoft.ML.OnnxRuntime.Tests;

using Google.Protobuf;
using Onnx;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

[Collection("Ort Inference Tests")]
public class EncryptedEpContextTests
{
    [SkippableFact]
    public void PersistedEncryptedModelAndContextRunInference()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Windows),
            "Requires the Windows example plugin EP integration artifact.");
        string libraryPath = Path.Combine(Directory.GetCurrentDirectory(), "example_plugin_ep.dll");
        Assert.True(File.Exists(libraryPath), $"Expected library {libraryPath} does not exist.");

        var env = OrtEnv.Instance();
        const string epName = "encrypted_ep_context";
        const string modelName = "encrypted_model.onnx";
        string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string modelFile = Path.Combine(directory, "model.aesgcm");
        string contextFile = Path.Combine(directory, "context.aesgcm");
        byte[] key = RandomNumberGenerator.GetBytes(32);
        string contextName = null;
        int writes = 0;
        int reads = 0;

        env.RegisterExecutionProviderLibrary(epName, libraryPath);
        try
        {
            var device = env.GetEpDevices().Single(d => d.EpName == epName);
            ModelProto original = ModelProto.Parser.ParseFrom(
                TestDataLoader.LoadModelFromEmbeddedResource("mul_1.onnx"));
            NodeProto mul = Assert.Single(original.Graph.Node);
            Assert.Equal("Mul", mul.OpType);
            var inputType = original.Graph.Input[0].Type.Clone();
            int[] dimensions = inputType.TensorType.Shape.Dim.Select(d => checked((int)d.DimValue)).ToArray();
            Assert.All(dimensions, d => Assert.True(d > 0));
            string[] inputNames = mul.Input.ToArray();
            original.Graph.Input.Clear();
            foreach (string name in inputNames)
            {
                original.Graph.Input.Add(new ValueInfoProto { Name = name, Type = inputType.Clone() });
            }
            original.Graph.Initializer.Clear();
            byte[] inputModel = original.ToByteArray();

            using (var options = CreateOptions(env, device))
            using (var compile = new OrtModelCompilationOptions(options))
            using (var inputPin = inputModel.AsMemory().Pin())
            {
                compile.SetInputModelFromBuffer(inputModel);
                compile.SetEpContextEmbedMode(false);
                compile.SetEpContextBinaryInformation(directory + Path.DirectorySeparatorChar, modelName);
                compile.SetEpContextDataWriteDelegate((name, data) =>
                {
                    Assert.Equal(0, writes++);
                    contextName = name;
                    byte[] plaintext = data.GetSpan().ToArray();
                    try
                    {
                        File.WriteAllBytes(contextFile, Encrypt(plaintext, key, name));
                    }
                    finally
                    {
                        CryptographicOperations.ZeroMemory(plaintext);
                    }
                });
                IntPtr buffer = IntPtr.Zero;
                UIntPtr size = UIntPtr.Zero;
                var allocator = OrtAllocator.DefaultInstance;
                compile.SetOutputModelBuffer(allocator, ref buffer, ref size);
                try
                {
                    compile.CompileModel();
                    byte[] plaintext = new byte[checked((int)size.ToUInt64())];
                    Marshal.Copy(buffer, plaintext, 0, plaintext.Length);
                    try
                    {
                        var compiled = ModelProto.Parser.ParseFrom(plaintext);
                        Assert.Equal("EPContext", Assert.Single(compiled.Graph.Node).OpType);
                        Assert.Empty(compiled.Graph.Initializer);
                        File.WriteAllBytes(modelFile, Encrypt(plaintext, key, modelName));
                    }
                    finally
                    {
                        CryptographicOperations.ZeroMemory(plaintext);
                    }
                }
                finally
                {
                    if (buffer != IntPtr.Zero)
                    {
                        allocator.FreeMemory(buffer);
                    }
                }
            }
            CryptographicOperations.ZeroMemory(inputModel);
            Assert.Equal(1, writes);
            Assert.False(string.IsNullOrEmpty(contextName));
            Assert.Equal(2, Directory.GetFiles(directory).Length);
            Assert.False(File.Exists(contextName));
            Assert.False(File.Exists(Path.Combine(directory, modelName)));

            byte[] encryptedModel = File.ReadAllBytes(modelFile);
            byte[] encryptedContext = File.ReadAllBytes(contextFile);
            byte[] restoredModel = Decrypt(encryptedModel, key, modelName);
            try
            {
                using (var options = CreateOptions(env, device))
                {
                    options.SetEpContextDataReadDelegate((name, output) =>
                    {
                        Assert.Equal(contextName, name);
                        ++reads;
                        byte[] plaintext = Decrypt(File.ReadAllBytes(contextFile), key, name);
                        try
                        {
                            plaintext.CopyTo(output.Allocate(plaintext.Length));
                        }
                        finally
                        {
                            CryptographicOperations.ZeroMemory(plaintext);
                        }
                    }, (ulong)(encryptedContext.Length - 28));
                    using var session = new InferenceSession(restoredModel, options);
                    int count = dimensions.Aggregate(1, (a, b) => checked(a * b));
                    for (int iteration = 0; iteration < 2; ++iteration)
                    {
                        float[] left = Enumerable.Range(0, count).Select(i => (float)(i - iteration)).ToArray();
                        float[] right = Enumerable.Range(0, count).Select(i => (float)(2 * i + iteration)).ToArray();
                        var inputs = new[]
                        {
                            NamedOnnxValue.CreateFromTensor(inputNames[0], new DenseTensor<float>(left, dimensions)),
                            NamedOnnxValue.CreateFromTensor(inputNames[1], new DenseTensor<float>(right, dimensions))
                        };
                        using var results = session.Run(inputs);
                        Assert.Equal(left.Zip(right, (a, b) => a * b), Assert.Single(results).AsEnumerable<float>());
                    }
                }
                Assert.Equal(1, reads);

                byte[] wrongKey = (byte[])key.Clone();
                wrongKey[0] ^= 1;
                try
                {
                    Assert.Throws<AuthenticationTagMismatchException>(() => Decrypt(encryptedModel, wrongKey, modelName));
                    AssertContextRejected(wrongKey, encryptedContext);
                }
                finally
                {
                    CryptographicOperations.ZeroMemory(wrongKey);
                }
                encryptedModel[28] ^= 1;
                Assert.Throws<AuthenticationTagMismatchException>(() => Decrypt(encryptedModel, key, modelName));
                encryptedContext[28] ^= 1;
                AssertContextRejected(key, encryptedContext);
                Assert.Throws<AuthenticationTagMismatchException>(() =>
                    Decrypt(File.ReadAllBytes(contextFile), key, "different-context-name"));

                using var noCallbackOptions = CreateOptions(env, device);
                Assert.Throws<OnnxRuntimeException>(() => new InferenceSession(restoredModel, noCallbackOptions));

                using var invalidPayloadOptions = CreateOptions(env, device);
                invalidPayloadOptions.SetEpContextDataReadDelegate((_, output) =>
                    Encoding.UTF8.GetBytes("binary_data").CopyTo(output.Allocate(11)), 11);
                var invalidPayload = Assert.Throws<OnnxRuntimeException>(() =>
                    new InferenceSession(restoredModel, invalidPayloadOptions));
                Assert.Contains("Invalid compiled test Mul payload", invalidPayload.Message);

                void AssertContextRejected(byte[] readKey, byte[] ciphertext)
                {
                    using var options = CreateOptions(env, device);
                    options.SetEpContextDataReadDelegate((name, _) =>
                    {
                        try
                        {
                            Decrypt(ciphertext, readKey, name);
                        }
                        catch (AuthenticationTagMismatchException ex)
                        {
                            throw new CryptographicException("EPContext authentication failed", ex);
                        }
                        throw new InvalidOperationException("Corrupt ciphertext was accepted.");
                    }, (ulong)(ciphertext.Length - 28));
                    var error = Assert.Throws<OnnxRuntimeException>(() => new InferenceSession(restoredModel, options));
                    Assert.Contains("EPContext authentication failed", error.Message);
                }
            }
            finally
            {
                CryptographicOperations.ZeroMemory(restoredModel);
            }
            Assert.Equal(2, Directory.GetFiles(directory).Length);
            Assert.False(File.Exists(contextName));
        }
        finally
        {
            env.UnregisterExecutionProviderLibrary(epName);
            CryptographicOperations.ZeroMemory(key);
            File.Delete(modelFile);
            File.Delete(contextFile);
            Directory.Delete(directory);
        }
    }

    private static SessionOptions CreateOptions(OrtEnv env, OrtEpDevice device)
    {
        var options = new SessionOptions();
        options.AddSessionConfigEntry("ep.example.test_execute_ep_context", "1");
        options.AppendExecutionProvider(env, new[] { device }, null);
        return options;
    }

    // Persist nonce || tag || ciphertext. Bind each record to its logical ORT asset name.
    private static byte[] Encrypt(byte[] plaintext, byte[] key, string name)
    {
        byte[] record = new byte[28 + plaintext.Length];
        RandomNumberGenerator.Fill(record.AsSpan(0, 12));
        using var aes = new AesGcm(key, 16);
        aes.Encrypt(record.AsSpan(0, 12), plaintext, record.AsSpan(28), record.AsSpan(12, 16),
            Encoding.UTF8.GetBytes(name));
        return record;
    }

    private static byte[] Decrypt(byte[] record, byte[] key, string name)
    {
        byte[] plaintext = new byte[record.Length - 28];
        using var aes = new AesGcm(key, 16);
        aes.Decrypt(record.AsSpan(0, 12), record.AsSpan(28), record.AsSpan(12, 16), plaintext,
            Encoding.UTF8.GetBytes(name));
        return plaintext;
    }
}
#endif
