// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

internal static class RuntimeSmokeTest
{
    private static void Main(string[] args)
    {
        string libraryPath = Path.GetFullPath(args[1]);
        NativeLibrary.SetDllImportResolver(
            typeof(InferenceSession).Assembly,
            (name, assembly, searchPath) =>
                name == "onnxruntime" ? NativeLibrary.Load(libraryPath) : IntPtr.Zero);
        using var session = new InferenceSession(args[0]);
        using var options = new RunOptions();
        using var a = OrtValue.CreateTensorValueFromMemory(
            new float[] { 1, 2, 3, 4, 5, 6 }, new long[] { 2, 3 });
        using var b = OrtValue.CreateTensorValueFromMemory(
            new float[] { 10, 20, 30, 40, 50, 60 }, new long[] { 2, 3 });
        using var output = session.Run(options, new[] { "A", "B" }, new[] { a, b }, new[] { "C" });
        var result = output.First();
        var typeAndShape = result.GetTensorTypeAndShape();
        if (!typeAndShape.Shape.SequenceEqual(new long[] { 2, 3 }) ||
            !result.GetTensorDataAsSpan<float>().SequenceEqual(new float[] { 11, 22, 33, 44, 55, 66 }))
        {
            throw new InvalidOperationException("C# Add inference returned incorrect values or shape.");
        }
        Console.WriteLine("C#/PInvoke Add inference passed.");
    }
}
