﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime.CompileApi;

using System;
using System.Runtime.InteropServices;

// NOTE: The order of the APIs in this struct should match exactly that in OrtCompileApi
// See onnxruntime/core/session/compile_api.cc.
[StructLayout(LayoutKind.Sequential)]
public struct OrtCompileApi
{
    public IntPtr ReleaseModelCompilationOptions;
    public IntPtr CreateModelCompilationOptionsFromSessionOptions;
    public IntPtr ModelCompilationOptions_SetInputModelPath;
    public IntPtr ModelCompilationOptions_SetInputModelFromBuffer;
    public IntPtr ModelCompilationOptions_SetOutputModelPath;
    public IntPtr ModelCompilationOptions_SetOutputModelExternalInitializersFile;
    public IntPtr ModelCompilationOptions_SetOutputModelBuffer;
    public IntPtr ModelCompilationOptions_SetEpContextEmbedMode;
    public IntPtr CompileModel;
}

internal class NativeMethods
{
    private static OrtCompileApi _compileApi;

    //
    // Define the delegate signatures, and a static member for each to hold the marshaled function pointer.
    //
    // We populate the static members in the constructor of this class.
    //
    // The C# code will call the C++ API through the delegate instances in the static members.
    //

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate void DOrtReleaseModelCompilationOptions(IntPtr /* OrtModelCompilationOptions* */ options);
    public DOrtReleaseModelCompilationOptions OrtReleaseModelCompilationOptions;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtCreateModelCompilationOptionsFromSessionOptions(
        IntPtr /* const OrtEnv* */ env,
        IntPtr /* const OrtSessionOptions* */ sessionOptions,
        out IntPtr /* OrtModelCompilationOptions** */ outOptions);
    public DOrtCreateModelCompilationOptionsFromSessionOptions 
                    OrtCreateModelCompilationOptionsFromSessionOptions;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtModelCompilationOptions_SetInputModelPath(
        IntPtr /* OrtModelCompilationOptions* */ options,
        byte[] /* const ORTCHAR_T* */ inputModelPath);
    public DOrtModelCompilationOptions_SetInputModelPath OrtModelCompilationOptions_SetInputModelPath;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtModelCompilationOptions_SetInputModelFromBuffer(
        IntPtr /* OrtModelCompilationOptions* */ options,
        byte[] /* const void* */ inputModelData,
        UIntPtr /* size_t */ inputModelDataSize);
    public DOrtModelCompilationOptions_SetInputModelFromBuffer 
                    OrtModelCompilationOptions_SetInputModelFromBuffer;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtModelCompilationOptions_SetOutputModelPath(
        IntPtr /* OrtModelCompilationOptions* */ options,
        byte[] /* const ORTCHAR_T* */ outputModelPath);
    public DOrtModelCompilationOptions_SetOutputModelPath OrtModelCompilationOptions_SetOutputModelPath;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtModelCompilationOptions_SetOutputModelExternalInitializersFile(
        IntPtr /* OrtModelCompilationOptions* */ options,
        byte[] /* const ORTCHAR_T* */ externalInitializersFilePath,
        UIntPtr /* size_t */ externalInitializerSizeThreshold);
    public DOrtModelCompilationOptions_SetOutputModelExternalInitializersFile 
                    OrtModelCompilationOptions_SetOutputModelExternalInitializersFile;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtModelCompilationOptions_SetOutputModelBuffer(
        IntPtr /* OrtModelCompilationOptions* */ options,
        IntPtr /* OrtAllocator* */ allocator,
        ref IntPtr /* void** */ outputModelBufferPtr,
        ref UIntPtr /* size_t* */ outputModelBufferSizePtr);
    public DOrtModelCompilationOptions_SetOutputModelBuffer OrtModelCompilationOptions_SetOutputModelBuffer;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtModelCompilationOptions_SetEpContextEmbedMode(
        IntPtr /* OrtModelCompilationOptions* */ options,
        bool embedEpContextInModel);
    public DOrtModelCompilationOptions_SetEpContextEmbedMode OrtModelCompilationOptions_SetEpContextEmbedMode;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    public delegate IntPtr /* OrtStatus* */ DOrtCompileModel(
        IntPtr /* const OrtEnv* */ env,
        IntPtr /* const OrtModelCompilationOptions* */ modelOptions);
    public DOrtCompileModel OrtCompileModel;

    internal NativeMethods(OnnxRuntime.NativeMethods.DOrtGetCompileApi getCompileApi)
    {

#if NETSTANDARD2_0
        IntPtr compileApiPtr = getCompileApi();
        _compileApi = (OrtCompileApi)Marshal.PtrToStructure(compileApiPtr, typeof(OrtCompileApi));
#else
        _compileApi = (OrtCompileApi)getCompileApi();
#endif

        OrtReleaseModelCompilationOptions = 
            (DOrtReleaseModelCompilationOptions)Marshal.GetDelegateForFunctionPointer(
                _compileApi.ReleaseModelCompilationOptions, 
                typeof(DOrtReleaseModelCompilationOptions));

        OrtCreateModelCompilationOptionsFromSessionOptions = 
            (DOrtCreateModelCompilationOptionsFromSessionOptions)Marshal.GetDelegateForFunctionPointer(
                _compileApi.CreateModelCompilationOptionsFromSessionOptions, 
                typeof(DOrtCreateModelCompilationOptionsFromSessionOptions));

        OrtModelCompilationOptions_SetInputModelPath = 
            (DOrtModelCompilationOptions_SetInputModelPath)Marshal.GetDelegateForFunctionPointer(
                _compileApi.ModelCompilationOptions_SetInputModelPath, 
                typeof(DOrtModelCompilationOptions_SetInputModelPath));

        OrtModelCompilationOptions_SetInputModelFromBuffer = 
            (DOrtModelCompilationOptions_SetInputModelFromBuffer)Marshal.GetDelegateForFunctionPointer(
                _compileApi.ModelCompilationOptions_SetInputModelFromBuffer, 
                typeof(DOrtModelCompilationOptions_SetInputModelFromBuffer));

        OrtModelCompilationOptions_SetOutputModelPath = 
            (DOrtModelCompilationOptions_SetOutputModelPath)Marshal.GetDelegateForFunctionPointer(
                _compileApi.ModelCompilationOptions_SetOutputModelPath, 
                typeof(DOrtModelCompilationOptions_SetOutputModelPath));

        OrtModelCompilationOptions_SetOutputModelExternalInitializersFile = 
            (DOrtModelCompilationOptions_SetOutputModelExternalInitializersFile)Marshal.GetDelegateForFunctionPointer(
                _compileApi.ModelCompilationOptions_SetOutputModelExternalInitializersFile, 
                typeof(DOrtModelCompilationOptions_SetOutputModelExternalInitializersFile));

        OrtModelCompilationOptions_SetOutputModelBuffer = 
            (DOrtModelCompilationOptions_SetOutputModelBuffer)Marshal.GetDelegateForFunctionPointer(
                _compileApi.ModelCompilationOptions_SetOutputModelBuffer, 
                typeof(DOrtModelCompilationOptions_SetOutputModelBuffer));

        OrtModelCompilationOptions_SetEpContextEmbedMode = 
            (DOrtModelCompilationOptions_SetEpContextEmbedMode)Marshal.GetDelegateForFunctionPointer(
                _compileApi.ModelCompilationOptions_SetEpContextEmbedMode, 
                typeof(DOrtModelCompilationOptions_SetEpContextEmbedMode));

        OrtCompileModel = 
            (DOrtCompileModel)Marshal.GetDelegateForFunctionPointer(
                _compileApi.CompileModel, 
                typeof(DOrtCompileModel));
    }
}
