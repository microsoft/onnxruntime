// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// NamedOnnxValue type, must match the native enum
    /// </summary>

    internal static class NativeMethods
    {
        private const string nativeLib = "onnxruntime";
        internal const CharSet charSet = CharSet.Ansi;

        #region Runtime/Environment API
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* OrtStatus* */OrtCreateEnv(
                                                         LogLevel default_warning_level,
                                                         string logId,
                                                         out IntPtr /*(OrtEnv*)*/ env);
        // OrtReleaseEnv should not be used
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseEnv(IntPtr /*(OrtEnv*)*/ env);
        #endregion Runtime/Environment API

        #region Status API
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern ErrorCode OrtGetErrorCode(IntPtr /*(OrtStatus*)*/status);

        // returns char*, need to convert to string by the caller.
        // does not free the underlying OrtStatus*
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* char* */OrtGetErrorMessage(IntPtr /* (OrtStatus*) */status);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseStatus(IntPtr /*(OrtStatus*)*/ statusPtr);

        #endregion Status API

        #region InferenceSession API

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* OrtStatus* */OrtCreateSession(
                                                IntPtr /* (OrtEnv*) */ environment,
                                                //[MarshalAs(UnmanagedType.LPStr)]string modelPath
                                                byte[] modelPath,
                                                IntPtr /* (OrtSessionOptions*) */sessopnOptions,
                                                out IntPtr /**/ session);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNStatus*)*/ OrtRun(
                                                IntPtr /*(OrtSession*)*/ session,
                                                IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                string[] inputNames,
                                                IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                UIntPtr inputCount,
                                                string[] outputNames,
                                                UIntPtr outputCount,

                                                [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5 /*index of outputCount*/)][In, Out]
                                                IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                );


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionGetInputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionGetOutputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetInputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetOutputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);

        // release the typeinfo using OrtReleaseTypeInfo
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetInputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /*(struct OrtTypeInfo**)*/ typeInfo);

        // release the typeinfo using OrtReleaseTypeInfo
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetOutputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseTypeInfo(IntPtr /*(OrtTypeInfo*)*/session);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseSession(IntPtr /*(OrtSession*)*/session);

        #endregion InferenceSession API

        #region SessionOptions API

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtCreateSessionOptions(out IntPtr /*(OrtSessionOptions**)*/ sessionOptions);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseSessionOptions(IntPtr /*(OrtSessionOptions*)*/session);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtCloneSessionOptions(IntPtr /*(OrtSessionOptions*)*/ sessionOptions, out IntPtr /*(OrtSessionOptions**)*/ output);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtEnableSequentialExecution(IntPtr /*(OrtSessionOptions*)*/ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtDisableSequentialExecution(IntPtr /*(OrtSessionOptions*)*/ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSetOptimizedModelFilePath(IntPtr /* OrtSessionOptions* */ options, [MarshalAs(UnmanagedType.LPWStr)]string optimizedModelFilepath);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtEnableProfiling(IntPtr /* OrtSessionOptions* */ options, string profilePathPrefix);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtDisableProfiling(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtEnableMemPattern(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtDisableMemPattern(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtEnableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtDisableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSetSessionLogId(IntPtr /* OrtSessionOptions* */ options, string logId);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSetSessionLogVerbosityLevel(IntPtr /* OrtSessionOptions* */ options, LogLevel sessionLogVerbosityLevel);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSetSessionLogSeverityLevel(IntPtr /* OrtSessionOptions* */ options, LogLevel sessionLogSeverityLevel);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSetSessionThreadPoolSize(IntPtr /* OrtSessionOptions* */ options, int sessionThreadPoolSize);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSetSessionGraphOptimizationLevel(IntPtr /* OrtSessionOptions* */ options, GraphOptimizationLevel graphOptimizationLevel);


        ///**
        //  * The order of invocation indicates the preference order as well. In other words call this method
        //  * on your most preferred execution provider first followed by the less preferred ones.
        //  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
        //  */
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CPU(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Mkldnn(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CUDA(IntPtr /*(OrtSessionOptions*) */ options, int device_id);

        //[DllImport(nativeLib, CharSet = charSet)]
        //public static extern IntPtr /*(OrtStatus*)*/ OrtCreateNupharExecutionProviderFactory(int device_id, string target_str, out IntPtr /*(OrtProviderFactoryPtr**)*/ factory);

        //[DllImport(nativeLib, CharSet = charSet)]
        //public static extern void OrtAddCustomOp(IntPtr /*(OrtSessionOptions*)*/ options, string custom_op_path);

        #endregion

        #region RunOptions API
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtCreateRunOptions( out IntPtr /* OrtRunOptions** */ runOptions);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseRunOptions(IntPtr /*(OrtRunOptions*)*/options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtRunOptionsSetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, LogLevel value);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtRunOptionsSetRunTag(IntPtr /* OrtRunOptions* */ options, string /* const char* */ runTag);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtRunOptionsGetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, out LogLevel verbosityLevel);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtRunOptionsGetRunTag(IntPtr /* const OrtRunOptions* */options, out IntPtr /* const char** */ runtag);

        // Set a flag so that any running OrtRun* calls that are using this instance of OrtRunOptions
        // will exit as soon as possible if the flag is true.
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtRunOptionsEnableTerminate(IntPtr /* OrtRunOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtRunOptionsDisableTerminate(IntPtr /* OrtRunOptions* */ options);



        #endregion

        #region Allocator/AllocatorInfo API

        //TODO: consider exposing them publicly, when allocator API is exposed
        public enum AllocatorType
        {
            DeviceAllocator = 0,
            ArenaAllocator = 1
        }

        //TODO: consider exposing them publicly when allocator API is exposed
        public enum MemoryType
        {
            CpuInput = -2,                      // Any CPU memory used by non-CPU execution provider
            CpuOutput = -1,                     // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
            Cpu = CpuOutput,                    // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
            Default = 0,                        // the default allocator for execution provider
        }


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* (OrtStatus*)*/ OrtCreateAllocatorInfo(
                                                            IntPtr /*(const char*) */name,
                                                            AllocatorType allocatorType,
                                                            int identifier,
                                                            MemoryType memType,
                                                            out IntPtr /*(OrtAllocatorInfo*)*/ allocatorInfo    // memory ownership transfered to caller
                                                       );

        //ORT_API_STATUS(OrtCreateCpuAllocatorInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1, _Out_ OrtAllocatorInfo** out)
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* (OrtStatus*)*/ OrtCreateCpuAllocatorInfo(
                                                            AllocatorType allocatorType,
                                                            MemoryType memoryType,
                                                            out IntPtr /*(OrtAllocatorInfo*)*/ allocatorInfo
                                                        );

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseAllocatorInfo(IntPtr /*(OrtAllocatorInfo*)*/ allocatorInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtGetAllocatorWithDefaultOptions(out IntPtr /*(OrtAllocator**)*/ allocator);

        /// <summary>
        /// Release any object allocated by an allocator
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="memory"></param>
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtAllocatorFree(IntPtr allocator, IntPtr memory);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtAllocatorGetInfo(IntPtr /*(const OrtAllocator*)*/ ptr, out IntPtr /*(const struct OrtAllocatorInfo**)*/info);

        #endregion Allocator/AllocatorInfo API

        #region Tensor/OnnxValue API

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetValue(IntPtr /*(OrtValue*)*/ value,
                                                                 int index,
                                                                 IntPtr /*(OrtAllocator*)*/ allocator,
                                                                 out IntPtr /*(OrtValue**)*/ outputValue);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetValueType(IntPtr /*(OrtValue*)*/ value, IntPtr /*(OnnxValueType*)*/ onnxtype);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetOnnxTypeFromTypeInfo(IntPtr /*(OrtTypeInfo*)*/ typeinfo, IntPtr /*(OnnxValueType*)*/ onnxtype);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetValueCount(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(size_t*)*/ count);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTypeInfo(IntPtr /*(OrtValue*)*/ value, IntPtr /*(OrtValue**)*/ typeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtCreateTensorAsOrtValue(
                        IntPtr /*_Inout_ OrtAllocator* */ allocator,
                        long[] /*_In_ const int64_t* */ shape, 
                        UIntPtr /*size_t*/ shape_len, 
                        TensorElementType type,
                        out IntPtr /* OrtValue** */ outputValue);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* OrtStatus */ OrtCreateTensorWithDataAsOrtValue(
                                                        IntPtr /* (const OrtAllocatorInfo*) */ allocatorInfo,
                                                        IntPtr /* (void*) */dataBufferHandle,
                                                        UIntPtr dataLength,
                                                        long[] shape,
                                                        UIntPtr shapeLength,
                                                        TensorElementType type,
                                                        out IntPtr /* OrtValue** */ outputValue);

        /// This function doesn't work with string tensor
        /// this is a no-copy method whose pointer is only valid until the backing OrtValue* is free'd.
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTensorMutableData(IntPtr /*(OrtValue*)*/ value, out IntPtr /* (void**)*/ dataBufferHandle);


        /// \param value A tensor created from OrtCreateTensor... function.
        /// \param len total data length, not including the trailing '\0' chars.
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtFillStringTensor(
                                                        IntPtr /* OrtValue */ value,
                                                        string[] /* const char* const* */s, 
                                                        UIntPtr /* size_t */ s_len);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetStringTensorContent(
                                                        IntPtr /*(OrtValue*)*/ value,
                                                        IntPtr /*(void*)*/  dst_buffer,
                                                        UIntPtr dst_buffer_len,
                                                        IntPtr offsets,
                                                        UIntPtr offsets_len);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetStringTensorDataLength(IntPtr /*(OrtValue*)*/ value,
                                                        out UIntPtr /*(size_t*)*/ len);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/
                                OrtCastTypeInfoToTensorInfo(IntPtr /*(struct OrtTypeInfo*)*/ typeInfo, out IntPtr /*(const struct OrtTensorTypeAndShapeInfo**)*/ typeAndShapeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTensorTypeAndShape(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseTensorTypeAndShapeInfo(IntPtr /*(OrtTensorTypeAndShapeInfo*)*/ value);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTensorElementType(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, IntPtr /*(TensorElementType*)*/ output);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetDimensionsCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, out UIntPtr output);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetDimensions(
                            IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo,
                            long[] dim_values,
                            UIntPtr dim_values_length);

        /**
         * How many elements does this tensor have.
         * May return a negative value
         * e.g.
         * [] -> 1
         * [1,3,4] -> 12
         * [2,0,4] -> 0
         * [-1,3,4] -> -1
         */
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTensorShapeElementCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, IntPtr /*(long*)*/ output);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseValue(IntPtr /*(OrtValue*)*/ value);

        #endregion
    } //class NativeMethods
} //namespace
