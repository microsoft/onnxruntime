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
                                                ulong inputCount,  /* TODO: size_t, make it portable for x86 arm */
                                                string[] outputNames,
                                                ulong outputCount,  /* TODO: size_t, make it portable for x86 and arm */

                                                [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5 /*index of outputCount*/)][In, Out]
                                                IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                );


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionGetInputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out ulong /* TODO: size_t */ count);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionGetOutputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out ulong /*TODO: size_t port*/ count);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetInputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                ulong index,  //TODO: port size_t 
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetOutputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                ulong index,  //TODO: port size_t 
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);

        // release the typeinfo using OrtReleaseTypeInfo
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetInputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                ulong index, //TODO: port for size_t
                                                out IntPtr /*(struct OrtTypeInfo**)*/ typeInfo);

        // release the typeinfo using OrtReleaseTypeInfo
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/OrtSessionGetOutputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                ulong index, //TODO: port for size_t
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseTypeInfo(IntPtr /*(OrtTypeInfo*)*/session);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseSession(IntPtr /*(OrtSession*)*/session);

        #endregion InferenceSession API

        #region SessionOptions API

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*OrtSessionOptions* */ OrtCreateSessionOptions();

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseSessionOptions(IntPtr /*(OrtSessionOptions*)*/session);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtSessionOptions*)*/OrtCloneSessionOptions(IntPtr /*(OrtSessionOptions*)*/ sessionOptions);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtEnableSequentialExecution(IntPtr /*(OrtSessionOptions*)*/ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtDisableSequentialExecution(IntPtr /*(OrtSessionOptions*)*/ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtEnableProfiling(IntPtr /* OrtSessionOptions* */ options, string profilePathPrefix);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtDisableProfiling(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtEnableMemPattern(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtDisableMemPattern(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtEnableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtDisableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtSetSessionLogId(IntPtr /* OrtSessionOptions* */ options, string logId);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtSetSessionLogVerbosityLevel(IntPtr /* OrtSessionOptions* */ options, LogLevel sessionLogVerbosityLevel);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern int OrtSetSessionThreadPoolSize(IntPtr /* OrtSessionOptions* */ options, int sessionThreadPoolSize);


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
        public static extern IntPtr /*(OrtStatus*)*/OrtCreateDefaultAllocator(out IntPtr /*(OrtAllocator**)*/ allocator);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseAllocator(IntPtr /*(OrtAllocator*)*/ allocator);

        /// <summary>
        /// Release any object allocated by an allocator
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="memory"></param>
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtAllocatorFree(IntPtr allocator, IntPtr memory);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(const struct OrtAllocatorInfo*)*/ OrtAllocatorGetInfo(IntPtr /*(const OrtAllocator*)*/ ptr);

        #endregion Allocator/AllocatorInfo API

        #region Tensor/OnnxValue API

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetValue(IntPtr /*(OrtValue*)*/ value,
                                                                 int index,
                                                                 IntPtr /*(OrtAllocator*)*/ allocator,
                                                                 out IntPtr /*(OrtValue**)*/ outputValue);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern OnnxValueType /*Onnxtype*/   OrtGetValueType(IntPtr /*(OrtValue*)*/ value);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetValueCount(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(size_t*)*/ count);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTypeInfo(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(OrtValue**)*/ typeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* OrtStatus */ OrtCreateTensorWithDataAsOrtValue(
                                                        IntPtr /* (const OrtAllocatorInfo*) */ allocatorInfo,
                                                        IntPtr /* (void*) */dataBufferHandle,
                                                        ulong dataLength,   //size_t, TODO: make it portable for x86, arm
                                                        ulong[] shape,   //size_t* or size_t[], TODO: make it portable for x86, arm
                                                        ulong shapeLength, //size_t,  TODO: make it portable for x86, arm
                                                        TensorElementType type,
                                                        out IntPtr /* OrtValue** */ outputValue);

        /// This function doesn't work with string tensor
        /// this is a no-copy method whose pointer is only valid until the backing OrtValue* is free'd.
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTensorMutableData(IntPtr /*(OrtValue*)*/ value, out IntPtr /* (void**)*/ dataBufferHandle);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetStringTensorContent(
                                                        IntPtr /*(OrtValue*)*/ value,
                                                        IntPtr /*(void*)*/  dst_buffer,
                                                        ulong dst_buffer_len,  //size_t,  TODO: make it portable for x86, arm
                                                        IntPtr offsets,
                                                        ulong offsets_len);    //size_t,  TODO: make it portable for x86, arm

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetStringTensorDataLength(IntPtr /*(OrtValue*)*/ value, out ulong /*(size_t*)*/ len);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/
                                OrtCastTypeInfoToTensorInfo(IntPtr /*(struct OrtTypeInfo*)*/ typeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtGetTensorShapeAndType(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseTensorTypeAndShapeInfo(IntPtr /*(OrtTensorTypeAndShapeInfo*)*/ value);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern TensorElementType OrtGetTensorElementType(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern ulong /*TODO: port for size_t */OrtGetNumOfDimensions(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtGetDimensions(
                            IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo,
                            long[] dim_values,
                            ulong dim_values_length);

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
        public static extern long OrtGetTensorShapeElementCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void OrtReleaseValue(IntPtr /*(OrtValue*)*/ value);

        #endregion
    } //class NativeMethods
} //namespace
