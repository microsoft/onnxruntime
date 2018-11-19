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
        private const string nativeLib = "onnxruntime.dll";
        internal const CharSet charSet = CharSet.Ansi;

        #region Runtime/Environment API
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* ONNXStatus* */ONNXRuntimeInitialize(
                                                         LogLevel default_warning_level,
                                                         string logId,
                                                         out IntPtr /*(ONNXEnv*)*/ env);
        // ReleaseONNXEnv should not be used
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ReleaseONNXEnv(IntPtr /*(ONNXEnv*)*/ env);
        #endregion Runtime/Environment API

        #region Status API
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern ErrorCode ONNXRuntimeGetErrorCode(IntPtr /*(ONNXStatus*)*/status);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* char* */ONNXRuntimeGetErrorMessage(IntPtr /* (ONNXStatus*) */status);
                                                                                           // returns char*, need to convert to string by the caller.
                                                                                           // does not free the underlying ONNXStatus*

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ReleaseONNXStatus(IntPtr /*(ONNXStatus*)*/ statusPtr);

        #endregion Status API

        #region InferenceSession API

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* ONNXStatus* */ONNXRuntimeCreateInferenceSession(
                                                        IntPtr /* (ONNXEnv*) */ environment,
                                                        [MarshalAs(UnmanagedType.LPWStr)]string modelPath, //the model path is consumed as a wchar* in the C-api
                                                        IntPtr /* (ONNXRuntimeSessionOptions*) */sessopnOptions,
                                                        out IntPtr /**/ session);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNStatus*)*/ ONNXRuntimeRunInferenceAndFetchAll(
                                                IntPtr /*(ONNXSessionPtr)*/ session,
                                                string[] inputNames,
                                                IntPtr[] /*(ONNXValuePtr[])*/ inputValues,
                                                ulong inputLength,  // size_t, TODO: make it portable for x86, arm
                                                out IntPtr /* (ONNXValueListPtr*)*/ outputValues,
                                                out ulong /* (size_t*) */ outputLength); //TODO: make it portable for x86, arm


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNStatus*)*/ ONNXRuntimeRunInference(
                                                IntPtr /*(ONNXSession*)*/ session,
                                                string[] inputNames,
                                                IntPtr[] /* (ONNXValue*[])*/ inputValues,
                                                ulong inputCount,  /* TODO: size_t, make it portable for x86 arm */
                                                string[] outputNames,
                                                ulong outputCount,  /* TODO: size_t, make it portable for x86 and arm */

                                                [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5 /*index of outputCount*/)][In, Out]
                                                IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                );


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXStatus*)*/ ONNXRuntimeInferenceSessionGetInputCount(
                                                IntPtr /*(ONNXSession*)*/ session, 
                                                out ulong /* TODO: size_t */ count);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXStatus*)*/ ONNXRuntimeInferenceSessionGetOutputCount(
                                                IntPtr /*(ONNXSession*)*/ session,
                                                out ulong /*TODO: size_t port*/ count);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXStatus*)*/ONNXRuntimeInferenceSessionGetInputName(
                                                IntPtr /*(ONNXSession*)*/ session,
                                                ulong index,  //TODO: port size_t 
                                                IntPtr /*(ONNXRuntimeAllocator*)*/ allocator, 
                                                out IntPtr /*(char**)*/name);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXStatus*)*/ONNXRuntimeInferenceSessionGetOutputName(
                                                IntPtr /*(ONNXSession*)*/ session,
                                                ulong index,  //TODO: port size_t 
                                                IntPtr /*(ONNXRuntimeAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ReleaseONNXSession(IntPtr /*(ONNXSession*)*/session);

        #endregion InferenceSession API

        #region SessionOptions API
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*ONNXRuntimeSessionOptions* */ ONNXRuntimeCreateSessionOptions();

        //DEFINE_RUNTIME_CLASS(ONNXRuntimeSessionOptions)
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ReleaseONNXRuntimeSessionOptions(IntPtr /*(ONNXRuntimeSessionOptions*)*/ sessionOptions);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeEnableSequentialExecution(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeDisableSequentialExecution(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeEnableProfiling(IntPtr /* ONNXRuntimeSessionOptions* */ options, string profilePathPrefix);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeDisableProfiling(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeEnableMemPattern(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeDisableMemPattern(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeEnableCpuMemArena(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeDisableCpuMemArena(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeSetSessionLogId(IntPtr /* ONNXRuntimeSessionOptions* */ options, string logId);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeSetSessionLogVerbosityLevel(IntPtr /* ONNXRuntimeSessionOptions* */ options, LogLevel sessionLogVerbosityLevel);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern int ONNXRuntimeSetSessionThreadPoolSize(IntPtr /* ONNXRuntimeSessionOptions* */ options, int sessionThreadPoolSize);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern int ONNXRuntimeEnableCudaProvider(IntPtr /* ONNXRuntimeSessionOptions* */ options, int deviceId);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeDisableCudaProvider(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern int ONNXRuntimeEnableMklProvider(IntPtr /* ONNXRuntimeSessionOptions* */ options);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeDisableMklProvider(IntPtr /* ONNXRuntimeSessionOptions* */ options);

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
        public static extern IntPtr /* (ONNXStatus*)*/ ONNXRuntimeCreateAllocatorInfo(
                                                            IntPtr /*(const char*) */name,
                                                            AllocatorType allocatorType,
                                                            int identifier,
                                                            MemoryType memType,
                                                            out IntPtr /*(ONNXRuntimeAllocatorInfo*)*/ allocatorInfo    // memory ownership transfered to caller
                                                       );

        //ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateCpuAllocatorInfo, enum ONNXRuntimeAllocatorType type, enum ONNXRuntimeMemType mem_type1, _Out_ ONNXRuntimeAllocatorInfo** out)
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* (ONNXStatus*)*/ ONNXRuntimeCreateCpuAllocatorInfo(
                                                            AllocatorType allocatorType,
                                                            MemoryType memoryType,
                                                            out IntPtr /*(ONNXRuntimeAllocatorInfo*)*/ allocatorInfo
                                                        );

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ReleaseONNXRuntimeAllocatorInfo(IntPtr /*(ONNXRuntimeAllocatorInfo*)*/ allocatorInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXStatus*)*/ONNXRuntimeCreateDefaultAllocator(out IntPtr /*(ONNXRuntimeAllocator**)*/ allocator);

        /// <summary>
        ///  Releases/Unrefs any object, including the Allocator
        /// </summary>
        /// <param name="ptr"></param>
        /// <returns>remaining ref count</returns>
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern uint /*remaining ref count*/ ONNXRuntimeReleaseObject(IntPtr /*(void*)*/ ptr);

        /// <summary>
        /// Release any object allocated by an allocator
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="memory"></param>
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeAllocatorFree(IntPtr allocator, IntPtr memory);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(const struct ONNXRuntimeAllocatorInfo*)*/ ONNXRuntimeAllocatorGetInfo(IntPtr /*(const ONNXRuntimeAllocator*)*/ ptr);

        #endregion Allocator/AllocatorInfo API

        #region Tensor/OnnxValue API

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /* ONNXStatus */ ONNXRuntimeCreateTensorWithDataAsONNXValue(
                                                        IntPtr /* (const ONNXRuntimeAllocatorInfo*) */ allocatorInfo,
                                                        IntPtr /* (void*) */dataBufferHandle,
                                                        ulong dataLength,   //size_t, TODO: make it portable for x86, arm
                                                        ulong[] shape,   //size_t* or size_t[], TODO: make it portable for x86, arm
                                                        ulong shapeLength, //size_t,  TODO: make it portable for x86, arm
                                                        TensorElementType type,
                                                        out IntPtr /* ONNXValuePtr* */ outputValue);

        /// This function doesn't work with string tensor
        /// this is a no-copy method whose pointer is only valid until the backing ONNXValuePtr is free'd.
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXStatus*)*/ ONNXRuntimeGetTensorMutableData(IntPtr /*(ONNXValue*)*/ value, out IntPtr /* (void**)*/ dataBufferHandle);

        //[DllImport(nativeLib, CharSet = charSet)]
        //public static extern IntPtr /*(ONNXStatus*)*/ ONNXRuntimeGetTensorShapeDimCount(IntPtr /*(ONNXValue*)*/ value, out ulong dimension); //size_t TODO: make it portable for x86, arm

        //[DllImport(nativeLib, CharSet = charSet)]
        //public static extern IntPtr /*(ONNXStatus*)*/ ONNXRuntimeGetTensorShapeElementCount(IntPtr /*(ONNXValue*)*/value, out ulong count);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXStatus*)*/ ONNXRuntimeGetTensorShapeAndType(IntPtr /*(ONNXValue*)*/ value, out IntPtr /*(struct ONNXRuntimeTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern TensorElementType ONNXRuntimeGetTensorElementType(IntPtr /*(const struct ONNXRuntimeTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern ulong /*TODO: port for size_t */ONNXRuntimeGetNumOfDimensions(IntPtr /*(const struct ONNXRuntimeTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ONNXRuntimeGetDimensions(
                            IntPtr /*(const struct ONNXRuntimeTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, 
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
        public static extern long ONNXRuntimeGetTensorShapeElementCount(IntPtr /*(const struct ONNXRuntimeTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);


        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(ONNXValuePtr)*/ ONNXRuntimeONNXValueListGetNthValue(IntPtr /*(ONNXValueListPtr)*/ list, ulong index);  // 0-based index TODO: size_t, make it portable for x86, arm

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ReleaseONNXValue(IntPtr /*(ONNXValue*)*/ value);

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern void ReleaseONNXValueList(IntPtr /*(ONNXValueList*)*/ valueList);

        #endregion
    } //class NativeMethods
} //namespace
