// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
#if __ENABLE_TRAINING_APIS__
        // NOTE: The order of the APIs in this struct should match exactly that in
        // OrtTrainingApi  (onnxruntime_training_c_api.cc)
        [StructLayout(LayoutKind.Sequential)]
        public struct OrtTrainingApi
        {
            public IntPtr LoadCheckpoint;
            public IntPtr SaveCheckpoint;
            public IntPtr CreateTrainingSession;
            public IntPtr CreateTrainingSessionFromBuffer;
            public IntPtr TrainingSessionGetTrainingModelOutputCount;
            public IntPtr TrainingSessionGetEvalModelOutputCount;
            public IntPtr TrainingSessionGetTrainingModelOutputName;
            public IntPtr TrainingSessionGetEvalModelOutputName;
            public IntPtr LazyResetGrad;
            public IntPtr TrainStep;
            public IntPtr EvalStep;
            public IntPtr SetLearningRate;
            public IntPtr GetLearningRate;
            public IntPtr OptimizerStep;
            public IntPtr RegisterLinearLRScheduler;
            public IntPtr SchedulerStep;
            public IntPtr GetParametersSize;
            public IntPtr CopyParametersToBuffer;
            public IntPtr CopyBufferToParameters;
            public IntPtr ReleaseTrainingSession;
            public IntPtr ReleaseCheckpointState;
            public IntPtr ExportModelForInferencing;
            public IntPtr SetSeed;
            public IntPtr TrainingSessionGetTrainingModelInputCount;
            public IntPtr TrainingSessionGetEvalModelInputCount;
            public IntPtr TrainingSessionGetTrainingModelInputName;
            public IntPtr TrainingSessionGetEvalModelInputName;
            public IntPtr AddProperty;
            public IntPtr GetProperty;
            public IntPtr LoadCheckpointFromBuffer;
            public IntPtr GetParameterTypeAndShape;
            public IntPtr UpdateParameter;
            public IntPtr GetParameter;
        }

        internal static class NativeTrainingMethods
        {
            static OrtApi api_;
            static OrtTrainingApi trainingApi_;
            static IntPtr trainingApiPtr;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate ref OrtApi DOrtGetApi(UInt32 version);

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtTrainingApi* */ DOrtGetTrainingApi(UInt32 version);
            public static DOrtGetTrainingApi OrtGetTrainingApi;

        static NativeTrainingMethods()
            {
                DOrtGetApi OrtGetApi = (DOrtGetApi)Marshal.GetDelegateForFunctionPointer(NativeMethods.OrtGetApiBase().GetApi, typeof(DOrtGetApi));

                // TODO: Make this save the pointer, and not copy the whole structure across
                api_ = (OrtApi)OrtGetApi(17 /*ORT_API_VERSION*/);

                OrtGetTrainingApi = (DOrtGetTrainingApi)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingApi, typeof(DOrtGetTrainingApi));
                trainingApiPtr = OrtGetTrainingApi(17 /*ORT_API_VERSION*/);
                if (trainingApiPtr != IntPtr.Zero)
                {
                    trainingApi_ = (OrtTrainingApi)Marshal.PtrToStructure(trainingApiPtr, typeof(OrtTrainingApi));
                    OrtLoadCheckpoint = (DOrtLoadCheckpoint)Marshal.GetDelegateForFunctionPointer(trainingApi_.LoadCheckpoint, typeof(DOrtLoadCheckpoint));
                    OrtSaveCheckpoint = (DOrtSaveCheckpoint)Marshal.GetDelegateForFunctionPointer(trainingApi_.SaveCheckpoint, typeof(DOrtSaveCheckpoint));
                    OrtCreateTrainingSession = (DOrtCreateTrainingSession)Marshal.GetDelegateForFunctionPointer(trainingApi_.CreateTrainingSession, typeof(DOrtCreateTrainingSession));
                    OrtGetTrainingModelOutputCount = (DOrtGetTrainingModelOutputCount)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetTrainingModelOutputCount, typeof(DOrtGetTrainingModelOutputCount));
                    OrtGetEvalModelOutputCount = (DOrtGetEvalModelOutputCount)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetEvalModelOutputCount, typeof(DOrtGetEvalModelOutputCount));
                    OrtGetTrainingModelOutputName = (DOrtGetTrainingModelOutputName)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetTrainingModelOutputName, typeof(DOrtGetTrainingModelOutputName));
                    OrtGetEvalModelOutputName = (DOrtGetEvalModelOutputName)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetEvalModelOutputName, typeof(DOrtGetEvalModelOutputName));
                    OrtLazyResetGrad = (DOrtLazyResetGrad)Marshal.GetDelegateForFunctionPointer(trainingApi_.LazyResetGrad, typeof(DOrtLazyResetGrad));
                    OrtTrainStep = (DOrtTrainStep)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainStep, typeof(DOrtTrainStep));
                    OrtEvalStep = (DOrtEvalStep)Marshal.GetDelegateForFunctionPointer(trainingApi_.EvalStep, typeof(DOrtEvalStep));
                    OrtSetLearningRate = (DOrtSetLearningRate)Marshal.GetDelegateForFunctionPointer(trainingApi_.SetLearningRate, typeof(DOrtSetLearningRate));
                    OrtGetLearningRate = (DOrtGetLearningRate)Marshal.GetDelegateForFunctionPointer(trainingApi_.GetLearningRate, typeof(DOrtGetLearningRate));
                    OrtOptimizerStep = (DOrtOptimizerStep)Marshal.GetDelegateForFunctionPointer(trainingApi_.OptimizerStep, typeof(DOrtOptimizerStep));
                    OrtRegisterLinearLRScheduler = (DOrtRegisterLinearLRScheduler)Marshal.GetDelegateForFunctionPointer(trainingApi_.RegisterLinearLRScheduler, typeof(DOrtRegisterLinearLRScheduler));
                    OrtSchedulerStep = (DOrtSchedulerStep)Marshal.GetDelegateForFunctionPointer(trainingApi_.SchedulerStep, typeof(DOrtSchedulerStep));
                    OrtGetParametersSize = (DOrtGetParametersSize)Marshal.GetDelegateForFunctionPointer(trainingApi_.GetParametersSize, typeof(DOrtGetParametersSize));
                    OrtCopyParametersToBuffer = (DOrtCopyParametersToBuffer)Marshal.GetDelegateForFunctionPointer(trainingApi_.CopyParametersToBuffer, typeof(DOrtCopyParametersToBuffer));
                    OrtCopyBufferToParameters = (DOrtCopyBufferToParameters)Marshal.GetDelegateForFunctionPointer(trainingApi_.CopyBufferToParameters, typeof(DOrtCopyBufferToParameters));
                    OrtReleaseTrainingSession = (DOrtReleaseTrainingSession)Marshal.GetDelegateForFunctionPointer(trainingApi_.ReleaseTrainingSession, typeof(DOrtReleaseTrainingSession));
                    OrtReleaseCheckpointState = (DOrtReleaseCheckpointState)Marshal.GetDelegateForFunctionPointer(trainingApi_.ReleaseCheckpointState, typeof(DOrtReleaseCheckpointState));
                    OrtExportModelForInferencing = (DOrtExportModelForInferencing)Marshal.GetDelegateForFunctionPointer(trainingApi_.ExportModelForInferencing, typeof(DOrtExportModelForInferencing));
                    OrtSetSeed = (DOrtSetSeed)Marshal.GetDelegateForFunctionPointer(trainingApi_.SetSeed, typeof(DOrtSetSeed));
                    OrtGetTrainingModelInputCount = (DOrtGetTrainingModelInputCount)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetTrainingModelInputCount, typeof(DOrtGetTrainingModelInputCount));
                    OrtGetEvalModelInputCount = (DOrtGetEvalModelInputCount)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetEvalModelInputCount, typeof(DOrtGetEvalModelInputCount));
                    OrtGetTrainingModelInputName = (DOrtGetTrainingModelInputName)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetTrainingModelInputName, typeof(DOrtGetTrainingModelInputName));
                    OrtGetEvalModelInputName = (DOrtGetEvalModelInputName)Marshal.GetDelegateForFunctionPointer(trainingApi_.TrainingSessionGetEvalModelInputName, typeof(DOrtGetEvalModelInputName));
                    OrtAddProperty = (DOrtAddProperty)Marshal.GetDelegateForFunctionPointer(trainingApi_.AddProperty, typeof(DOrtAddProperty));
                    OrtGetProperty = (DOrtGetProperty)Marshal.GetDelegateForFunctionPointer(trainingApi_.GetProperty, typeof(DOrtGetProperty));
                    OrtGetParameterTypeAndShape = (DOrtGetParameterTypeAndShape)Marshal.GetDelegateForFunctionPointer(trainingApi_.GetParameterTypeAndShape, typeof(DOrtGetParameterTypeAndShape));
                    OrtUpdateParameter = (DOrtUpdateParameter)Marshal.GetDelegateForFunctionPointer(trainingApi_.UpdateParameter, typeof(DOrtUpdateParameter));
                    OrtGetParameter = (DOrtGetParameter)Marshal.GetDelegateForFunctionPointer(trainingApi_.GetParameter, typeof(DOrtGetParameter));
                }

            }

    #region TrainingSession API

            /// <summary>
            /// Creates an instance of OrtSession with provided parameters
            /// </summary>
            /// <param name="checkpointPath">checkpoint string path</param>
            /// <param name="checkpointState">(Output) Loaded OrtCheckpointState instance</param>
            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtStatus* */DOrtLoadCheckpoint(
                                            byte[] checkpointPath,
                                            out IntPtr /* (OrtCheckpointState**) */ checkpointState);

            public static DOrtLoadCheckpoint OrtLoadCheckpoint;

            /// <summary>
            /// Creates an instance of OrtSession with provided parameters
            /// </summary>
            /// <param name="checkpointState">OrtCheckpointState instance to save</param>
            /// <param name="checkpointPath">Checkpoint string path</param>
            /// <param name="includeOptimizerState">Flag indicating whether to save the optimizer state.</param>
            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtStatus* */DOrtSaveCheckpoint(
                                            IntPtr /*(OrtCheckpointState*)*/ checkpointState,
                                            byte[] checkpointPath,
                                            bool includeOptimizerState);

            public static DOrtSaveCheckpoint OrtSaveCheckpoint;

            /// <summary>
            /// Creates an instance of OrtSession with provided parameters
            /// </summary>
            /// <param name="environment">Native OrtEnv instance</param>
            /// <param name="sessionOptions">Native SessionOptions instance</param>
            /// <param name="checkpointState">Loaded OrtCheckpointState instance</param>
            /// <param name="trainModelPath">model string path</param>
            /// <param name="evalModelPath">model string path</param>
            /// <param name="optimizerModelPath">model string path</param>
            /// <param name="session">(Output) Created native OrtTrainingSession instance</param>
            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtStatus* */DOrtCreateTrainingSession(
                                            IntPtr /* (OrtEnv*) */ environment,
                                            IntPtr /* (OrtSessionOptions*) */ sessionOptions,
                                            IntPtr /* (OrtCheckpointState*) */ checkpointState,
                                            byte[] trainModelPath,
                                            byte[] evalModelPath,
                                            byte[] optimizerModelPath,
                                            out IntPtr /* (OrtTrainingSession**) */ session);

            public static DOrtCreateTrainingSession OrtCreateTrainingSession;


            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTrainingModelOutputCount(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    out UIntPtr count);

            public static DOrtGetTrainingModelOutputCount OrtGetTrainingModelOutputCount;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetEvalModelOutputCount(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    out UIntPtr count);

            public static DOrtGetEvalModelOutputCount OrtGetEvalModelOutputCount;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTrainingModelOutputName(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    UIntPtr index,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out IntPtr /*(char**)*/name);

            public static DOrtGetTrainingModelOutputName OrtGetTrainingModelOutputName;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetEvalModelOutputName(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    UIntPtr index,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out IntPtr /*(char**)*/name);

            public static DOrtGetEvalModelOutputName OrtGetEvalModelOutputName;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtLazyResetGrad(
                                                    IntPtr /*(OrtTrainingSession*)*/ session);

            public static DOrtLazyResetGrad OrtLazyResetGrad;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtTrainStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                    UIntPtr inputCount,
                                                    IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                    UIntPtr outputCount,
                                                    IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                    );

            public static DOrtTrainStep OrtTrainStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtEvalStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                    UIntPtr inputCount,
                                                    IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                    UIntPtr outputCount,
                                                    IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                    );

            public static DOrtEvalStep OrtEvalStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtOptimizerStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions  // can be null to use the default options
                                                    );

            public static DOrtOptimizerStep OrtOptimizerStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtSetLearningRate(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    float learningRate
                                                    );

            public static DOrtSetLearningRate OrtSetLearningRate;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetLearningRate(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    out float learningRate
                                                    );

            public static DOrtGetLearningRate OrtGetLearningRate;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtRegisterLinearLRScheduler(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    long warmupStepCount,
                                                    long totalStepCount,
                                                    float learningRate
                                                    );
            public static DOrtRegisterLinearLRScheduler OrtRegisterLinearLRScheduler;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtSchedulerStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session
                                                    );
            public static DOrtSchedulerStep OrtSchedulerStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetParametersSize(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    out UIntPtr buffer_size,
                                                    bool only_trainable
                                                    );
            public static DOrtGetParametersSize OrtGetParametersSize;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtCopyParametersToBuffer(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtValue*)*/ buffer,
                                                    bool only_trainable
                                                    );
            public static DOrtCopyParametersToBuffer OrtCopyParametersToBuffer;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtCopyBufferToParameters(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtValue*)*/ buffer,
                                                    bool only_trainable
                                                    );
            public static DOrtCopyBufferToParameters OrtCopyBufferToParameters;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate void DOrtReleaseTrainingSession(IntPtr /*(OrtTrainingSession*)*/session);
            public static DOrtReleaseTrainingSession OrtReleaseTrainingSession;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate void DOrtReleaseCheckpointState(IntPtr /*(OrtCheckpointState*)*/checkpointState);
            public static DOrtReleaseCheckpointState OrtReleaseCheckpointState;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtExportModelForInferencing(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    byte[] inferenceModelPath,
                                                    UIntPtr graphOutputCount,
                                                    IntPtr[] /*(const char* const*)*/ graphOutputNames
                                                    );

            public static DOrtExportModelForInferencing OrtExportModelForInferencing;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSeed(
                                                    long seed
                                                    );

            public static DOrtSetSeed OrtSetSeed;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTrainingModelInputCount(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    out UIntPtr inputCount
                                                    );

            public static DOrtGetTrainingModelInputCount OrtGetTrainingModelInputCount;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetEvalModelInputCount(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    out UIntPtr inputCount
                                                    );

            public static DOrtGetEvalModelInputCount OrtGetEvalModelInputCount;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTrainingModelInputName(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    UIntPtr index,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out IntPtr /*(char**)*/name
                                                    );

            public static DOrtGetTrainingModelInputName OrtGetTrainingModelInputName;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetEvalModelInputName(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    UIntPtr index,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out IntPtr /*(char**)*/name
                                                    );

            public static DOrtGetEvalModelInputName OrtGetEvalModelInputName;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtAddProperty(
                                                    IntPtr /*(OrtCheckpointState*)*/ checkpointState,
                                                    byte[] /*(const char*)*/ propertyName,
                                                    CheckpointState.PropertyType propertyType,
                                                    IntPtr /*(const void*)*/ propertyValue
                                                    );

            public static DOrtAddProperty OrtAddProperty;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetProperty(
                                                    IntPtr /*(OrtCheckpointState*)*/ checkpointState,
                                                    byte[] /*(const char*)*/ propertyName,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out CheckpointState.PropertyType propertyType,
                                                    out IntPtr /*(const void**)*/ propertyValue
                                                    );

            public static DOrtGetProperty OrtGetProperty;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetParameterTypeAndShape(
                                                    IntPtr /*(OrtCheckpointState*)*/ checkpointState,
                                                    byte[] /*(const char*)*/ parameterName,
                                                    out IntPtr /*(OrtTensorTypeAndShapeInfo**)*/ parameterTypeAndShape
                                                    );

            public static DOrtGetParameterTypeAndShape OrtGetParameterTypeAndShape;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtUpdateParameter(
                                                    IntPtr /*(OrtCheckpointState*)*/ checkpointState,
                                                    byte[] /*(const char*)*/ parameterName,
                                                    IntPtr /*(OrtValue*)*/ parameter
                                                    );

            public static DOrtUpdateParameter OrtUpdateParameter;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(OrtStatus*)*/ DOrtGetParameter(
                                                    IntPtr /*(OrtCheckpointState*)*/ checkpointState,
                                                    byte[] /*(const char*)*/ parameterName,
                                                    IntPtr /*(OrtAllocator*)*/ allocator,
                                                    out IntPtr /*(OrtValue**)*/ parameter
                                                    );

            public static DOrtGetParameter OrtGetParameter;

    #endregion TrainingSession API

            public static bool TrainingEnabled()
            {
                if (trainingApiPtr == IntPtr.Zero)
                {
                    return false;
                }
                return true;
            }
        } //class NativeTrainingMethods
#endif
} //namespace
