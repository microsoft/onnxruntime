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
                api_ = (OrtApi)OrtGetApi(13 /*ORT_API_VERSION*/);

                OrtGetTrainingApi = (DOrtGetTrainingApi)Marshal.GetDelegateForFunctionPointer(api_.GetTrainingApi, typeof(DOrtGetTrainingApi));
                trainingApiPtr = OrtGetTrainingApi(13 /*ORT_API_VERSION*/);
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
                    OrtReleaseTrainingSession = (DOrtReleaseTrainingSession)Marshal.GetDelegateForFunctionPointer(trainingApi_.ReleaseTrainingSession, typeof(DOrtReleaseTrainingSession));
                    OrtReleaseCheckpointState = (DOrtReleaseCheckpointState)Marshal.GetDelegateForFunctionPointer(trainingApi_.ReleaseCheckpointState, typeof(DOrtReleaseCheckpointState));
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
            /// <param name="checkpointPath">checkpoint string path</param>
            /// <param name="checkpointState">(Output) Loaded OrtCheckpointState instance</param>
            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /* OrtStatus* */DOrtSaveCheckpoint(
                                            byte[] checkpointPath,
                                            IntPtr /*(OrtTrainingSession*)*/ session,
                                            bool saveOptimizerState);

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
            public delegate IntPtr /*(ONNStatus*)*/ DOrtTrainStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                    UIntPtr inputCount,
                                                    IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                    UIntPtr outputCount,
                                                    IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                    );

            public static DOrtTrainStep OrtTrainStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtEvalStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                    UIntPtr inputCount,
                                                    IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                    UIntPtr outputCount,
                                                    IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                    );

            public static DOrtEvalStep OrtEvalStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtOptimizerStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    IntPtr /*(OrtSessionRunOptions*)*/ runOptions  // can be null to use the default options
                                                    );

            public static DOrtOptimizerStep OrtOptimizerStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtSetLearningRate(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    float learningRate
                                                    );

            public static DOrtSetLearningRate OrtSetLearningRate;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtGetLearningRate(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    out float learningRate
                                                    );

            public static DOrtGetLearningRate OrtGetLearningRate;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtRegisterLinearLRScheduler(
                                                    IntPtr /*(OrtTrainingSession*)*/ session,
                                                    long warmupStepCount,
                                                    long totalStepCount,
                                                    float learningRate
                                                    );
            public static DOrtRegisterLinearLRScheduler OrtRegisterLinearLRScheduler;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate IntPtr /*(ONNStatus*)*/ DOrtSchedulerStep(
                                                    IntPtr /*(OrtTrainingSession*)*/ session
                                                    );
            public static DOrtSchedulerStep OrtSchedulerStep;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate void DOrtReleaseTrainingSession(IntPtr /*(OrtTrainingSession*)*/session);
            public static DOrtReleaseTrainingSession OrtReleaseTrainingSession;

            [UnmanagedFunctionPointer(CallingConvention.Winapi)]
            public delegate void DOrtReleaseCheckpointState(IntPtr /*(OrtCheckpointState*)*/checkpointState);
            public static DOrtReleaseCheckpointState OrtReleaseCheckpointState;

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
