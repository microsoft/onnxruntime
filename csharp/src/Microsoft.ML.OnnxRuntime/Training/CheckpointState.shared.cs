// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
#if __ENABLE_TRAINING_APIS__
    /// <summary>
    ///  Holds the Checkpoint State as generated/consumed by on-device training APIs
    /// </summary>
    public class CheckpointState : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        /// <summary>
        /// Creates CheckpointState by loading state from path.
        /// <param name="checkpointPath"> absolute path to checkpoint file.</param>
        /// </summary>
        public CheckpointState(string checkpointPath)
            : base(IntPtr.Zero, true)
        {
            if (NativeTrainingMethods.TrainingEnabled())
            {
                var envHandle = OrtEnv.Handle; // just so it is initialized
                LoadCheckpoint(checkpointPath);
            }
            else
            {
                throw new InvalidOperationException("Training is disabled in the current build. Please build ONNXRuntime from source with the build flags enable_training_apis. \n");
            }
        }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Loads Checkpoint state from path
        /// </summary>
        /// <param name="checkpointPath"> absolute path to checkpoint</param>
        private void LoadCheckpoint(string checkpointPath)
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtLoadCheckpoint(NativeOnnxValueHelper.GetPlatformSerializedString(checkpointPath), out handle));
        }

#region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of CheckpointState
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeTrainingMethods.OrtReleaseCheckpointState(handle);
            handle = IntPtr.Zero;
            return true;
        }

#endregion
    }
#endif
}
