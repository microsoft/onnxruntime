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

        private CheckpointState(IntPtr checkpointHandle)
            : base(checkpointHandle, true)
        {
            var envHandle = OrtEnv.Instance().Handle; // just so it is initialized
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
        public static CheckpointState LoadCheckpoint(string checkpointPath)
        {
            if (!NativeTrainingMethods.TrainingEnabled())
            {
                throw new InvalidOperationException("Training is disabled in the current build. Please build ONNXRuntime from source with the build flag enable_training_apis.\n");
            }

            var envHandle = OrtEnv.Instance().Handle; // just so it is initialized
            IntPtr checkpointHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtLoadCheckpoint(NativeOnnxValueHelper.GetPlatformSerializedString(checkpointPath), out checkpointHandle));

            return new CheckpointState(checkpointHandle);
        }

        /// <summary>
        /// Saves the checkpoint
        /// <param name="checkpointPath"> absolute path to the checkpoint file.</param>
        /// <param name="includeOptimizerState"> absolute path to the checkpoint file.</param>
        /// </summary>
        public static void SaveCheckpoint(CheckpointState state, string checkpointPath, bool includeOptimizerState = false)
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtSaveCheckpoint(state.Handle, NativeOnnxValueHelper.GetPlatformSerializedString(checkpointPath), includeOptimizerState));
        }

        /// <summary>
        /// Adds the given int property to the checkpoint state.
        /// <param name="propertyName">Unique name of the property being added.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        /// </summary>
        public void AddProperty(string propertyName, long propertyValue)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            long[] value = new long[1];
            value[0] = propertyValue;
            Memory<long> memory = value;
            var memHandle = memory.Pin();
            try
            {
                IntPtr memPtr;
                unsafe
                {
                    memPtr = (IntPtr)memHandle.Pointer;
                }
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtAddProperty(handle, propertyNameUtf8, (long)0, memPtr));
            }
            catch (Exception)
            {
                memHandle.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Adds the given float property to the checkpoint state.
        /// <param name="propertyName">Unique name of the property being added.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        /// </summary>
        public void AddProperty(string propertyName, float propertyValue)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            float[] value = new float[1];
            value[0] = propertyValue;
            Memory<float> memory = value;
            var memHandle = memory.Pin();
            try
            {
                IntPtr memPtr;
                unsafe
                {
                    memPtr = (IntPtr)memHandle.Pointer;
                }
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtAddProperty(handle, propertyNameUtf8, (long)1, memPtr));
            }
            catch (Exception)
            {
                memHandle.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Adds the given string property to the checkpoint state.
        /// <param name="propertyName">Unique name of the property being added.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        /// </summary>
        public void AddProperty(string propertyName, string propertyValue)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            var propertyValueUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyValue);

            IntPtr unmanagedPointer = Marshal.AllocHGlobal(propertyValueUtf8.Length);
            Marshal.Copy(propertyValueUtf8, 0, unmanagedPointer, propertyValueUtf8.Length);
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtAddProperty(handle, propertyNameUtf8, /* (OrtPropertyType::OrtStringProperty) */(long)2, unmanagedPointer));
            Marshal.FreeHGlobal(unmanagedPointer);
        }

        /// <summary>
        /// Gets the property value associated with the given name from the checkpoint state.
        /// <param name="propertyName">Unique name of the property being retrieved.</param>
        /// </summary>
        public object GetProperty(string propertyName)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr propertyValue = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetProperty(handle, propertyNameUtf8, allocator.Pointer, out long propertyType, out propertyValue));

            if (propertyType == (long)0)
            {
                return Marshal.ReadInt64(propertyValue);
            }
            else if (propertyType == (long)1)
            {
                float[] value = new float[1];
                Marshal.Copy(propertyValue, value, 0, 1);
                return value[0];
            }
            else if (propertyType == (long)2)
            {
                return NativeOnnxValueHelper.StringFromNativeUtf8(propertyValue, allocator);
            }

            throw new ArgumentException("Expected the property type to be one of long, float or string. Unknown type retrieved.");
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
