﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
#if __ENABLE_TRAINING_APIS__
    /// <summary>
    ///  Holds the state of the training session.
    /// This class holds the entire training session state that includes model parameters, their gradients,
    /// optimizer parameters, and user properties. The TrainingSession leverages the CheckpointState
    /// by accessing and updating the contained training state.
    /// <note type="note">
    /// Note that the training session created with a checkpoint state uses this state to store the entire
    /// training state (including model parameters, its gradients, the optimizer states and the properties).
    /// The TrainingSession does not hold a copy of the CheckpointState and as a result, it is required
    /// that the checkpoint state outlives the lifetime of the training session.
    /// </note>
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
        }

        internal enum PropertyType : long
        {
            Int = 0,
            Float = 1,
            String = 2
        }

        private void AddPropertyImpl<T>(string propertyName, PropertyType propertyType, T propertyValue)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            T[] value = new T[1];
            value[0] = propertyValue;
            Memory<T> memory = value;
            using (var memHandle = memory.Pin())
            {
                IntPtr memPtr;
                unsafe
                {
                    memPtr = (IntPtr)memHandle.Pointer;
                }
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtAddProperty(handle, propertyNameUtf8, propertyType, memPtr));
            }
        }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Load a checkpoint state from a directory on disk into checkpoint_state.
        ///
        /// This function will parse a checkpoint directory, pull relevant files and load the training
        /// state into the checkpoint_state. This checkpoint state can then be used to create the
        /// training session by instantiating the TrainingSession. By doing so, the training
        /// session will begin or resume training from the given checkpoint state.
        /// </summary>
        /// <param name="checkpointPath"> Absolute path to the checkpoint directory.</param>
        /// <returns>CheckpointState object which holds the state of the training session parameters.</returns>
        public static CheckpointState LoadCheckpoint(string checkpointPath)
        {
            if (!NativeTrainingMethods.TrainingEnabled())
            {
                throw new InvalidOperationException("This package does not contain the training API. Please install the Microsoft.ML.OnnxRuntime.Training NuGet package.\n");
            }

            var envHandle = OrtEnv.Instance().Handle; // just so it is initialized
            IntPtr checkpointHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtLoadCheckpoint(NativeOnnxValueHelper.GetPlatformSerializedString(checkpointPath), out checkpointHandle));

            return new CheckpointState(checkpointHandle);
        }

        /// <summary>
        /// Save the given state to a checkpoint directory on disk.
        ///
        /// This function serializes the provided checkpoint state to a directory on disk.
        /// This checkpoint can later be loaded by invoking CheckpointState.LoadCheckpoint to begin or resume
        /// training from this snapshot of the state.
        /// </summary>
        /// <param name="state"> The checkpoint state to save.</param>
        /// <param name="checkpointPath"> Absolute path to the checkpoint directory.</param>
        /// <param name="includeOptimizerState"> Flag to indicate whether to save the optimizer state or not.</param>
        public static void SaveCheckpoint(CheckpointState state, string checkpointPath, bool includeOptimizerState = false)
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtSaveCheckpoint(state.Handle, NativeOnnxValueHelper.GetPlatformSerializedString(checkpointPath), includeOptimizerState));
        }

        /// <summary>
        /// Adds the given int property to the checkpoint state.
        ///
        /// Runtime properties that are ints such as epoch, training step, and others can be added to the checkpoint
        /// state by the user if they desire by calling this function with the appropriate property name and
        /// value. The given property name must be unique to be able to successfully add the property.
        /// </summary>
        /// <param name="propertyName">Unique name of the property being added.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        public void AddProperty(string propertyName, long propertyValue)
        {
            AddPropertyImpl(propertyName, PropertyType.Int, propertyValue);
        }

        /// <summary>
        /// Adds the given float property to the checkpoint state.
        ///
        /// Runtime properties that are floats such as loss, best score, and others can be added to the checkpoint
        /// state by the user if they desire by calling this function with the appropriate property name and
        /// value. The given property name must be unique to be able to successfully add the property.
        /// </summary>
        /// <param name="propertyName">Unique name of the property being added.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        public void AddProperty(string propertyName, float propertyValue)
        {
            AddPropertyImpl(propertyName, PropertyType.Float, propertyValue);
        }

        /// <summary>
        /// Adds the given string property to the checkpoint state.
        ///
        /// Runtime properties that are strings such as parameter names, custom strings, and others can be added
        /// to the checkpoint state by the user if they desire by calling this function with the appropriate property
        /// name and value. The given property name must be unique to be able to successfully add the property.
        /// </summary>
        /// <param name="propertyName">Unique name of the property being added.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        public void AddProperty(string propertyName, string propertyValue)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            var propertyValueUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyValue);

            IntPtr unmanagedPointer = Marshal.AllocHGlobal(propertyValueUtf8.Length);
            try
            {
                Marshal.Copy(propertyValueUtf8, 0, unmanagedPointer, propertyValueUtf8.Length);
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtAddProperty(handle, propertyNameUtf8, PropertyType.String, unmanagedPointer));
            }
            finally
            {
                Marshal.FreeHGlobal(unmanagedPointer);
            }
        }

        /// <summary>
        /// Gets the property value associated with the given name from the checkpoint state.
        ///
        /// Gets the property value from an existing entry in the checkpoint state. The property must
        /// exist in the checkpoint state to be able to retrieve it successfully.
        /// </summary>
        /// <param name="propertyName">Unique name of the property being retrieved.</param>
        /// <returns>Property value associated with the given property name.</returns>
        public object GetProperty(string propertyName)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr propertyValue = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetProperty(handle, propertyNameUtf8, allocator.Pointer, out PropertyType propertyType, out propertyValue));

            if (propertyType == PropertyType.Int)
            {
                var longPropertyValue = Marshal.ReadInt64(propertyValue);
                allocator.FreeMemory(propertyValue);
                return longPropertyValue;
            }
            else if (propertyType == PropertyType.Float)
            {
                float[] value = new float[1];
                Marshal.Copy(propertyValue, value, 0, 1);
                allocator.FreeMemory(propertyValue);
                return value[0];
            }
            else if (propertyType == PropertyType.String)
            {
                return NativeOnnxValueHelper.StringFromNativeUtf8(propertyValue, allocator);
            }

            throw new ArgumentException("Expected the property type to be one of long, float or string. Unknown type retrieved " + propertyValue.ToString());
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
