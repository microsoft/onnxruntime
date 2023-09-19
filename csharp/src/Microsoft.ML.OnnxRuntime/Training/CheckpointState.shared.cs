// Copyright (c) Microsoft Corporation. All rights reserved.
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

        private void AddPropertyImpl<T>(string propertyName, PropertyType propertyType, T propertyValue) where T : unmanaged
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            T[] value = new T[1];
            value[0] = propertyValue;
            unsafe
            {
                fixed (T* memPtr = value)
                {
                    NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtAddProperty(handle, propertyNameUtf8, propertyType, (IntPtr)memPtr));
                }
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
        /// Adds or updates the given int property to/in the checkpoint state.
        ///
        /// Runtime properties such as epoch, training step, best score, and others can be added to the checkpoint
        /// state by the user by calling this function with the corresponding property name and value.
        /// The given property name must be unique to be able to successfully add the property.
        /// </summary>
        /// <param name="propertyName">Name of the property being added or updated.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        public void AddProperty(string propertyName, long propertyValue)
        {
            AddPropertyImpl(propertyName, PropertyType.Int, propertyValue);
        }

        /// <summary>
        /// Adds or updates the given float property to/in the checkpoint state.
        ///
        /// Runtime properties such as epoch, training step, best score, and others can be added to the checkpoint
        /// state by the user by calling this function with the corresponding property name and value.
        /// The given property name must be unique to be able to successfully add the property.
        /// </summary>
        /// <param name="propertyName">Name of the property being added or updated.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        public void AddProperty(string propertyName, float propertyValue)
        {
            AddPropertyImpl(propertyName, PropertyType.Float, propertyValue);
        }

        /// <summary>
        /// Adds or updates the given string property to/in the checkpoint state.
        ///
        /// Runtime properties such as epoch, training step, best score, and others can be added to the checkpoint
        /// state by the user by calling this function with the corresponding property name and value.
        /// The given property name must be unique to be able to successfully add the property.
        /// </summary>
        /// <param name="propertyName">Name of the property being added or updated.</param>
        /// <param name="propertyValue">Property value associated with the given name.</param>
        public void AddProperty(string propertyName, string propertyValue)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            var propertyValueUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyValue);

            unsafe
            {
                fixed (byte* p = propertyValueUtf8)
                {
                    NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtAddProperty(handle, propertyNameUtf8, PropertyType.String, (IntPtr)p));
                }
            }
        }

        /// <summary>
        /// Gets the property value associated with the given name from the checkpoint state.
        ///
        /// Gets the property value from an existing entry in the checkpoint state. The property must
        /// exist in the checkpoint state to be able to retrieve it successfully.
        /// </summary>
        /// <param name="propertyName">Name of the property being retrieved.</param>
        /// <returns>Property value associated with the given property name.</returns>
        public object GetProperty(string propertyName)
        {
            var propertyNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(propertyName);
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr propertyValue = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetProperty(handle, propertyNameUtf8, allocator.Pointer, out PropertyType propertyType, out propertyValue));

            if (propertyType == PropertyType.Int)
            {
                Int64 value;
                unsafe
                {
                    value = *(Int64*)propertyValue;
                }
                return value;
            }
            else if (propertyType == PropertyType.Float)
            {
                float value;
                unsafe
                {
                    value = *(float*)propertyValue;
                }
                return value;
            }
            else if (propertyType == PropertyType.String)
            {
                return NativeOnnxValueHelper.StringFromNativeUtf8(propertyValue, allocator);
            }

            try
            {
                throw new ArgumentException("Expected the property type to be one of long, float or string. Unknown type retrieved " + propertyValue.ToString());
            }
            finally
            {
                allocator.FreeMemory(propertyValue);
            }
        }

        /// <summary>
        /// Updates the data associated with the model parameter in the checkpoint state for the given parameter name.
        ///
        /// This function updates a model parameter in the checkpoint state with the given parameter data.
        /// The training session must be already created with the checkpoint state that contains the parameter
        /// being updated. The given parameter is copied over to the registered device for the training session.
        /// The parameter must exist in the checkpoint state to be able to update it successfully.
        /// </summary>
        /// <param name="parameterName">Name of the parameter being updated.</param>
        /// <param name="parameter">The parameter data that should replace the existing parameter data.</param>
        public void UpdateParameter(string parameterName, OrtValue parameter)
        {
            if (parameter.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new ArgumentException("Incorrect buffer received. Expected a tensor parameter.");
            }

            var parameterNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(parameterName);
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtUpdateParameter(handle, parameterNameUtf8, parameter.Handle));
        }

        /// <summary>
        /// Gets the data associated with the model parameter from the checkpoint state for the given parameter name.
        ///
        /// This function retrieves the model parameter data from the checkpoint state for the given parameter name.
        /// The parameter is copied over to the provided OrtValue. The training session must be already created
        /// with the checkpoint state that contains the parameter being retrieved.
        /// The parameter must exist in the checkpoint state to be able to retrieve it successfully.
        /// </summary>
        /// <param name="parameterName">Name of the parameter being updated.</param>
        /// <returns>The parameter data that is retrieved from the checkpoint state.</returns>
        public OrtValue GetParameter(string parameterName)
        {
            var parameterNameUtf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(parameterName);

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParameterTypeAndShape(handle, parameterNameUtf8, out IntPtr typeAndShapeInfoHandle));

            try
            {
                var typeAndShapeInfo = new OrtTensorTypeAndShapeInfo(typeAndShapeInfoHandle);
                var parameter = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, typeAndShapeInfo.ElementDataType, typeAndShapeInfo.Shape);

                try
                {
                    NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParameter(handle, parameterNameUtf8, parameter.Handle));
                }
                catch (OnnxRuntimeException e)
                {
                    parameter.Dispose();
                    throw e;
                }

                return parameter;
            }
            finally
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShapeInfoHandle);
            }

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
