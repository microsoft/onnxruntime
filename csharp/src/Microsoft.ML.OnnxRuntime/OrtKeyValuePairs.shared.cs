// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


namespace Microsoft.ML.OnnxRuntime
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Class to manage key-value pairs. 
    /// These are most often used for options and metadata.
    /// </summary>
    /// <see cref="OrtHardwareDevice.Metadata"/>
    /// <see cref="OrtEpDevice.EpMetadata"/>
    /// <see cref="OrtEpDevice.EpOptions"/>
    public class OrtKeyValuePairs : SafeHandle
    {
        private readonly bool _createdHandle;

        // cache the values here for convenience.
        // we could force a call to the C API every time in case something was changed in the background.
        private Dictionary<string, string> _keyValuePairs;

        /// <summary>
        /// Create a new OrtKeyValuePairs instance. 
        /// </summary>
        /// <remarks>
        /// A backing native instance is created and kept in sync with the C# content.
        /// </remarks>
        public OrtKeyValuePairs()
            : base(IntPtr.Zero, ownsHandle: true)
        {
            NativeMethods.OrtCreateKeyValuePairs(out handle);
            _createdHandle = true;
            _keyValuePairs = new Dictionary<string, string>();
        }

        /// <summary>
        /// Create a new OrtKeyValuePairs instance from an existing native OrtKeyValuePairs handle.
        /// </summary>
        /// <param name="constHandle">Native OrtKeyValuePairs handle.</param>
        /// <remarks>
        /// The instance is read-only, so calling Add or Remove will throw an InvalidOperationError.
        /// </remarks>
        internal OrtKeyValuePairs(IntPtr constHandle)
            : base(constHandle, ownsHandle: false)
        {
            _createdHandle = false;
            _keyValuePairs = GetLatest();
        }

        /// <summary>
        /// Create a new OrtKeyValuePairs instance from a dictionary.
        /// </summary>
        /// <param name="keyValuePairs">Key-value pairs to add.</param>
        /// <remarks>
        /// A backing native instance is created and kept in sync with the C# content.
        /// </remarks>
        public OrtKeyValuePairs(IReadOnlyDictionary<string, string> keyValuePairs)
            : base(IntPtr.Zero, ownsHandle: true)
        {
            NativeMethods.OrtCreateKeyValuePairs(out handle);
            _createdHandle = true;
            _keyValuePairs = new Dictionary<string, string>(keyValuePairs != null ? keyValuePairs.Count : 0);

            if (keyValuePairs != null && keyValuePairs.Count > 0)
            {
                foreach (var kvp in keyValuePairs)
                {
                    Add(kvp.Key, kvp.Value);
                }
            }
        }

        /// <summary>
        /// Current key-value pair entries.
        /// </summary>
        /// <remarks>
        /// Call Refresh() to update the cached values with the latest from the backing native instance.
        /// In general that should not be required as it's not expected an OrtKeyValuePairs instance would be 
        /// updated by both native and C# code.
        /// </remarks>
        public IReadOnlyDictionary<string, string> Entries => _keyValuePairs;

        /// <summary>
        /// Adds a key-value pair. Overrides any existing value for the key.
        /// </summary>
        /// <param name="key"> Key to add. Must not be null or empty.</param>
        /// <param name="value"> Value to add. May be empty. Must not be null.</param>
        public void Add(string key, string value)
        {
            if (!_createdHandle)
            {
                throw new InvalidOperationException($"{nameof(Add)} can only be called on instances you created.");
            }

            var keyPtr = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(key);
            var valuePtr = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(value);
            NativeMethods.OrtAddKeyValuePair(handle, keyPtr, valuePtr);
            _keyValuePairs[key] = value; // update the cached value
        }

        /// <summary>
        /// Update the cached values with the latest from the backing native instance as that is the source of truth.
        /// </summary>
        public void Refresh()
        {
            // refresh the cached values.
            _keyValuePairs = GetLatest();
        }

        /// <summary>
        /// Removes a key-value pair by key. Ignores keys that do not exist.
        /// </summary>
        /// <param name="key">Key to remove.</param>
        public void Remove(string key)
        {
            if (!_createdHandle)
            {
                throw new InvalidOperationException($"{nameof(Remove)} can only be called on instances you created.");
            }

            var keyPtr = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(key);
            NativeMethods.OrtRemoveKeyValuePair(handle, keyPtr);

            _keyValuePairs.Remove(key); // update the cached value
        }

        // for internal usage to pass into the call to OrtSessionOptionsAppendExecutionProvider_V2
        // from SessionOptions::AppendExecutionProvider
        internal void GetKeyValuePairHandles(out IntPtr keysHandle, out IntPtr valuesHandle, out UIntPtr numEntries)
        {
            if (IsInvalid)
            {
                throw new InvalidOperationException($"{nameof(GetKeyValuePairHandles)}: Invalid instance.");
            }

            NativeMethods.OrtGetKeyValuePairs(handle, out keysHandle, out valuesHandle, out numEntries);
        }

        /// <summary>
        /// Fetch all the key/value pairs to make sure we are in sync with the C API.
        /// </summary>
        private Dictionary<string, string> GetLatest()
        {
            var dict = new Dictionary<string, string>();
            if (IsInvalid)
            {
                return dict;
            }

            IntPtr keys, values;
            UIntPtr numEntries;
            NativeMethods.OrtGetKeyValuePairs(handle, out keys, out values, out numEntries);

            ulong count = numEntries.ToUInt64();
            int offset = 0;
            for (ulong i = 0; i < count; i++, offset += IntPtr.Size)
            {
                IntPtr keyPtr = Marshal.ReadIntPtr(keys, offset);
                IntPtr valuePtr = Marshal.ReadIntPtr(values, offset);
                var key = NativeOnnxValueHelper.StringFromNativeUtf8(keyPtr);
                var value = NativeOnnxValueHelper.StringFromNativeUtf8(valuePtr);
                dict.Add(key, value);
            }

            return dict;
        }

        /// <summary>
        /// Native handle to the OrtKeyValuePairs instance.
        /// </summary>
        internal IntPtr Handle => handle;

        /// <summary>
        /// Indicates whether the native handle is invalid.
        /// </summary>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Release the native instance of OrtKeyValuePairs if we own it.
        /// </summary>
        /// <returns>true</returns>
        protected override bool ReleaseHandle()
        {
            if (_createdHandle)
            {
                NativeMethods.OrtReleaseKeyValuePairs(handle);
                handle = IntPtr.Zero;
            }

            return true;
        }
    }
}