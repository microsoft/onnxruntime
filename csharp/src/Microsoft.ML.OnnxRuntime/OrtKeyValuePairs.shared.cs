// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


namespace Microsoft.ML.OnnxRuntime;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class OrtKeyValuePairs : SafeHandle
{
    private readonly bool _createdHandle;

    // cache the values here for convenience.
    // we could force a call to the C API every time in case something was changed in the background.
    private Dictionary<string, string> _keyValuePairs;

    public OrtKeyValuePairs()
        : base(IntPtr.Zero, ownsHandle: true)
    {
        NativeMethods.OrtCreateKeyValuePairs(out handle);
        _createdHandle = true;
        _keyValuePairs = new Dictionary<string, string>();
    }

    public OrtKeyValuePairs(IntPtr constHandle)
        : base(constHandle, ownsHandle: false)
    {
        _createdHandle = false;
        _keyValuePairs = GetLatest();
    }

    /// <summary>
    /// Cached key-value pair entries.
    /// Call Refresh() to update the values; 
    /// </summary>
    public IReadOnlyDictionary<string, string> Entries => _keyValuePairs;

    /// <summary>
    /// Overrides SafeHandle.IsInvalid
    /// </summary>
    /// <value>returns true if handle is equal to Zero</value>
    public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

    /// <summary>
    /// Adds a key-value pair to the key-value pairs instance.
    /// </summary>
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

    public void Refresh()
    {
        // refresh the cached values.
        _keyValuePairs = GetLatest();
    }

    /// <summary>
    /// Removes a key-value pair by key.
    /// </summary>
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

    #region SafeHandle
    /// <summary>
    /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
    /// the native instance of OrtKeyValuePairs.
    /// </summary>
    protected override bool ReleaseHandle()
    {
        if (_createdHandle)
        {
            NativeMethods.OrtReleaseKeyValuePairs(handle);
            handle = IntPtr.Zero;
        }

        return true;
    }
    #endregion
}

