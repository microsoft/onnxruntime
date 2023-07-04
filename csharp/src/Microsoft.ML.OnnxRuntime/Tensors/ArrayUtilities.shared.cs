// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/src/System/Numerics/Tensors/ArrayUtilities.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using System;

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    internal static class ArrayUtilities
    {
        public const int StackallocMax = 16;

        public static long GetSizeForShape(long[] shape)
        {
            long product = 1;
            foreach (var dim in shape)
            {
                if (dim < 0)
                {
                    throw new ArgumentOutOfRangeException("Shape must not have negative elements:" + dim);
                }
                product *= dim;
            }
            return product;
        }

        public static long GetProduct(ReadOnlySpan<int> dimensions, int startIndex = 0)
        {
            long product = 1;
            for (int i = startIndex; i < dimensions.Length; i++)
            {
                if (dimensions[i] < 0)
                {
                    throw new ArgumentOutOfRangeException($"{nameof(dimensions)}[{i}]");
                }

                // we use a long which should be much larger than is ever used here,
                // but still force checked
                checked
                {
                    product *= dimensions[i];
                }
            }

            return product;
        }

        public static bool IsAscending(ReadOnlySpan<int> values)
        {
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] < values[i - 1])
                {
                    return false;
                }
            }

            return true;
        }

        public static bool IsDescending(ReadOnlySpan<int> values)
        {
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > values[i - 1])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="reverseStride"></param>
        /// <returns></returns>
        public static int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride = false)
        {
            int[] strides = new int[dimensions.Length];

            if (dimensions.Length == 0)
            {
                return strides;
            }

            int stride = 1;
            if (reverseStride)
            {
                for (int i = 0; i < strides.Length; i++)
                {
                    strides[i] = stride;
                    stride *= dimensions[i];
                }
            }
            else
            {
                for (int i = strides.Length - 1; i >= 0; i--)
                {
                    strides[i] = stride;
                    stride *= dimensions[i];
                }
            }

            return strides;
        }

        public static void SplitStrides(int[] strides, int[] splitAxes, int[] newStrides, int stridesOffset, int[] splitStrides, int splitStridesOffset)
        {
            int newStrideIndex = 0;
            for (int i = 0; i < strides.Length; i++)
            {
                int stride = strides[i];
                bool isSplit = false;
                for (int j = 0; j < splitAxes.Length; j++)
                {
                    if (splitAxes[j] == i)
                    {
                        splitStrides[splitStridesOffset + j] = stride;
                        isSplit = true;
                        break;
                    }
                }

                if (!isSplit)
                {
                    newStrides[stridesOffset + newStrideIndex++] = stride;
                }
            }
        }

        /// <summary>
        /// Calculates the 1-d index for n-d indices in layout specified by strides.
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        /// <returns></returns>
        public static int GetIndex(int[] strides, ReadOnlySpan<int> indices, int startFromDimension = 0)
        {
            Debug.Assert(strides.Length == indices.Length);

            int index = 0;
            for (int i = startFromDimension; i < indices.Length; i++)
            {
                index += strides[i] * indices[i];
            }

            return index;
        }

        /// <summary>
        /// Calculates the n-d indices from the 1-d index in a layout specificed by strides
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="reverseStride"></param>
        /// <param name="index"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, int[] indices, int startFromDimension = 0)
        {
            Debug.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
            Debug.Assert(strides.Length == indices.Length);

            // scalar tensor - nothing to process
            if (indices.Length == 0)
            {
                return;
            }

            int remainder = index;
            for (int i = startFromDimension; i < strides.Length; i++)
            {
                // reverse the index for reverseStride so that we divide by largest stride first
                var nIndex = reverseStride ? strides.Length - 1 - i : i;

                var stride = strides[nIndex];
                indices[nIndex] = remainder / stride;
                remainder %= stride;
            }
        }

        /// <summary>
        /// Calculates the n-d indices from the 1-d index in a layout specificed by strides
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="reverseStride"></param>
        /// <param name="index"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, Span<int> indices, int startFromDimension = 0)
        {
            Debug.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
            Debug.Assert(strides.Length == indices.Length);

            // scalar tensor - nothing to process
            if (indices.Length == 0)
            {
                return;
            }

            int remainder = index;
            for (int i = startFromDimension; i < strides.Length; i++)
            {
                // reverse the index for reverseStride so that we divide by largest stride first
                var nIndex = reverseStride ? strides.Length - 1 - i : i;

                var stride = strides[nIndex];
                indices[nIndex] = remainder / stride;
                remainder %= stride;
            }
        }

        /// <summary>
        /// Takes an 1-d index over n-d sourceStrides and recalculates it assuming same n-d coordinates over a different n-d strides
        /// </summary>
        public static int TransformIndexByStrides(int index, int[] sourceStrides, bool sourceReverseStride, int[] transformStrides)
        {
            Debug.Assert(index >= 0);
            Debug.Assert(sourceReverseStride ? IsAscending(sourceStrides) : IsDescending(sourceStrides), "Index decomposition requires ordered strides");
            Debug.Assert(sourceStrides.Length == transformStrides.Length);

            // scalar tensor
            if (sourceStrides.Length == 0)
            {
                Debug.Assert(index == 0, "Index has to be zero for a scalar tensor");
                return 0;
            }

            int transformIndex = 0;
            int remainder = index;

            for (int i = 0; i < sourceStrides.Length; i++)
            {
                // reverse the index for reverseStride so that we divide by largest stride first
                var nIndex = sourceReverseStride ? sourceStrides.Length - 1 - i : i;

                var sourceStride = sourceStrides[nIndex];
                var transformStride = transformStrides[nIndex];

                transformIndex += transformStride * (remainder / sourceStride);
                remainder %= sourceStride;
            }

            return transformIndex;
        }

        public static T[] GetEmpty<T>()
        {
            // Match the implementation of Array.GetEmpty<T>()
            // from dotnet/runtime. Having it as a static in a
            // nested class ensures we only allocate the empty
            // array once and only when actually necessary.
            return EmptyArray<T>.Value;
        }
        private static class EmptyArray<T>
        {
            public static readonly T[] Value = new T[0];
        }
    }
}
