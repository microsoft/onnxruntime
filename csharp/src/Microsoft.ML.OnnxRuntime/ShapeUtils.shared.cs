using System.Diagnostics;
using System;

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    /// <summary>
    /// This class contains utilities for useful calculations with shape.
    /// </summary>
    public static class ShapeUtils
    {
        /// <summary>
        /// Returns a number of elements in the tensor from the given shape
        /// </summary>
        /// <param name="shape"></param>
        /// <returns>size</returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static long GetSizeForShape(ReadOnlySpan<long> shape)
        {
            long product = 1;
            foreach (var dim in shape)
            {
                if (dim < 0)
                {
                    throw new ArgumentOutOfRangeException($"Shape must not have negative elements: {dim}");
                }
                checked
                {
                    product *= dim;
                }
            }
            return product;
        }

        /// <summary>
        /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns>an array of strides</returns>
        public static long[] GetStrides(ReadOnlySpan<long> dimensions)
        {
            long[] strides = new long[dimensions.Length];

            if (dimensions.Length == 0)
            {
                return strides;
            }

            long stride = 1;
            for (int i = strides.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                if (dimensions[i] < 0)
                {
                    throw new ArgumentException($"Dimension {i} is negative");
                }
                stride *= dimensions[i];
            }

            return strides;
        }

        /// <summary>
        /// Calculates the 1-d index for n-d indices in layout specified by strides.
        /// </summary>
        /// <param name="strides">pre-calculated strides</param>
        /// <param name="indices">Indices. Must have the same length as strides</param>
        /// <param name="startFromDimension"></param>
        /// <returns>A 1-d index into the tensor buffer</returns>
        public static long GetIndex(ReadOnlySpan<long> strides, ReadOnlySpan<long> indices, int startFromDimension = 0)
        {
            Debug.Assert(strides.Length == indices.Length);

            long index = 0;
            for (int i = startFromDimension; i < indices.Length; i++)
            {
                index += strides[i] * indices[i];
            }

            return index;
        }
    }
}