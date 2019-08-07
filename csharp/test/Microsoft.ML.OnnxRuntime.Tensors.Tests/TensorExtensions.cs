// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    public static partial class TensorExtensions
    {
        private static int[] s_zeroArray = new[] { 0 };
        private static int[] s_oneArray = new[] { 1 };

        internal static Tensor<T> MatrixMultiply<T>(this Tensor<T> left, Tensor<T> right)
        {
            if (left.Rank != 2)
            {
                throw new InvalidOperationException($"{nameof(MatrixMultiply)} is only valid for a {nameof(Tensor<T>)} of {nameof(left.Rank)} 2.");
            }

            if (right.Rank != 2)
            {
                throw new ArgumentException($"{nameof(Tensor<T>)} {nameof(right)} must have {nameof(left.Rank)} 2.", nameof(right));
            }

            if (left.dimensions[1] != right.dimensions[0])
            {
                throw new ArgumentException($"{nameof(Tensor<T>)} {nameof(right)} must have first dimension of {left.dimensions[1]}.", nameof(right));
            }

            return TensorOperations.Contract(left, right, s_oneArray, s_zeroArray);
        }
    }
}
