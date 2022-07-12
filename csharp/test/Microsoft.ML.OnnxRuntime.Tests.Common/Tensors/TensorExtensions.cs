// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/tests/TensorExtensions.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;

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

            if (left.Dimensions[1] != right.Dimensions[0])
            {
                throw new ArgumentException($"{nameof(Tensor<T>)} {nameof(right)} must have first dimension of {left.Dimensions[1]}.", nameof(right));
            }

            return TensorOperations.Contract(left, right, s_oneArray, s_zeroArray);
        }
    }
}
