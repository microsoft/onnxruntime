// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/tests/TensorOperations.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    public static partial class TensorOperations
    {
        internal static void ValidateBinaryArgs<T>(Tensor<T> left, Tensor<T> right)
        {
            if (left.Rank != right.Rank || left.Length != right.Length)
            {
                throw new ArgumentException("Operands must have matching dimensions", nameof(right));
            }

            if (left.Rank == 0)
            {
                throw new ArgumentException($"Cannot operate on Tensor with {nameof(Tensor<T>.Rank)} of 0.", nameof(left));
            }

            for (int i = 0; i < left.Rank; i++)
            {
                if (left.dimensions[i] != right.dimensions[i])
                {
                    throw new ArgumentException("Operands must have matching dimensions", nameof(right));
                }
            }
        }

        internal static void ValidateBinaryArgs<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            if (left.Rank != right.Rank || left.Length != right.Length)
            {
                throw new ArgumentException("Operands must have matching dimensions", nameof(right));
            }

            if (left.Rank != result.Rank || left.Length != result.Length)
            {
                throw new ArgumentException("Operands must have matching dimensions", nameof(result));
            }

            if (left.Rank == 0)
            {
                throw new ArgumentException($"Cannot operate on Tensor with {nameof(Tensor<T>.Rank)} of 0.", nameof(left));
            }

            for (int i = 0; i < result.Rank; i++)
            {
                if (left.dimensions[i] != right.dimensions[i])
                {
                    throw new ArgumentException("Operands must have matching dimensions", nameof(right));
                }

                if (left.dimensions[i] != result.dimensions[i])
                {
                    throw new ArgumentException("Operands and result must have matching dimensions", nameof(result));
                }
            }
        }

        internal static void ValidateBinaryArgs<T>(Tensor<T> left, Tensor<T> right, Tensor<bool> result)
        {
            if (left.Rank != right.Rank || left.Length != right.Length)
            {
                throw new ArgumentException("Operands must have matching dimensions", nameof(right));
            }

            if (left.Rank != result.Rank || left.Length != result.Length)
            {
                throw new ArgumentException("Operands must have matching dimensions", nameof(result));
            }

            if (left.Rank == 0)
            {
                throw new ArgumentException($"Cannot operate on Tensor with {nameof(Tensor<T>.Rank)} of 0.", nameof(left));
            }

            for (int i = 0; i < result.Rank; i++)
            {
                if (left.dimensions[i] != right.dimensions[i])
                {
                    throw new ArgumentException("Operands must have matching dimensions", nameof(right));
                }

                if (left.dimensions[i] != result.dimensions[i])
                {
                    throw new ArgumentException("Operands and result must have matching dimensions", nameof(result));
                }
            }
        }

        internal static void ValidateArgs<T>(Tensor<T> tensor)
        {
            if (tensor.Rank == 0)
            {
                throw new ArgumentException($"Cannot operate on Tensor with {nameof(Tensor<T>.Rank)} of 0.", nameof(tensor));
            }
        }

        internal static void ValidateArgs<T>(Tensor<T> tensor, Tensor<T> result)
        {
            if (tensor.Rank != result.Rank || tensor.Length != result.Length)
            {
                throw new ArgumentException("Operands and result must have matching dimensions", nameof(result));
            }

            if (tensor.Rank == 0)
            {
                throw new ArgumentException($"Cannot operate on Tensor with {nameof(Tensor<T>.Rank)} of 0.", nameof(tensor));
            }

            for (int i = 0; i < result.Rank; i++)
            {
                if (tensor.dimensions[i] != result.dimensions[i])
                {
                    throw new ArgumentException("Operands and result must have matching dimensions", nameof(result));
                }
            }
        }

        internal static int[] ValidateContractArgs<T>(Tensor<T> left, Tensor<T> right, int[] leftAxes, int[] rightAxes)
        {
            if (leftAxes == null)
            {
                throw new ArgumentNullException(nameof(left));
            }

            if (rightAxes == null)
            {
                throw new ArgumentNullException(nameof(left));
            }

            if (leftAxes.Length != rightAxes.Length)
            {
                throw new ArgumentException($"{nameof(leftAxes)} and {nameof(rightAxes)} must have the same length, but were {leftAxes.Length} and {rightAxes.Length}, respectively.");
            }

            for (int i = 0; i < leftAxes.Length; i++)
            {
                var leftAxis = leftAxes[i];

                if (leftAxis >= left.Rank)
                {
                    throw new ArgumentOutOfRangeException($"{nameof(leftAxes)}[{i}] was set to axis index {leftAxis} which exceeds the Rank of {left}.");
                }

                var leftDimension = left.dimensions[leftAxis];

                var rightAxis = rightAxes[i];

                if (rightAxis >= right.Rank)
                {
                    throw new ArgumentOutOfRangeException($"{nameof(rightAxes)}[{i}] was set to axis index {rightAxis} which exceeds the Rank of {right}.");
                }

                var rightDimension = right.dimensions[rightAxis];

                if (leftDimension != rightDimension)
                {
                    throw new ArgumentOutOfRangeException($"Tensors may only be contracted on axes of the same length, but {nameof(leftAxes)} index {i} was length {leftDimension} and {nameof(rightAxes)} index {i} was length {rightDimension}.");
                }
            }

            var leftNonSummingDimensions = left.Rank - leftAxes.Length;
            var rightNonSummingDimensions = right.Rank - rightAxes.Length;
            var resultDimensions = new int[leftNonSummingDimensions + rightNonSummingDimensions];
            int dimensionsIndex = 0;

            Action<Tensor<T>, int[]> fillDimensions = (tensor, axes) =>
            {
                for (int i = 0; i < tensor.Rank; i++)
                {
                    var skip = false;
                    foreach (var contractionIndex in axes)
                    {
                        if (contractionIndex == i)
                        {
                            skip = true;
                            break;
                        }
                    }

                    if (!skip)
                    {
                        resultDimensions[dimensionsIndex++] = tensor.dimensions[i];
                    }
                }
            };

            fillDimensions(left, leftAxes);
            fillDimensions(right, rightAxes);

            return resultDimensions;
        }

        internal static int[] ValidateContractArgs<T>(Tensor<T> left, Tensor<T> right, int[] leftAxes, int[] rightAxes, Tensor<T> result)
        {
            var expectedDimensions = ValidateContractArgs(left, right, leftAxes, rightAxes);

            if (result.Rank != expectedDimensions.Length)
            {
                throw new ArgumentException($"{nameof(result)} should have {expectedDimensions.Length} dimensions but had {result.Rank}.");
            }

            for (int i = 0; i < expectedDimensions.Length; i++)
            {
                if (result.dimensions[i] != expectedDimensions[i])
                {
                    throw new ArgumentException($"{nameof(result)} dimension {i} should be {expectedDimensions[i]} but was {result.dimensions[i]}.");
                }
            }

            return expectedDimensions;
        }

        internal static void Add<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Add(left, right, result);
        }

        internal static Tensor<T> Add<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Add(left, right, result);

            return result;
        }

        internal static void Add<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Add(tensor, scalar, result);
        }

        internal static Tensor<T> Add<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Add(tensor, scalar, result);

            return result;
        }

        internal static void And<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.And(left, right, result);
        }

        internal static Tensor<T> And<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.And(left, right, result);

            return result;
        }

        internal static void And<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.And(tensor, scalar, result);
        }

        internal static Tensor<T> And<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.And(tensor, scalar, result);

            return result;
        }

        internal static void Contract<T>(Tensor<T> left, Tensor<T> right, int[] leftAxes, int[] rightAxes, Tensor<T> result)
        {
            var resultDimensions = ValidateContractArgs(left, right, leftAxes, rightAxes, result);

            TensorArithmetic<T>.Instance.Contract(left, right, leftAxes, rightAxes, result);
        }

        internal static Tensor<T> Contract<T>(Tensor<T> left, Tensor<T> right, int[] leftAxes, int[] rightAxes)
        {
            var resultDimensions = ValidateContractArgs(left, right, leftAxes, rightAxes);

            var result = left.CloneEmpty(resultDimensions);
            
            TensorArithmetic<T>.Instance.Contract(left, right, leftAxes, rightAxes, result);

            return result;
        }

        internal static void Decrement<T>(Tensor<T> tensor, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Decrement(tensor, result);
        }

        internal static Tensor<T> Decrement<T>(Tensor<T> tensor)
        {
            ValidateArgs(tensor);

            var result = tensor.Clone();
            
            TensorArithmetic<T>.Instance.Decrement(tensor, result);

            return result;
        }

        internal static void Divide<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Divide(left, right, result);
        }

        internal static Tensor<T> Divide<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Divide(left, right, result);

            return result;
        }

        internal static void Divide<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Divide(tensor, scalar, result);
        }

        internal static Tensor<T> Divide<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Divide(tensor, scalar, result);

            return result;
        }

        internal static void Equals<T>(Tensor<T> left, Tensor<T> right, Tensor<bool> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Equals(left, right, result);
        }

        internal static Tensor<bool> Equals<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty<bool>();
            
            TensorArithmetic<T>.Instance.Equals(left, right, result);

            return result;
        }

        internal static void GreaterThan<T>(Tensor<T> left, Tensor<T> right, Tensor<bool> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.GreaterThan(left, right, result);
        }

        internal static Tensor<bool> GreaterThan<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty<bool>();
            
            TensorArithmetic<T>.Instance.GreaterThan(left, right, result);

            return result;
        }

        internal static void GreaterThanOrEqual<T>(Tensor<T> left, Tensor<T> right, Tensor<bool> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.GreaterThanOrEqual(left, right, result);
        }

        internal static Tensor<bool> GreaterThanOrEqual<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty<bool>();
            
            TensorArithmetic<T>.Instance.GreaterThanOrEqual(left, right, result);

            return result;
        }

        internal static void Increment<T>(Tensor<T> tensor, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Increment(tensor, result);
        }

        internal static Tensor<T> Increment<T>(Tensor<T> tensor)
        {
            ValidateArgs(tensor);

            var result = tensor.Clone();
            
            TensorArithmetic<T>.Instance.Increment(tensor, result);

            return result;
        }

        internal static void LeftShift<T>(Tensor<T> tensor, int value, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.LeftShift(tensor, value, result);
        }

        internal static Tensor<T> LeftShift<T>(Tensor<T> tensor, int value)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.LeftShift(tensor, value, result);

            return result;
        }

        internal static void LessThan<T>(Tensor<T> left, Tensor<T> right, Tensor<bool> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.LessThan(left, right, result);
        }

        internal static Tensor<bool> LessThan<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty<bool>();
            
            TensorArithmetic<T>.Instance.LessThan(left, right, result);

            return result;
        }

        internal static void LessThanOrEqual<T>(Tensor<T> left, Tensor<T> right, Tensor<bool> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.LessThanOrEqual(left, right, result);
        }

        internal static Tensor<bool> LessThanOrEqual<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty<bool>();
            
            TensorArithmetic<T>.Instance.LessThanOrEqual(left, right, result);

            return result;
        }

        internal static void Modulo<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Modulo(left, right, result);
        }

        internal static Tensor<T> Modulo<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Modulo(left, right, result);

            return result;
        }

        internal static void Modulo<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Modulo(tensor, scalar, result);
        }

        internal static Tensor<T> Modulo<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Modulo(tensor, scalar, result);

            return result;
        }

        internal static void Multiply<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Multiply(left, right, result);
        }

        internal static Tensor<T> Multiply<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Multiply(left, right, result);

            return result;
        }

        internal static void Multiply<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Multiply(tensor, scalar, result);
        }

        internal static Tensor<T> Multiply<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Multiply(tensor, scalar, result);

            return result;
        }

        internal static void NotEquals<T>(Tensor<T> left, Tensor<T> right, Tensor<bool> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.NotEquals(left, right, result);
        }

        internal static Tensor<bool> NotEquals<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty<bool>();
            
            TensorArithmetic<T>.Instance.NotEquals(left, right, result);

            return result;
        }

        internal static void Or<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Or(left, right, result);
        }

        internal static Tensor<T> Or<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Or(left, right, result);

            return result;
        }

        internal static void Or<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Or(tensor, scalar, result);
        }

        internal static Tensor<T> Or<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Or(tensor, scalar, result);

            return result;
        }

        internal static void RightShift<T>(Tensor<T> tensor, int value, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.RightShift(tensor, value, result);
        }

        internal static Tensor<T> RightShift<T>(Tensor<T> tensor, int value)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.RightShift(tensor, value, result);

            return result;
        }

        internal static void Subtract<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Subtract(left, right, result);
        }

        internal static Tensor<T> Subtract<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Subtract(left, right, result);

            return result;
        }

        internal static void Subtract<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Subtract(tensor, scalar, result);
        }

        internal static Tensor<T> Subtract<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Subtract(tensor, scalar, result);

            return result;
        }

        internal static void UnaryMinus<T>(Tensor<T> tensor, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.UnaryMinus(tensor, result);
        }

        internal static Tensor<T> UnaryMinus<T>(Tensor<T> tensor)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.UnaryMinus(tensor, result);

            return result;
        }

        internal static void UnaryPlus<T>(Tensor<T> tensor, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.UnaryPlus(tensor, result);
        }

        internal static Tensor<T> UnaryPlus<T>(Tensor<T> tensor)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.UnaryPlus(tensor, result);

            return result;
        }

        internal static void Xor<T>(Tensor<T> left, Tensor<T> right, Tensor<T> result)
        {
            ValidateBinaryArgs(left, right, result);

            TensorArithmetic<T>.Instance.Xor(left, right, result);
        }

        internal static Tensor<T> Xor<T>(Tensor<T> left, Tensor<T> right)
        {
            ValidateBinaryArgs(left, right);

            var result = left.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Xor(left, right, result);

            return result;
        }

        internal static void Xor<T>(Tensor<T> tensor, T scalar, Tensor<T> result)
        {
            ValidateArgs(tensor, result);

            TensorArithmetic<T>.Instance.Xor(tensor, scalar, result);
        }

        internal static Tensor<T> Xor<T>(Tensor<T> tensor, T scalar)
        {
            ValidateArgs(tensor);

            var result = tensor.CloneEmpty();
            
            TensorArithmetic<T>.Instance.Xor(tensor, scalar, result);

            return result;
        }

    }
}
