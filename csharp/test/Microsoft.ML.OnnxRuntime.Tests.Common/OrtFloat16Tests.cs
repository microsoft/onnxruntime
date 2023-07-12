// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    [Collection("Ort Float16 tests")]
    public class OrtFloat16Tests
    {
        const float oneThird = 1 / 3.0f;
        const float oneSeventh = 1 / 7.0f;
        const float oneTenth = 1 / 10.0f;

        [Fact(DisplayName = "ConvertFloatToFloat16")]
        public void ConvertFloatToFloat16()
        {
            // Generate integer floats and insert between them
            // fractions.  This will test the rounding logic.
            float start = -10;

            var floatValues = new float[21 * 4];
            for (int i = 0; i < floatValues.Length; i += 4)
            {
                floatValues[i] = start;
                floatValues[i + 1] = start + oneThird;
                floatValues[i + 2] = start + oneSeventh;
                floatValues[i + 3] = start + oneTenth;
                start += 1;
            }

            var f16Converted = Array.ConvertAll(floatValues, f => (Float16)f);
            var backConverted = Array.ConvertAll(f16Converted, f16 => (float)f16);
            Assert.Equal(floatValues, backConverted, new FloatComparer());
        }

        [Fact(DisplayName = "TestZeros")]
        public void TestZeros()
        {
            var positiveZero = new Float16(0);
            Assert.False(Float16.IsNegative(positiveZero));
            Assert.True(Float16.IsNaNOrZero(positiveZero));

            float singlePositiveZero = (float)positiveZero;
            Assert.Equal(+0.0f, singlePositiveZero);
#if NET6_0_OR_GREATER
            Assert.False(float.IsNegative(singlePositiveZero));
#endif

            var negativeZero = Float16.Negate(positiveZero);
            Assert.True(Float16.IsNegative(negativeZero));
            Assert.True(Float16.IsNaNOrZero(negativeZero));

            float singleNegativeZero = (float)negativeZero;
            Assert.Equal(-0.0f, singleNegativeZero);
#if NET6_0_OR_GREATER
            Assert.True(float.IsNegative(singleNegativeZero));
#endif
        }

        [Fact(DisplayName = "TestComparisonOperators")]
        public void TestComparisonOperators()
        {
            Float16 left = (Float16)(float)-33.33f;
            Float16 leftSame = (Float16)(float)-33.33f;
            Float16 right = (Float16)(float)66.66f;
            Float16 rightSame = (Float16)(float)66.66f;

            Assert.False(Float16.IsNaNOrZero(left));
            Assert.False(Float16.IsNaNOrZero(right));

            Assert.True(right > Float16.Epsilon);

            Assert.True(left == leftSame);
            Assert.False(left == Float16.Negate(leftSame));

            Assert.True(right == rightSame);
            Assert.False(right == Float16.Negate(rightSame));

            Assert.True(left < right);
            Assert.True(left > Float16.Negate(right));
            Assert.True(Float16.Negate(left) < right);

            Assert.True(left <= right);
            Assert.True(left >= Float16.Negate(right));
            Assert.False(left > right);
            Assert.False(left >= right);
            Assert.True(Float16.Negate(left) <= right);
            Assert.False(left == right);
            Assert.False(right == left);
            Assert.True(left != right);
            Assert.True(right != left);
        }

        [Fact(DisplayName = "TestNAN")]
        public void TestNAN()
        {
            Float16 fp16NANFromSingle = (Float16)float.NaN;
            Assert.True(Float16.IsNaN(fp16NANFromSingle));
            Assert.Equal(Float16.NaN, fp16NANFromSingle);
            Assert.True(Float16.IsNaNOrZero(fp16NANFromSingle));

            float NanFromFloat16 = fp16NANFromSingle.ToFloat();
            Assert.True(float.IsNaN(NanFromFloat16));

            // IEqualityComparable returns true, because it tests
            // objects, not numbers.
            Assert.Equal(fp16NANFromSingle, Float16.NaN);

            Assert.Equal(Float16.NaN, Float16.Negate(Float16.NaN));
        }

        [Fact(DisplayName = "TestNANComparision")]
        public void TestNANComparisionOperators()
        {
            // NaN is not ordered with respect to anything
            // including itself

            // IEqualityComparable returns true, because it tests
            // objects, not numbers.
            Assert.Equal(Float16.NaN, Float16.NaN);
            Assert.False(Float16.NaN < Float16.NaN);
            Assert.False(Float16.NaN > Float16.NaN);
            Assert.False(Float16.NaN <= Float16.NaN);
            Assert.False(Float16.NaN >= Float16.NaN);
            Assert.False(Float16.NaN == Float16.NaN);

            // IEqualityComparable returns false, because it tests
            // objects, not numbers.
            Assert.NotEqual(Float16.NaN, Float16.MaxValue);

            Assert.False(Float16.NaN < Float16.MaxValue);
            Assert.False(Float16.MaxValue < Float16.NaN);
            Assert.False(Float16.NaN == Float16.MaxValue);
            Assert.False(Float16.MaxValue == Float16.NaN);
            Assert.False(Float16.NaN > Float16.MinValue);
            Assert.False(Float16.MaxValue > Float16.NaN);
            Assert.False(Float16.NaN == Float16.MinValue);
            Assert.False(Float16.MaxValue == Float16.NaN);
            Assert.True(Float16.MinValue < Float16.MaxValue);
        }

        [Fact(DisplayName = "TestInfinity")]
        public void TestInfinity()
        {
            Assert.False(Float16.IsInfinity(Float16.MinValue));
            Assert.False(Float16.IsInfinity(Float16.MaxValue));

            Float16 posInfinityFromSingle = (Float16)float.PositiveInfinity;
            Assert.True(Float16.IsPositiveInfinity(posInfinityFromSingle));
            Assert.Equal(Float16.PositiveInfinity, posInfinityFromSingle);
            Assert.False(Float16.IsFinite(posInfinityFromSingle));
            Assert.True(Float16.IsInfinity(posInfinityFromSingle));
            Assert.True(Float16.IsPositiveInfinity(posInfinityFromSingle));
            Assert.False(Float16.IsNegativeInfinity(posInfinityFromSingle));

            Assert.False(Float16.IsPositiveInfinity(Float16.MinValue));
            Assert.False(Float16.IsPositiveInfinity(Float16.MaxValue));


            Assert.Equal(float.PositiveInfinity < 0, Float16.IsNegative(posInfinityFromSingle));

            Float16 negInfinityFromSingle = (Float16)float.NegativeInfinity;
            Assert.True(Float16.IsNegativeInfinity(negInfinityFromSingle));
            Assert.Equal(Float16.NegativeInfinity, negInfinityFromSingle);
            Assert.False(Float16.IsFinite(negInfinityFromSingle));
            Assert.True(Float16.IsInfinity(negInfinityFromSingle));
            Assert.True(Float16.IsNegativeInfinity(negInfinityFromSingle));
            Assert.False(Float16.IsPositiveInfinity(negInfinityFromSingle));

            Assert.False(Float16.IsNegativeInfinity(Float16.MinValue));
            Assert.False(Float16.IsNegativeInfinity(Float16.MaxValue));


            Assert.Equal(float.NegativeInfinity < 0, Float16.IsNegative(negInfinityFromSingle));

            // Convert infinity to float and test the fact
            float infFromFloat16 = (float)Float16.PositiveInfinity;
            Assert.True(float.IsInfinity(infFromFloat16));
            Assert.True(float.IsPositiveInfinity(infFromFloat16));
        }


        [Fact(DisplayName = "TestNormalSubnormal")]
        public void TestNormalSubnormal()
        {
            Float16 fp16FromSingleMaxValue = (Float16)float.MaxValue;

            // Float MaxValue is outside Float16 range. This is different
            // from BFloat16 that retains sufficient range.
            Assert.True(Float16.IsInfinity(fp16FromSingleMaxValue));
            Assert.False(Float16.IsNormal(fp16FromSingleMaxValue));

            Assert.False(Float16.IsNormal(Float16.PositiveInfinity));
            Assert.True(Float16.IsNormal((Float16)45.6f));
            Assert.False(Float16.IsSubnormal((Float16)45.6f));

            Assert.False(Float16.IsSubnormal(fp16FromSingleMaxValue));
            Assert.False(Float16.IsSubnormal(Float16.PositiveInfinity));

            // 0b0_00000_0000000001 => 5.9604645E-08
            const ushort minSubnormalBits = 0x0001;
            const float smallestF16Subnormal = 5.9604645E-08f;
            Float16 smallestSubnormal = new Float16(minSubnormalBits);
            Assert.True(Float16.IsSubnormal(smallestSubnormal));
            Assert.False(Float16.IsNormal(smallestSubnormal));

            // 0b0_00000_1111111111 => 6.09755516E-05
            const float largestF16Subnormal = 6.09755516E-05f;
            const ushort maxSubnormalBits = 0x03FF;
            Float16 largestSubnormal = new Float16(maxSubnormalBits);
            Assert.True(Float16.IsSubnormal(largestSubnormal));
            Assert.False(Float16.IsNormal(largestSubnormal));

            // Convert subnormal to float and see if we match
            float convertedFromSmallestSubnormal = (float)smallestSubnormal;
            Assert.Equal(smallestF16Subnormal, convertedFromSmallestSubnormal, 6);

            float convertedFromLargestSubnormal = (float)largestSubnormal;
            Assert.Equal(largestF16Subnormal, convertedFromLargestSubnormal, 6);
        }

        [Fact(DisplayName = "TestEqual")]
        public void TestEqual()
        {
            // Box it
            object obj_1 = Float16.MaxValue;
            object obj_2 = new Float16(Float16.MaxValue.value);
            Assert.True(obj_1.Equals(obj_2));


            Assert.NotEqual(0, obj_1.GetHashCode());
            Assert.Equal(obj_1.GetHashCode(), obj_2.GetHashCode());
            Assert.True(Float16.NaN.Equals(Float16.NaN));

            Float16 fp16Zero = (Float16)0.0f;
            const ushort ushortZero = 0;
            Float16 fp16FromUshortZero = (Float16)ushortZero;

            Assert.True(fp16Zero.Equals(fp16FromUshortZero));

            // Should have the same hash code constant
            Assert.Equal(fp16Zero.GetHashCode(), fp16FromUshortZero.GetHashCode());
            Assert.Equal(Float16.NaN.GetHashCode(), Float16.NaN.GetHashCode());
        }

        [Fact(DisplayName = "TestCompare")]
        public void TestCompare()
        {
            object objMaxValue = new Float16(Float16.MaxValue.value);
            Assert.Equal(0, Float16.MaxValue.CompareTo(objMaxValue));

            Float16 one = (Float16)1.0f;
            Assert.Equal(-1, Float16.MinValue.CompareTo(one));
            Assert.Equal(1, Float16.MaxValue.CompareTo(one));

            // one is bigger than NaN
            Assert.Equal(-1, Float16.NaN.CompareTo(one));
            // Two NaNs are equal according to CompareTo()
            Assert.Equal(0, Float16.NaN.CompareTo((Float16)float.NaN));
            Assert.Equal(1, one.CompareTo(Float16.NaN));

            // Compare to null
            Assert.Equal(1, one.CompareTo(null));

            // Make sure it throws
            var obj = new object();
            Assert.Throws<ArgumentException>(() => one.CompareTo(obj));
        }
    }

    [Collection("Ort BFloat16 tests")]
    public class OrtBFloat16Tests
    {
        const float oneThird = 1 / 3.0f;
        const float oneSeventh = 1 / 7.0f;
        const float oneTenth = 1 / 10.0f;

        [Fact(DisplayName = "ConvertFloatToBFloat16")]
        public void ConvertFloatToBFloat16()
        {
            // Generate integer floats and insert between them
            // fractions.  This will test the rounding logic.
            float start = -10;

            var floatValues = new float[21 * 4];
            for (int i = 0; i < floatValues.Length; i += 4)
            {
                floatValues[i] = start;
                floatValues[i + 1] = start + oneThird;
                floatValues[i + 2] = start + oneSeventh;
                floatValues[i + 3] = start + oneTenth;
                start += 1;
            }

            var f16Converted = Array.ConvertAll(floatValues, f => (BFloat16)f);
            var backConverted = Array.ConvertAll(f16Converted, f16 => (float)f16);
            Assert.Equal(floatValues, backConverted, new FloatComparer());
        }

        [Fact(DisplayName = "TestZeros")]
        public void TestZeros()
        {
            var positiveZero = new BFloat16(0);
            Assert.False(BFloat16.IsNegative(positiveZero));
            Assert.True(BFloat16.IsNaNOrZero(positiveZero));
            float singlePositiveZero = (float)positiveZero;
            Assert.Equal(+0.0f, singlePositiveZero);
#if NET6_0_OR_GREATER
            Assert.False(float.IsNegative(singlePositiveZero));
#endif

            var negativeZero = BFloat16.Negate(positiveZero);
            Assert.True(BFloat16.IsNegative(negativeZero));
            Assert.True(BFloat16.IsNaNOrZero(negativeZero));

            float singleNegativeZero = (float)negativeZero;
            Assert.Equal(-0.0f, singleNegativeZero);
#if NET6_0_OR_GREATER
            Assert.True(float.IsNegative(singleNegativeZero));
#endif
        }

        [Fact(DisplayName = "TestComparisonOperators")]
        public void TestComparisionOperators()
        {
            BFloat16 left = (BFloat16)(float)-33.33f;
            BFloat16 leftSame = (BFloat16)(float)-33.33f;
            BFloat16 right = (BFloat16)(float)66.66f;
            BFloat16 rightSame = (BFloat16)(float)66.66f;

            Assert.False(BFloat16.IsNaNOrZero(left));
            Assert.False(BFloat16.IsNaNOrZero(right));

            Assert.True(right > BFloat16.Epsilon);

            Assert.True(left == leftSame);
            Assert.False(left == BFloat16.Negate(leftSame));

            Assert.True(right == rightSame);
            Assert.False(right == BFloat16.Negate(rightSame));

            Assert.True(left < right);
            Assert.True(left > BFloat16.Negate(right));
            Assert.True(BFloat16.Negate(left) < right);

            Assert.True(left <= right);
            Assert.True(left >= BFloat16.Negate(right));
            Assert.False(left > right);
            Assert.False(left >= right);
            Assert.True(BFloat16.Negate(left) <= right);
            Assert.False(left == right);
            Assert.False(right == left);
            Assert.True(left != right);
            Assert.True(right != left);
        }

        [Fact(DisplayName = "TestNAN")]
        public void TestNAN()
        {
            BFloat16 fp16NANFromSingle = (BFloat16)float.NaN;
            Assert.True(BFloat16.IsNaN(fp16NANFromSingle));
            Assert.Equal(BFloat16.NaN, fp16NANFromSingle);
            Assert.True(BFloat16.IsNaNOrZero(fp16NANFromSingle));

            float NanFromBFloat16 = fp16NANFromSingle.ToFloat();
            Assert.True(float.IsNaN(NanFromBFloat16));

            // IEqualityComparable returns true, because it tests
            // objects, not numbers.
            Assert.Equal(fp16NANFromSingle, BFloat16.NaN);
            Assert.Equal(BFloat16.NaN, BFloat16.Negate(BFloat16.NaN));

            Assert.False(BFloat16.IsNaN(BFloat16.MaxValue));
        }

        [Fact(DisplayName = "TestNANComparision")]
        public void TestNANComparisionOperators()
        {
            // NaN is not ordered with respect to anything
            // including itself

            // IEqualityComparable returns true, because it tests
            // objects, not numbers.
            Assert.Equal(BFloat16.NaN, BFloat16.NaN);
            Assert.False(BFloat16.NaN < BFloat16.NaN);
            Assert.False(BFloat16.NaN > BFloat16.NaN);
            Assert.False(BFloat16.NaN <= BFloat16.NaN);
            Assert.False(BFloat16.NaN >= BFloat16.NaN);
            Assert.False(BFloat16.NaN == BFloat16.NaN);

            // IEqualityComparable returns false, because it tests
            // objects, not numbers.
            Assert.NotEqual(BFloat16.NaN, BFloat16.MaxValue);

            Assert.False(BFloat16.NaN < BFloat16.MaxValue);
            Assert.False(BFloat16.MaxValue < BFloat16.NaN);
            Assert.False(BFloat16.NaN == BFloat16.MaxValue);
            Assert.False(BFloat16.MaxValue == BFloat16.NaN);
            Assert.False(BFloat16.NaN > BFloat16.MinValue);
            Assert.False(BFloat16.MaxValue > BFloat16.NaN);
            Assert.False(BFloat16.NaN == BFloat16.MinValue);
            Assert.False(BFloat16.MaxValue == BFloat16.NaN);
            Assert.True(BFloat16.MinValue < BFloat16.MaxValue);
        }

        [Fact(DisplayName = "TestInfinity")]
        public void TestInfinity()
        {
            Assert.False(BFloat16.IsInfinity(BFloat16.MinValue));
            Assert.False(BFloat16.IsInfinity(BFloat16.MaxValue));

            BFloat16 posInfinityFromSingle = (BFloat16)float.PositiveInfinity;
            Assert.True(BFloat16.IsPositiveInfinity(posInfinityFromSingle));
            Assert.Equal(BFloat16.PositiveInfinity, posInfinityFromSingle);
            Assert.False(BFloat16.IsFinite(posInfinityFromSingle));
            Assert.True(BFloat16.IsInfinity(posInfinityFromSingle));
            Assert.True(BFloat16.IsPositiveInfinity(posInfinityFromSingle));
            Assert.False(BFloat16.IsNegativeInfinity(posInfinityFromSingle));

            Assert.False(BFloat16.IsPositiveInfinity(BFloat16.MinValue));
            Assert.False(BFloat16.IsPositiveInfinity(BFloat16.MaxValue));


            Assert.Equal(float.PositiveInfinity < 0, BFloat16.IsNegative(posInfinityFromSingle));

            BFloat16 negInfinityFromSingle = (BFloat16)float.NegativeInfinity;
            Assert.True(BFloat16.IsNegativeInfinity(negInfinityFromSingle));
            Assert.Equal(BFloat16.NegativeInfinity, negInfinityFromSingle);
            Assert.False(BFloat16.IsFinite(negInfinityFromSingle));
            Assert.True(BFloat16.IsInfinity(negInfinityFromSingle));
            Assert.True(BFloat16.IsNegativeInfinity(negInfinityFromSingle));
            Assert.False(BFloat16.IsPositiveInfinity(negInfinityFromSingle));

            Assert.False(BFloat16.IsNegativeInfinity(BFloat16.MinValue));
            Assert.False(BFloat16.IsNegativeInfinity(BFloat16.MaxValue));


            Assert.True(BFloat16.IsNegative(negInfinityFromSingle));

            // Convert infinity to float and test the fact
            float infFromBFloat16 = (float)BFloat16.PositiveInfinity;
            Assert.True(float.IsInfinity(infFromBFloat16));
            Assert.True(float.IsPositiveInfinity(infFromBFloat16));
        }

        [Fact(DisplayName = "TestNormalSubnormal")]
        public void TestNormalSubnormal()
        {
            BFloat16 fp16FromSingleMaxValue = (BFloat16)float.MaxValue;

            Assert.True(BFloat16.IsInfinity(fp16FromSingleMaxValue));
            Assert.False(BFloat16.IsNormal(fp16FromSingleMaxValue));


            Assert.False(BFloat16.IsNormal(BFloat16.PositiveInfinity));
            Assert.True(BFloat16.IsNormal((BFloat16)45.6f));
            Assert.False(BFloat16.IsSubnormal((BFloat16)45.6f));

            Assert.False(BFloat16.IsSubnormal(fp16FromSingleMaxValue));
            Assert.False(BFloat16.IsSubnormal(BFloat16.PositiveInfinity));

            // 0b0_0000_0000_000_0001
            const ushort minSubnormalBits = 0x0001;
            BFloat16 smallestSubnormal = new BFloat16(minSubnormalBits);
            Assert.True(BFloat16.IsSubnormal(smallestSubnormal));
            Assert.False(BFloat16.IsNormal(smallestSubnormal));
#if NET6_0_OR_GREATER
            float singleSmallestSubnormal = (float)smallestSubnormal;
            Assert.True(float.IsSubnormal(singleSmallestSubnormal));
#endif

            const ushort maxSubnormalBits = 0x007F; // 0b0_0000_0000_111_1111;
            BFloat16 largestSubnormal = new BFloat16(maxSubnormalBits);
            Assert.True(BFloat16.IsSubnormal(largestSubnormal));
            Assert.False(BFloat16.IsNormal(largestSubnormal));
#if NET6_0_OR_GREATER
            float singleLargestSubnornal = (float)largestSubnormal;
            Assert.True(float.IsSubnormal(singleLargestSubnornal));
#endif
        }

        [Fact(DisplayName = "TestEqual")]
        public void TestEqual()
        {
            // Box it
            object obj_1 = BFloat16.MaxValue;
            object obj_2 = new BFloat16(BFloat16.MaxValue.value);
            Assert.True(obj_1.Equals(obj_2));


            Assert.NotEqual(0, obj_1.GetHashCode());
            Assert.Equal(obj_1.GetHashCode(), obj_2.GetHashCode());
            Assert.True(BFloat16.NaN.Equals(BFloat16.NaN));

            BFloat16 fp16Zero = (BFloat16)0.0f;
            const ushort ushortZero = 0;
            BFloat16 fp16FromUshortZero = (BFloat16)ushortZero;

            Assert.True(fp16Zero.Equals(fp16FromUshortZero));

            // Should have the same hash code constant
            Assert.Equal(fp16Zero.GetHashCode(), fp16FromUshortZero.GetHashCode());
            Assert.Equal(BFloat16.NaN.GetHashCode(), BFloat16.NaN.GetHashCode());
        }

        [Fact(DisplayName = "TestCompare")]
        public void TestCompare()
        {
            object objMaxValue = new BFloat16(BFloat16.MaxValue.value);
            Assert.Equal(0, BFloat16.MaxValue.CompareTo(objMaxValue));

            BFloat16 one = (BFloat16)1.0f;
            Assert.Equal(-1, BFloat16.MinValue.CompareTo(one));
            Assert.Equal(1, BFloat16.MaxValue.CompareTo(one));

            // one is bigger than NaN
            Assert.Equal(-1, BFloat16.NaN.CompareTo(one));
            // Two NaNs are equal according to CompareTo()
            Assert.Equal(0, BFloat16.NaN.CompareTo((BFloat16)float.NaN));
            Assert.Equal(1, one.CompareTo(BFloat16.NaN));

            // Compare to null
            Assert.Equal(1, one.CompareTo(null));

            // Make sure it throws
            var obj = new object();
            Assert.Throws<ArgumentException>(() => one.CompareTo(obj));
        }
    }
}