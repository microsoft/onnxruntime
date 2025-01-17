// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Portions of this code are from System.Half struct dotnet runtime.
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    // Utilities class created to fill in the gaps
    // of functionality that is absent in BitConverter class in NETSTANDARD 2.0
    // as well as some Single precision bit constants.
    internal class BitOpsUtils
    {
        // Lifted from .NET source code internal code
        // Constants for Single precision format
        // https://source.dot.net/#System.Private.CoreLib/src/libraries/System.Private.CoreLib/src/System/Single.cs,dda909df0f8d2fd0
        internal const uint SingleBiasedExponentMask = 0x7F80_0000;
        internal const int SingleBiasedExponentShift = 23;

        internal const uint SingleSignMask = 0x8000_0000;
        internal const int SingleSignShift = 31;

        // Most significant significand bit
        internal const uint SingleMostSignificantSigBit = 0x400000;
        internal const uint SingleTrailingSignificandMask = 0x007F_FFFF;

        /// <summary>
        /// Required because BitOperations are not available in NETSTANDARD 2.0.
        /// There are more efficient ways with bit twiddling, but this one has clarity.
        /// </summary>
        /// <param name="num">value</param>
        /// <returns>number of leading zeros. Useful to compute log2 as well.</returns>
        internal static int LeadingZeroCount(uint num)
        {
            if (num == 0)
            {
                return 32;
            }

            int count = 0;
            while ((num & 0xF000_0000) == 0)
            {
                count += 4;
                num <<= 4;
            }

            while ((num & 0x8000_0000) == 0)
            {
                count += 1;
                num <<= 1;
            }
            return count;
        }

        /// <summary>
        /// Extracts single precision number bit representation as uint
        /// so its bits can be manipulated.
        ///
        /// This API is the reverse of UInt32BitsToSingle().
        ///
        /// </summary>
        /// <param name="single">float value</param>
        /// <returns></returns>
        internal static uint SingleToUInt32Bits(float single)
        {
            uint result;
            unsafe
            {
                Buffer.MemoryCopy(&single, &result, sizeof(uint), sizeof(uint));
            }
            return result;
        }

        /// <summary>
        /// Needed because BitConverter impl is not available until
        /// later versions. This API is the reverse of SingleToUInt32Bits().
        ///
        /// For the exact bit representation of float see IEEE 754 standard for single precision.
        ///
        /// </summary>
        /// <param name="singleBits">bit representation of float either obtained from
        /// SingleToUInt32Bits or assembled using bitwise operators</param>
        /// <returns></returns>
        internal static float UInt32BitsToSingle(uint singleBits)
        {
            float result;
            unsafe
            {
                Buffer.MemoryCopy(&singleBits, &result, sizeof(uint), sizeof(uint));
            }
            return result;
        }

        /// <summary>
        /// Converts single precision bits representation which can be obtained using
        /// SingleToUInt32Bits() or manually constructed according to IEEE 754 standard.
        ///
        /// </summary>
        /// <param name="singleBits">bits representation of a single precision number (float)</param>
        /// <returns></returns>
        internal static ushort SingleBitsToBFloat16Bits(uint singleBits)
        {
            if (!BitConverter.IsLittleEndian)
            {
                return (ushort)(singleBits & 0xFFFF);
            }
            else
            {
                return (ushort)(singleBits >> 16);
            }
        }

        /// <summary>
        /// Converts bfloat16 ushort bits representation to single precision bits which then in turn can be
        /// manipulated or converted to float using UInt32BitsToSingle()
        /// </summary>
        /// <param name="bfloatBits">ushort bits representation of bfloat16</param>
        /// <returns></returns>
        internal static uint BFloat16BitsToSingleBits(ushort bfloatBits)
        {
            if (!BitConverter.IsLittleEndian)
            {
                return bfloatBits;
            }
            else
            {
                return (uint)bfloatBits << 16;
            }
        }

        /// <summary>
        /// Creates float NaN with the given sign and fp16 significand shifted &lt;&lt; 54
        /// </summary>
        /// <param name="sign">true for negative</param>
        /// <param name="significand">should be shifted 54 bits left before calling the function
        /// so only 8 bits of significand remains</param>
        /// <returns></returns>
        internal static float CreateSingleNaN(bool sign, ulong significand)
        {
            // We need to set at least on bit in NaN significant
            const uint NaNBits = SingleBiasedExponentMask | SingleMostSignificantSigBit;

            uint signInt = (sign ? 1U : 0U) << SingleSignShift;
            uint sigInt = (uint)(significand >> 41);
            uint singleBits = signInt | NaNBits | sigInt;

            return UInt32BitsToSingle(singleBits);
        }

        /// <summary>
        /// Creates float from sign, exponent and significand
        /// </summary>
        /// <param name="sign">true if negative</param>
        /// <param name="exponent">exponent</param>
        /// <param name="significand">significand</param>
        /// <returns></returns>
        internal static float CreateSingle(bool sign, byte exponent, uint significand)
        {
            uint signInt = (sign ? 1U : 0U) << SingleSignShift;
            uint expInt = ((uint)exponent << SingleBiasedExponentShift) + significand;
            uint singleBits = signInt + expInt;

            return UInt32BitsToSingle(singleBits);
        }
    }


    /// <summary>
    /// This value type represents A Float16 value
    /// it is blittable as defined in https://docs.microsoft.com/en-us/dotnet/framework/interop/blittable-and-non-blittable-types
    /// and as such, represented the same way in managed and native memories. This means that arrays of this type
    /// do not have to be copied to be passed to native memory but simply pinned and read by native code. Thus,
    /// one can create a Tensor on top of an array of these structures and feed it directly to Onnxruntime library.
    /// Binary wise, it is the same as ushort[] (uint16_t in C++). However, we would like a separate type for type dispatching.
    ///
    /// The implementation is derived from
    /// https://source.dot.net/#System.Private.CoreLib/src/libraries/System.Private.CoreLib/src/System/Half.cs,7895d5942d33f974
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Float16 :
        IComparable,
        IComparable<Float16>,
        IEquatable<Float16>
    {
        internal const ushort SignMask = 0x8000;
        internal const int SignShift = 15;
        internal const byte ShiftedSignMask = SignMask >> SignShift;

        internal const ushort BiasedExponentMask = 0x7C00; // 0b0_111_1100_0000_0000;
        internal const int BiasedExponentShift = 10;
        internal const byte ShiftedBiasedExponentMask = BiasedExponentMask >> BiasedExponentShift;

        internal const ushort TrailingSignificandMask = 0x03FF; // 0b0_000_0011_1111_1111;

        internal const byte MinSign = 0;
        internal const byte MaxSign = 1;

        internal const byte MinBiasedExponent = 0x00;
        internal const byte MaxBiasedExponent = 0x1F;

        internal const byte ExponentBias = 15;

        internal const sbyte MinExponent = -14;
        internal const sbyte MaxExponent = +15;

        // Constants representing the private bit-representation for various default values

        private const ushort PositiveZeroBits = 0x0000;
        private const ushort NegativeZeroBits = 0x8000;

        private const ushort OneBits = 0x3C00;

        // Minimum positive normalized value. It is corresponding to numeric_limits<float16>::min() in C++.
        private const ushort EpsilonBits = 0x0400;

        private const ushort PositiveInfinityBits = 0x7C00;
        private const ushort NegativeInfinityBits = 0xFC00;

        private const ushort PositiveQNaNBits = 0x7E00;
        private const ushort NegativeQNaNBits = 0xFE00;

        private const ushort MinValueBits = 0xFBFF;
        private const ushort MaxValueBits = 0x7BFF;

        private const ushort PositiveOneBits = 0x3C00;
        private const ushort NegativeOneBits = 0xBC00;

        private const ushort EBits = 0x4170;
        private const ushort PiBits = 0x4248;
        private const ushort TauBits = 0x4648;

        // Well-defined and commonly used values

        /// <summary>
        /// Float16 Epsilon value
        /// </summary>
        public static Float16 Epsilon => new Float16(EpsilonBits);                        //  0.00006103515625

        /// <summary>
        /// Float16 Pi value
        /// </summary>
        public static Float16 Pi => new Float16(PiBits);                                 //  3.14159265358979323846

        /// <summary>
        /// Float16 Positive Infinity value
        /// </summary>
        public static Float16 PositiveInfinity => new Float16(PositiveInfinityBits);

        /// <summary>
        /// Float16 Negative Infinity value
        /// </summary>
        public static Float16 NegativeInfinity => new Float16(NegativeInfinityBits);

        /// <summary>
        /// Float16 NaN
        /// </summary>
        public static Float16 NaN => new Float16(NegativeQNaNBits);                       // Same as System.Half.NaN

        /// <summary>
        /// Float16 Zero value
        /// </summary>
        public static Float16 Zero => new Float16(PositiveZeroBits);                      //  0.0

        /// <summary>
        /// Float16 One value
        /// </summary>
        public static Float16 One => new Float16(OneBits);                                //  1.0

        /// <summary>
        /// Float16 Negative Zero value
        /// </summary>
        public static Float16 NegativeZero => new Float16(NegativeZeroBits);              // -0.0

        /// <summary>
        /// Float16 Lowest value
        /// </summary>
        public static Float16 MinValue => new Float16(MinValueBits);                      // -65504.0

        /// <summary>
        /// Float16 Max value
        /// </summary>
        public static Float16 MaxValue => new Float16(MaxValueBits);                      // 65504.0

        /// <summary>
        /// float16 representation bits
        /// </summary>
        public readonly ushort value;

        /// <summary>
        /// Ctor from ushort bits, no conversion is done
        /// </summary>
        /// <param name="v"></param>
        public Float16(ushort v)
        {
            value = v;
        }

        private Float16(bool sign, ushort exp, ushort sig) =>
            value = (ushort)(((sign ? 1 : 0) << SignShift) + (exp << BiasedExponentShift) + sig);

        internal byte BiasedExponent
        {
            get
            {
                ushort bits = value;
                return ExtractBiasedExponentFromBits(bits);
            }
        }

        internal sbyte Exponent
        {
            get
            {
                return (sbyte)(BiasedExponent - ExponentBias);
            }
        }

        internal ushort Significand
        {
            get
            {
                return (ushort)(TrailingSignificand | ((BiasedExponent != 0) ? (1U << BiasedExponentShift) : 0U));
            }
        }

        internal ushort TrailingSignificand
        {
            get
            {
                ushort bits = value;
                return ExtractTrailingSignificandFromBits(bits);
            }
        }

        internal static byte ExtractBiasedExponentFromBits(ushort bits)
        {
            return (byte)((bits >> BiasedExponentShift) & ShiftedBiasedExponentMask);
        }

        internal static ushort ExtractTrailingSignificandFromBits(ushort bits)
        {
            return (ushort)(bits & TrailingSignificandMask);
        }

        /// <summary>
        /// Compares values of two Float16
        ///
        /// </summary>
        /// <param name="left">left hand side</param>
        /// <param name="right">right hand side</param>
        /// <returns>returns true if left is less than right according to IEEE</returns>
        public static bool operator <(Float16 left, Float16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is unordered with respect to everything, including itself.
                return false;
            }

            bool leftIsNegative = IsNegative(left);

            if (leftIsNegative != IsNegative(right))
            {
                // When the signs of left and right differ, we know that left is less than right if it is
                // the negative value. The exception to this is if both values are zero, in which case IEEE
                // says they should be equal, even if the signs differ.
                return leftIsNegative && !AreZero(left, right);
            }

            return (left.value != right.value) && ((left.value < right.value) ^ leftIsNegative);
        }

        /// <summary>
        /// Compares values of two Float16
        ///
        /// </summary>
        /// <param name="left">left hand side</param>
        /// <param name="right">right hand side</param>
        /// <returns>returns true if left is greater than right according to IEEE</returns>
        public static bool operator >(Float16 left, Float16 right)
        {
            return right < left;
        }

        /// <summary>
        /// Compares values of two Float16
        ///
        /// </summary>
        /// <param name="left">left hand side</param>
        /// <param name="right">right hand side</param>
        /// <returns>returns true if left is less or equal than right according to IEEE</returns>
        public static bool operator <=(Float16 left, Float16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is unordered with respect to everything, including itself.
                return false;
            }

            bool leftIsNegative = IsNegative(left);

            if (leftIsNegative != IsNegative(right))
            {
                // When the signs of left and right differ, we know that left is less than right if it is
                // the negative value. The exception to this is if both values are zero, in which case IEEE
                // says they should be equal, even if the signs differ.
                return leftIsNegative || AreZero(left, right);
            }

            return (left.value == right.value) || ((left.value < right.value) ^ leftIsNegative);
        }

        /// <summary>
        /// Compares values of two Float16
        /// </summary>
        /// <param name="left">left hand side</param>
        /// <param name="right">right hand side</param>
        /// <returns>returns true if left is greater or equal than right according to IEEE</returns>
        /// <inheritdoc />
        public static bool operator >=(Float16 left, Float16 right)
        {
            return right <= left;
        }

        /// <summary>
        /// Compares values of two Float16 for binary equality.
        /// If either of the values is NaN, this will return false.
        ///
        /// </summary>
        /// <param name="left">left hand side</param>
        /// <param name="right">right hand side</param>
        /// <returns>true if values are equal according to IEEE</returns>
        public static bool operator ==(Float16 left, Float16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is not equal to anything, including itself.
                return false;
            }

            return left.value == right.value;
        }

        /// <summary>
        /// Compares values of two Float16 for binary inequality
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if values are not equal according to IEEE</returns>
        public static bool operator !=(Float16 left, Float16 right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Determines whether the specified value is finite (zero, subnormal, or normal).
        /// </summary>
        /// <param name="value">Float16 instance.</param>
        /// <returns>true if the value is finite</returns>
        public static bool IsFinite(Float16 value)
        {
            return StripSign(value) < PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is infinite.
        /// </summary>
        /// <param name="value">Float16 instance.</param>
        /// <returns>true if the value is infinite</returns>
        public static bool IsInfinity(Float16 value)
        {
            return StripSign(value) == PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is NaN.
        /// </summary>
        ///
        /// <param name="value">Float16 instance</param>
        /// <returns>true if the value is not a number</returns>
        public static bool IsNaN(Float16 value)
        {
            return StripSign(value) > PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is negative.
        /// </summary>
        /// <param name="value">Float16 instance</param>
        /// <returns>true if the value is negative</returns>
        public static bool IsNegative(Float16 value)
        {
            return (short)(value.value) < 0;
        }

        /// <summary>
        /// Determines whether the specified value is negative infinity.
        /// </summary>
        ///
        /// <param name="value">Float16 instance</param>
        /// <returns>true if the value is negative infinity</returns>
        public static bool IsNegativeInfinity(Float16 value)
        {
            return value.value == NegativeInfinityBits;
        }


        /// <summary>
        /// Determines whether the specified value is normal
        /// </summary>
        /// <param name="value"></param>
        /// <returns>true or false</returns>
        public static bool IsNormal(Float16 value)
        {
            uint absValue = StripSign(value);
            return (absValue < PositiveInfinityBits)    // is finite
                && (absValue != 0)                      // is not zero
                && ((absValue & BiasedExponentMask) != 0);    // is not subnormal (has a non-zero exponent)
        }


        /// <summary>
        /// Determines whether the specified value is positive infinity.
        /// </summary>
        /// <param name="value">Float16 instance</param>
        /// <returns></returns>
        public static bool IsPositiveInfinity(Float16 value)
        {
            return value.value == PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is subnormal.
        /// </summary>
        /// <param name="value">Float16 instance</param>
        /// <returns>true if the value is subnormal</returns>
        public static bool IsSubnormal(Float16 value)
        {
            uint absValue = StripSign(value);
            return (absValue < PositiveInfinityBits)    // is finite
                && (absValue != 0)                      // is not zero
                && ((absValue & BiasedExponentMask) == 0);    // is subnormal (has a zero exponent)
        }

        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        ///
        /// <param name="obj">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="obj"/>,
        /// zero if this is equal to <paramref name="obj"/>, or a value greater than zero
        /// if this is greater than <paramref name="obj"/>.
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="obj"/> is not of type <see cref="Float16"/>.</exception>
        public int CompareTo(object obj)
        {
            if (!(obj is Float16))
            {
                return (obj is null) ? 1 : throw new ArgumentException("Object must be of type Float16");
            }
            return CompareTo((Float16)(obj));
        }

        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        /// <param name="other">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="other"/>,
        /// zero if this is equal to <paramref name="other"/>,
        /// or a value greater than zero if this is greater than <paramref name="other"/>.</returns>
        public int CompareTo(Float16 other)
        {
            if (this < other)
            {
                return -1;
            }

            if (this > other)
            {
                return 1;
            }

            if (this == other)
            {
                return 0;
            }

            if (IsNaN(this))
            {
                return IsNaN(other) ? 0 : -1;
            }

            Debug.Assert(IsNaN(other));
            return 1;
        }

        /// <summary>
        /// Returns a value indicating whether this instance and other Float16 represent the same value.
        /// </summary>
        /// <param name="other">A Float16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(Float16 other)
        {
            return value == other.value
                || AreZero(this, other)
                || (IsNaN(this) && IsNaN(other));
        }

        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
        /// <returns>true if obj is Float16 and its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            return (obj is Float16 other) && Equals(other);
        }

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            if (IsNaNOrZero(this))
            {
                // All NaNs should have the same hash code, as should both Zeros.
                return value & PositiveInfinityBits;
            }
            return value;
        }

        /// <summary>
        /// Returns a string representation of the current value.
        /// </summary>
        /// <returns>Text representation of Float16</returns>
        public override string ToString()
        {
            return $"{value} : {ToFloat()}";
        }

        /// <summary>
        /// Explicit conversion
        /// </summary>
        /// <returns>single precision value converted from Float16</returns>
        public float ToFloat()
        {
            return (float)this;
        }

        /// <summary>Explicitly converts a <see cref="float" /> value to its nearest representable half-precision floating-point value.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable half-precision floating-point value.</returns>
        public static explicit operator Float16(float value)
        {
            const int SingleMaxExponent = 0xFF;

            uint floatInt = BitOpsUtils.SingleToUInt32Bits(value);
            bool sign = (floatInt & BitOpsUtils.SingleSignMask) >> BitOpsUtils.SingleSignShift != 0;
            int exp = (int)(floatInt & BitOpsUtils.SingleBiasedExponentMask) >> BitOpsUtils.SingleBiasedExponentShift;
            uint sig = floatInt & BitOpsUtils.SingleTrailingSignificandMask;

            if (exp == SingleMaxExponent)
            {
                if (sig != 0) // NaN
                {
                    return CreateFloat16NaN(sign, (ulong)sig << 41); // Shift the significand bits to the left end
                }
                return sign ? NegativeInfinity : PositiveInfinity;
            }

            uint sigHalf = sig >> 9 | ((sig & 0x1FFU) != 0 ? 1U : 0U); // RightShiftJam

            if ((exp | (int)sigHalf) == 0)
            {
                return new Float16(sign, 0, 0);
            }

            return new Float16(RoundPackToFloat16(sign, (short)(exp - 0x71), (ushort)(sigHalf | 0x4000)));
        }

        /// <summary>Explicitly converts a half-precision floating-point value to its nearest representable <see cref="float" /> value.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable <see cref="float" /> value.</returns>
        public static explicit operator float(Float16 value)
        {
            bool sign = IsNegative(value);
            int exp = value.BiasedExponent;
            uint sig = value.TrailingSignificand;

            if (exp == MaxBiasedExponent)
            {
                if (sig != 0)
                {
                    // Shift sig left so only 8 bits of it remains
                    return BitOpsUtils.CreateSingleNaN(sign, (ulong)sig << 54);
                }
                return sign ? float.NegativeInfinity : float.PositiveInfinity;
            }

            if (exp == 0)
            {
                if (sig == 0)
                {
                    // Positive / Negative zero
                    return (sign) ? -0.0f : 0.0f;
                }
                (exp, sig) = NormSubnormalF16Sig(sig);
                exp -= 1;
            }

            return BitOpsUtils.CreateSingle(sign, (byte)(exp + 0x70), sig << 13);
        }

        /// <summary>
        /// Flips the sign. NaNs are not affected.
        /// IEEE 754 specifies NaNs to be propagated
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static Float16 Negate(Float16 value)
        {
            return IsNaN(value) ? value : new Float16((ushort)(value.value ^ SignMask));
        }

        #region Utilities

        private static bool AreZero(Float16 left, Float16 right)
        {
            // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
            // for two values by or'ing the private bits together and stripping the sign. They are both zero,
            // and therefore equivalent, if the resulting value is still zero.
            return (ushort)((left.value | right.value) & ~SignMask) == 0;
        }

        /// <summary>
        /// The function returns true if the value is either NaN or zero.
        /// </summary>
        /// <param name="value">instance of Float16</param>
        /// <returns>true if NaN or zero.</returns>
        public static bool IsNaNOrZero(Float16 value)
        {
            uint abs = StripSign(value);
            return (abs == 0 || abs > PositiveInfinityBits);
        }

        private static uint StripSign(Float16 value)
        {
            return (ushort)(value.value & ~SignMask);
        }

        private static (int Exp, uint Sig) NormSubnormalF16Sig(uint sig)
        {
            int shiftDist = BitOpsUtils.LeadingZeroCount(sig) - 16 - 5;
            return (1 - shiftDist, sig << shiftDist);
        }


        // Significand bits should be shifted towards to the left end before calling these methods
        // Creates Quiet NaN if significand == 0
        private static Float16 CreateFloat16NaN(bool sign, ulong significand)
        {
            const ushort NaNBits = BiasedExponentMask | 0x200; // Most significant significand bit

            uint signInt = (sign ? 1U : 0U) << SignShift;
            ushort sigInt = (ushort)(significand >> 54);

            ushort ushortBits = (ushort)(signInt | NaNBits | sigInt);
            return new Float16(ushortBits);
        }

        private static ushort RoundPackToFloat16(bool sign, short exp, ushort sig)
        {
            const int RoundIncrement = 0x8; // Depends on rounding mode but it's always towards closest / ties to even
            int roundBits = sig & 0xF;

            if ((uint)exp >= 0x1D)
            {
                if (exp < 0)
                {
                    sig = (ushort)ShiftRightJam(sig, -exp);
                    exp = 0;
                    roundBits = sig & 0xF;
                }
                else if (exp > 0x1D || sig + RoundIncrement >= 0x8000) // Overflow
                {
                    return sign ? NegativeInfinityBits : PositiveInfinityBits;
                }
            }

            sig = (ushort)((sig + RoundIncrement) >> 4);
            sig &= (ushort)~(((roundBits ^ 8) != 0 ? 0 : 1) & 1);

            if (sig == 0)
            {
                exp = 0;
            }

            return new Float16(sign, (ushort)exp, sig).value;
        }

        // If any bits are lost by shifting, "jam" them into the LSB.
        // if dist > bit count, Will be 1 or 0 depending on i
        // (unlike bitwise operators that masks the lower 5 bits)
        private static uint ShiftRightJam(uint i, int dist) => dist < 31 ? (i >> dist) | (i << (-dist & 31) != 0 ? 1U : 0U) : (i != 0 ? 1U : 0U);

        private static ulong ShiftRightJam(ulong l, int dist) => dist < 63 ? (l >> dist) | (l << (-dist & 63) != 0 ? 1UL : 0UL) : (l != 0 ? 1UL : 0UL);

        #endregion
    }

    /// <summary>
    /// This value type represents A BFloat16 value.
    /// See https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
    /// for details.
    /// it is blittable as defined in https://docs.microsoft.com/en-us/dotnet/framework/interop/blittable-and-non-blittable-types
    /// and as such, represented the same way in managed and native memories. This means that arrays of this type
    /// do not have to be copied to be passed to native memory but simply pinnned and read by native code. Thus,
    /// one can create a Tensor on top of an array of these structures and feed it directly to Onnxruntime library.
    /// Binary wise, it is the same as ushort[] (uint16_t in C++). However, we would like a separate type for type dispatching.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct BFloat16 :
        IComparable,
        IComparable<BFloat16>,
        IEquatable<BFloat16>
    {
        internal const ushort SignMask = 0x8000;
        internal const int SignShift = 15;
        internal const byte ShiftedSignMask = SignMask >> SignShift;

        internal const ushort BiasedExponentMask = 0x7F80; // 0b_0111_1111_1000_0000;
        internal const int BiasedExponentShift = 7;
        internal const byte ShiftedBiasedExponentMask = BiasedExponentMask >> BiasedExponentShift;

        internal const ushort TrailingSignificandMask = 0x007F; // 0b_0000_0000_0111_1111;

        internal const byte MinSign = 0;
        internal const byte MaxSign = 1;

        internal const byte MinBiasedExponent = 0x00;
        internal const byte MaxBiasedExponent = 0xFF;

        internal const byte ExponentBias = 127;

        internal const sbyte MinExponent = -126;
        internal const sbyte MaxExponent = +127;

        // Constants representing the private bit-representation for various default values

        private const ushort PositiveZeroBits = 0x0000;
        private const ushort NegativeZeroBits = 0x8000;

        private const ushort OneBits = 0x3F80;  // 0b0_01111111_0000000

        private const ushort PositiveInfinityBits = 0x7F80;
        private const ushort NegativeInfinityBits = 0xFF80;

        private const ushort PositiveQNaNBits = 0x7FC1;
        private const ushort NegativeQNaNBits = 0xFFC1;

        // Lowest finite value. It is corresponding to numeric_limits<BFloat16>::lowest() in C++.
        private const ushort MinValueBits = 0xFF7F; // 1b0_11111110_1111111

        private const ushort MaxValueBits = 0x7F7F; // 0b0_11111110_1111111

        // Minimum positive normalized value. It is corresponding to numeric_limits<BFloat16>::min() in C++.
        private const ushort EpsilonBits = 0x0080;

        private const ushort PiBits = 0x4049; // 0b0_10000000_1001001

        // Used for rounding subnormal values
        private const uint RoundingBase = 0x7FFF;

        // Well-defined and commonly used values

        /// <summary>
        /// BFloat16 Epsilon value
        /// </summary>
        public static BFloat16 Epsilon => new BFloat16(EpsilonBits);

        /// <summary>
        /// BFloat16 Pi value
        /// </summary>
        public static BFloat16 Pi => new BFloat16(PiBits);

        /// <summary>
        /// BFloat16 Positive infinity value
        /// </summary>
        public static BFloat16 PositiveInfinity => new BFloat16(PositiveInfinityBits);

        /// <summary>
        /// BFloat16 Negative infinity value
        /// </summary>
        public static BFloat16 NegativeInfinity => new BFloat16(NegativeInfinityBits);

        /// <summary>
        /// BFloat16 NaN
        /// </summary>
        public static BFloat16 NaN => new BFloat16(NegativeQNaNBits); // .Net has no BFloat16. Follow Float16 style.

        /// <summary>
        /// BFloat16 Positive Zero
        /// </summary>
        public static BFloat16 Zero => new BFloat16(PositiveZeroBits);  // 0.0

        /// <summary>
        /// BFloat16 One
        /// </summary>
        public static BFloat16 One => new BFloat16(OneBits);  // 1.0

        /// <summary>
        /// BFloat16 Negative Zero
        /// </summary>
        public static BFloat16 NegativeZero => new BFloat16(NegativeZeroBits);  // -0.0

        /// <summary>
        /// BFloat16 Min value
        /// </summary>
        public static BFloat16 MinValue => new BFloat16(MinValueBits);  // -3.38953139e38

        /// <summary>
        /// BFloat16 Max value
        /// </summary>

        public static BFloat16 MaxValue => new BFloat16(MaxValueBits); // 3.38953139e38

        /// <summary>
        /// bfloat16 representation bits
        /// </summary>
        public readonly ushort value;

        /// <summary>
        /// Constructor from ushort, no conversion takes place. The value
        /// is assumed to be converted
        /// </summary>
        /// <param name="v">bfloat16 representation bits</param>
        public BFloat16(ushort v)
        {
            value = v;
        }

        // Extracts biased exponent bits
        internal byte BiasedExponent
        {
            get
            {
                ushort bits = value;
                return ExtractBiasedExponentFromBits(bits);
            }
        }

        // Extracts all the Significand bits
        internal ushort TrailingSignificand
        {
            get
            {
                ushort bits = value;
                return ExtractTrailingSignificandFromBits(bits);
            }
        }

        internal static byte ExtractBiasedExponentFromBits(ushort bits)
        {
            return (byte)((bits >> BiasedExponentShift) & ShiftedBiasedExponentMask);
        }

        internal static ushort ExtractTrailingSignificandFromBits(ushort bits)
        {
            return (ushort)(bits & TrailingSignificandMask);
        }

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is less than right according to IEEE</returns>
        public static bool operator <(BFloat16 left, BFloat16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is unordered with respect to everything, including itself.
                return false;
            }

            bool leftIsNegative = IsNegative(left);

            if (leftIsNegative != IsNegative(right))
            {
                // When the signs of left and right differ, we know that left is less than right if it is
                // the negative value. The exception to this is if both values are zero, in which case IEEE
                // says they should be equal, even if the signs differ.
                return leftIsNegative && !AreZero(left, right);
            }

            return (left.value != right.value) && ((left.value < right.value) ^ leftIsNegative);
        }

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is greater than right according to IEEE</returns>
        public static bool operator >(BFloat16 left, BFloat16 right)
        {
            return right < left;
        }

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is less or equal than right according to IEEE</returns>
        public static bool operator <=(BFloat16 left, BFloat16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is unordered with respect to everything, including itself.
                return false;
            }

            bool leftIsNegative = IsNegative(left);

            if (leftIsNegative != IsNegative(right))
            {
                // When the signs of left and right differ, we know that left is less than right if it is
                // the negative value. The exception to this is if both values are zero, in which case IEEE
                // says they should be equal, even if the signs differ.
                return leftIsNegative || AreZero(left, right);
            }

            return (left.value == right.value) || ((left.value < right.value) ^ leftIsNegative);
        }

        /// <summary>
        /// Compares two BFloat16 instances.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>true if the left is greater or equal than right according to IEEE</returns>
        public static bool operator >=(BFloat16 left, BFloat16 right)
        {
            return right <= left;
        }

        /// <summary>
        /// Compares values of two BFloat16 for binary equality.
        /// If either of the values is NaN, this will return false.
        ///
        /// </summary>
        /// <param name="left">left hand side</param>
        /// <param name="right">right hand side</param>
        /// <returns>result of value comparisons</returns>
        public static bool operator ==(BFloat16 left, BFloat16 right)
        {
            if (IsNaN(left) || IsNaN(right))
            {
                // IEEE defines that NaN is not equal to anything, including itself.
                return false;
            }

            return left.value == right.value;
        }

        /// <summary>
        /// Compares values of two BFloat16 for binary inequality
        /// If either of the values is NaN it would return true.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator !=(BFloat16 left, BFloat16 right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Determines whether the specified value is finite (zero, subnormal, or normal).
        /// </summary>
        /// <param name="value">BFloat16 instance.</param>
        /// <returns>true if the value is finite</returns>
        public static bool IsFinite(BFloat16 value)
        {
            return StripSign(value) < PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is infinite.
        /// </summary>
        /// <param name="value">BFloat16 instance.</param>
        /// <returns>true if the value is infinite</returns>
        public static bool IsInfinity(BFloat16 value)
        {
            return StripSign(value) == PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is NaN.
        /// </summary>
        ///
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is not a number</returns>
        public static bool IsNaN(BFloat16 value)
        {
            return StripSign(value) > PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is negative.
        /// </summary>
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is negative</returns>
        public static bool IsNegative(BFloat16 value)
        {
            return (short)(value.value) < 0;
        }

        /// <summary>
        /// Determines whether the specified value is negative infinity.
        /// </summary>
        ///
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is negative infinity</returns>
        public static bool IsNegativeInfinity(BFloat16 value)
        {
            return value.value == NegativeInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is normal
        /// </summary>
        /// <param name="value"></param>
        /// <returns>true or false</returns>
        public static bool IsNormal(BFloat16 value)
        {
            uint absValue = StripSign(value);
            return (absValue < PositiveInfinityBits)    // is finite
                && (absValue != 0)                      // is not zero
                && ((absValue & BiasedExponentMask) != 0);    // is not subnormal (has a non-zero exponent)
        }

        /// <summary>
        /// Determines whether the specified value is positive infinity.
        /// </summary>
        /// <param name="value">BFloat16 instance</param>
        /// <returns></returns>
        public static bool IsPositiveInfinity(BFloat16 value)
        {
            return value.value == PositiveInfinityBits;
        }

        /// <summary>
        /// Determines whether the specified value is subnormal.
        /// </summary>
        /// <param name="value">BFloat16 instance</param>
        /// <returns>true if the value is subnormal</returns>
        public static bool IsSubnormal(BFloat16 value)
        {
            uint absValue = StripSign(value);
            return (absValue < PositiveInfinityBits)    // is finite
                && (absValue != 0)                      // is not zero
                && ((absValue & BiasedExponentMask) == 0);    // is subnormal (has a zero exponent)
        }

        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        ///
        /// <param name="obj">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="obj"/>,
        /// zero if this is equal to <paramref name="obj"/>, or a value greater than zero
        /// if this is greater than <paramref name="obj"/>.
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="obj"/> is not of type <see cref="BFloat16"/>.</exception>
        public int CompareTo(object obj)
        {
            if (!(obj is BFloat16))
            {
                return (obj is null) ? 1 : throw new ArgumentException("Object must be of type BFloat16");
            }
            return CompareTo((BFloat16)(obj));
        }

        /// <summary>
        /// Compares this object to another object, returning an integer that indicates the relationship.
        /// </summary>
        /// <param name="other">Object to compare to</param>
        /// <returns>A value less than zero if this is less than <paramref name="other"/>,
        /// zero if this is equal to <paramref name="other"/>,
        /// or a value greater than zero if this is greater than <paramref name="other"/>.</returns>
        public int CompareTo(BFloat16 other)
        {
            if (this < other)
            {
                return -1;
            }

            if (this > other)
            {
                return 1;
            }

            if (this == other)
            {
                return 0;
            }

            if (IsNaN(this))
            {
                return IsNaN(other) ? 0 : -1;
            }

            Debug.Assert(IsNaN(other));
            return 1;
        }

        /// <summary>
        /// Returns a value indicating whether this instance and other BFloat16 represent the same value.
        /// </summary>
        /// <param name="other">A BFloat16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(BFloat16 other)
        {
            return value == other.value
                || AreZero(this, other)
                || (IsNaN(this) && IsNaN(other));
        }

        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
        /// <returns>true if obj is BFloat16 its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            return (obj is BFloat16 other) && Equals(other);
        }

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            if (IsNaNOrZero(this))
            {
                // All NaNs should have the same hash code, as should both Zeros.
                return value & PositiveInfinityBits;
            }
            return value;
        }

        /// <summary>
        /// Returns a string representation of the current value.
        /// </summary>
        /// <returns>Text representation of BFloat16</returns>
        public override string ToString()
        {
            return $"{value} : {ToFloat()}";
        }

        /// <summary>
        /// Explicit conversion
        /// </summary>
        /// <returns>single precision value converted from Float16</returns>
        public float ToFloat()
        {
            return (float)this;
        }

        /// <summary>Explicitly converts a <see cref="float" /> value to its nearest representable bfloat16 value.</summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable half-precision floating-point value.</returns>
        public static explicit operator BFloat16(float value)
        {
            if (float.IsNaN(value))
            {
                return NaN;
            }

            uint singleBits = BitOpsUtils.SingleToUInt32Bits(value);
            ushort bfloatBits = BitOpsUtils.SingleBitsToBFloat16Bits(singleBits);

            // Round this up. Implement the same logic pytorch uses for rounding.
            // We use RoundingBase that is 0x7FFF + (1), so we carry the 1 to the next bit.
            // either the last bfloat bit is 1 or singleBits have some bits set.
            singleBits += ((uint)bfloatBits & 1) + RoundingBase;
            bfloatBits = BitOpsUtils.SingleBitsToBFloat16Bits(singleBits);
            return new BFloat16(bfloatBits);
        }

        /// <summary>
        /// Explicitly converts a BFloat16 value to its nearest representable <see cref="float" /> value.
        /// </summary>
        /// <param name="value">The value to convert.</param>
        /// <returns><paramref name="value" /> converted to its nearest representable <see cref="float" /> value.</returns>
        public static explicit operator float(BFloat16 value)
        {
            bool sign = IsNegative(value);
            int exp = value.BiasedExponent;
            uint sig = value.TrailingSignificand;

            if (exp == MaxBiasedExponent)
            {
                if (sig != 0)
                {
                    // Shift sig left 54 bits to get a 64-bit integer
                    // to cut off all but 8 bits of the significant
                    return BitOpsUtils.CreateSingleNaN(sign, (ulong)sig << 56);
                }
                return sign ? float.NegativeInfinity : float.PositiveInfinity;
            }

            if (exp == 0 && sig == 0)
            {
                // Positive / Negative zero
                return (sign) ? -0.0f : 0.0f;
            }

            // All subnormal numbers in BFloat16 would be also subnormal in FP32 because they
            // share the exponent.
            uint singleBits = BitOpsUtils.BFloat16BitsToSingleBits(value.value);
            return BitOpsUtils.UInt32BitsToSingle(singleBits);
        }

        /// <summary>
        /// Flips the sign. NaNs are not affected.
        /// IEEE 754 specifies NaNs to be propagated
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static BFloat16 Negate(BFloat16 value)
        {
            return IsNaN(value) ? value : new BFloat16((ushort)(value.value ^ SignMask));
        }

        /// <summary>
        /// The function returns true if the value is either NaN or zero.
        /// </summary>
        /// <param name="value">instance of BFloat16</param>
        /// <returns>true if NaN or zero.</returns>
        public static bool IsNaNOrZero(BFloat16 value)
        {
            uint abs = StripSign(value);
            return (abs == 0 || abs > PositiveInfinityBits);
        }

        #region Utilities

        private static bool AreZero(BFloat16 left, BFloat16 right)
        {
            // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
            // for two values by or'ing the private bits together and stripping the sign. They are both zero,
            // and therefore equivalent, if the resulting value is still zero.
            return (ushort)((left.value | right.value) & ~SignMask) == 0;
        }

        private static uint StripSign(BFloat16 value)
        {
            return (ushort)(value.value & ~SignMask);
        }

        #endregion
    }
}
