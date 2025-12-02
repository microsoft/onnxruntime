/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

		test_sqlutgemm.cpp

Abstract:

		Tests for MLAS LUT-based n-bit GEMM (TMAC/LUT path) for 2-bit.

--*/

#include "test_util.h"
#include "mlas_qnbit.h"
#include "mlas_q4.h"

// Generic template to future-proof for different bit widths; instantiate with 2 for now.
template <size_t BlkBitWidth, size_t BlkLen>
class MlasSQLutGemm2BitTest : public MlasTestBase {
 private:
	MatrixGuardBuffer<float> BufferA;
	MatrixGuardBuffer<float> BufferB;
	MatrixGuardBuffer<uint8_t> BufferQuantBData;
	MatrixGuardBuffer<uint8_t> BufferQuantBZeroPoint;
	MatrixGuardBuffer<float> BufferQuantBScale;
	MatrixGuardBuffer<std::byte> BufferPackedQuantB;
	MatrixGuardBuffer<float> BufferPackedScalesZP;
	MatrixGuardBuffer<float> BufferBias;
	MatrixGuardBuffer<float> BufferC;
	MatrixGuardBuffer<float> BufferCReference;

	void QuantizeB(size_t K, size_t N,
									const float* B,
									uint8_t*& qdata,
									float*& qscale,
									uint8_t*& qzp,
									bool symmetric) {
		size_t q_data_bytes = 0, q_scale_size = 0, q_zp_bytes = 0;
		MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(BlkLen, /*columnwise*/ true,
																				 static_cast<int>(K), static_cast<int>(N),
																				 q_data_bytes, q_scale_size, &q_zp_bytes);
		qdata = BufferQuantBData.GetBuffer(q_data_bytes);
		qscale = BufferQuantBScale.GetBuffer(q_scale_size);
		qzp = symmetric ? nullptr : BufferQuantBZeroPoint.GetBuffer(q_zp_bytes);

		MlasQuantizeBlockwise<float, BlkBitWidth>(qdata, qscale, qzp,
																		B, BlkLen,
																		/*columnwise*/ true,
																		static_cast<int>(K), static_cast<int>(N),
																		static_cast<int>(N),
																		GetMlasThreadPool());
	}

	void ReferenceDequantFp32(size_t M, size_t N, size_t K,
											const float* A,
											const uint8_t* QuantBData,
											const float* QuantBScale,
											const uint8_t* QuantBZeroPoint,
											const float* Bias,
											float* C) {
		MatrixGuardBuffer<float> deqBbuf;
		float* DeqB = deqBbuf.GetBuffer(K * N);
		MlasDequantizeBlockwise<float, BlkBitWidth>(
				DeqB, QuantBData, QuantBScale, QuantBZeroPoint, BlkLen, /*columnwise*/ true,
				static_cast<int>(K), static_cast<int>(N), GetMlasThreadPool());

		for (size_t m = 0; m < M; ++m) {
			for (size_t n = 0; n < N; ++n) {
				const float* a = A + m * K;
				const float* b = DeqB + n * K;
				float sum = Bias ? Bias[n] : 0.0f;
				for (size_t k = 0; k < K; ++k) {
					sum += a[k] * b[k];
				}
				C[m * N + n] = sum;
			}
		}
	}

	void ReferenceInt8(size_t M, size_t N, size_t K,
											const float* A,
											const uint8_t* QuantBData,
											const float* QuantBScale,
											const uint8_t* QuantBZeroPoint,
											const float* Bias,
											float* C) {
		// Reference path equivalent to CompInt8 for SQ: quantize A to int8 per block and accumulate with unpacked 2-bit B
		const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;

		MatrixGuardBuffer<int8_t> qa_buf;
		MatrixGuardBuffer<float> a_scales_buf;
		int8_t* QA = qa_buf.GetBuffer(M * BlockCountK * BlkLen);
		float* AScales = a_scales_buf.GetBuffer(M * BlockCountK);

		for (size_t m = 0; m < M; ++m) {
			for (size_t k = 0, k_blk = 0; k < K; k += BlkLen, ++k_blk) {
				const size_t local_blk_len = std::min(K - k, BlkLen);
				float amax = 0.0f;
				for (size_t kk = 0; kk < local_blk_len; ++kk) {
					amax = std::max(amax, fabsf(A[m * K + k + kk]));
				}
				constexpr float rmax = (1 << 7) - 1;
				float scale = amax / rmax;
				float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
				AScales[m * BlockCountK + k_blk] = scale;
				for (size_t kk = 0; kk < BlkLen; ++kk) {
					float q = roundf((k + kk < K ? A[m * K + k + kk] : 0.0f) * inv);
					QA[m * BlockCountK * BlkLen + k + kk] = static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
				}
			}
		}

		for (size_t m = 0; m < M; ++m) {
			for (size_t n = 0; n < N; ++n) {
				float sum = Bias ? Bias[n] : 0.0f;
				for (size_t k = 0, k_blk = 0; k < K; k += BlkLen, ++k_blk) {
					const size_t k_blk_len = std::min(K - k, BlkLen);
					const float a_scale = AScales[m * BlockCountK + k_blk];
					const float b_scale = QuantBScale[n * BlockCountK + k_blk];
					uint8_t b_zp = (BlkBitWidth == 4 ? 8 : (BlkBitWidth == 2 ? 2 : 0)); // symmetric default
					if (QuantBZeroPoint) {
						const int pack = 8 / BlkBitWidth;
						uint8_t zp_byte = QuantBZeroPoint[n * ((BlockCountK + 3) / pack) + k_blk / pack];
						if constexpr (BlkBitWidth == 2) {
							int shift = (k_blk & 3) * 2;
							b_zp = (zp_byte >> shift) & 0x03;
						} else if constexpr (BlkBitWidth == 4) {
							b_zp = (k_blk & 1) ? (zp_byte >> 4) : (zp_byte & 0x0F);
						}
					}
					int32_t qsum = 0;
					for (size_t kk = 0; kk < k_blk_len; ++kk) {
						const int8_t qa = QA[m * BlockCountK * BlkLen + k + kk];
						const int pack = 8 / BlkBitWidth; // entries per byte
						const size_t idx = (n * BlockCountK * BlkLen + k + kk) / pack;
						const uint8_t qb_byte = QuantBData[idx];
						int8_t qb = 0;
						if constexpr (BlkBitWidth == 2) {
							qb = static_cast<int8_t>((qb_byte >> ((kk & 3) * 2)) & 0x03);
						} else if constexpr (BlkBitWidth == 4) {
							qb = static_cast<int8_t>((kk & 1) ? (qb_byte >> 4) : (qb_byte & 0x0F));
						}
						qb -= static_cast<int8_t>(b_zp);
						qsum += static_cast<int32_t>(qa) * static_cast<int32_t>(qb);
					}
					sum += static_cast<float>(qsum) * a_scale * b_scale;
				}
				C[m * N + n] = sum;
			}
		}
	}

 public:
	void Test(size_t M, size_t N, size_t K, bool with_threadpool, bool symmetric, bool with_bias) {
		MLAS_THREADPOOL* tp = with_threadpool ? GetMlasThreadPool() : nullptr;

		const float* A = BufferA.GetBuffer(K * M);
		const float* B = BufferB.GetBuffer(N * K);

		const float* Bias = nullptr;
		if (with_bias) {
			Bias = BufferBias.GetBuffer(N);
		}

		// Quantize B to BlkBitWidth-bit blockwise
		uint8_t* qB = nullptr;
		float* sB = nullptr;
		uint8_t* zpB = nullptr;
		QuantizeB(K, N, B, qB, sB, zpB, symmetric);

		// Initialize LUT config and pack B/scales/zp
		MlasInitLUTGemmKernelConfig(N, K, /*nbits*/ BlkBitWidth, BlkLen, /*has_zp*/ zpB != nullptr);

		void* packedB = nullptr;
		size_t packedBSize = MlasLUTGemmPackQuantBDataSize(N, K, /*nbits*/ BlkBitWidth, BlkLen, /*has_zp*/ zpB != nullptr, TMAC);
		if (packedBSize > 0) {
			packedB = BufferPackedQuantB.GetBuffer(packedBSize);
			MlasLUTGemmPackQuantBData(N, K, /*nbits*/ BlkBitWidth, BlkLen,
																static_cast<const std::byte*>(reinterpret_cast<const void*>(qB)),
																static_cast<std::byte*>(packedB), tp);
		}

		size_t packedSZSize = MlasLUTPackScalesAndZeroPointsSize(N, K, BlkLen, /*has_zp*/ zpB != nullptr);
		if (packedSZSize > 0) {
			float* packedSZ = BufferPackedScalesZP.GetBuffer(packedSZSize);
			MlasLUTPackScalesAndZeroPoints(N, K, /*nbits*/ BlkBitWidth, BlkLen, /*has_zp*/ zpB != nullptr,
																		 packedSZ, sB, zpB);
		}

		float* C = BufferC.GetBuffer(N * M, true);
		float* CRef = BufferCReference.GetBuffer(N * M, true);

		// Execute LUT GEMM
		MlasLUTGemm(A, BlkLen,
								static_cast<const std::byte*>(packedB),
								BufferPackedScalesZP.GetBuffer(packedSZSize),
								C,
								K, M, N,
								tp);

		// Reference implementation (int8-style accumulation)
		ReferenceInt8(M, N, K, A, qB, sB, zpB, Bias, CRef);

		// Cross-check via explicit dequantization + FP32 GEMM
		MatrixGuardBuffer<float> CRefDeqBuf;
		float* CRefDeq = CRefDeqBuf.GetBuffer(N * M, true);
		ReferenceDequantFp32(M, N, K, A, qB, sB, zpB, Bias, CRefDeq);

		// Compare results
		for (size_t m = 0; m < M; ++m) {
			for (size_t n = 0; n < N; ++n) {
				size_t idx = m * N + n;
				ASSERT_TRUE(CloseEnough(C[idx], CRef[idx]))
						<< "Expected: " << CRef[idx] << " Actual: " << C[idx]
						<< "@[" << m << "x" << n << "], M=" << M << ", N=" << N << ", K=" << K;
				ASSERT_TRUE(CloseEnough(C[idx], CRefDeq[idx]))
						<< "DequantRef mismatch. Expected: " << CRefDeq[idx] << " Actual: " << C[idx]
						<< "@[" << m << "x" << n << "], M=" << M << ", N=" << N << ", K=" << K;
			}
		}
	}

	static const char* GetTestSuiteName() {
		static std::string suite_name = std::string("SQLutGemm2Bit") + "BlkLen" + std::to_string(BlkLen);
		return suite_name.c_str();
	}
};

// Fixture to register parameterized tests quickly
template <size_t BlkBitWidth, size_t BlkLen>
class SQLutGemm2BitShortExecuteTest : public MlasTestFixture<MlasSQLutGemm2BitTest<BlkBitWidth, BlkLen>> {
 public:
	explicit SQLutGemm2BitShortExecuteTest(size_t M, size_t N, size_t K,
																				 bool with_threadpool, bool symmetric, bool with_bias)
			: M_(M), N_(N), K_(K), with_threadpool_(with_threadpool), symmetric_(symmetric), with_bias_(with_bias) {}

	void TestBody() override {
		MlasTestFixture<MlasSQLutGemm2BitTest<BlkBitWidth, BlkLen>>::mlas_tester->Test(M_, N_, K_, with_threadpool_, symmetric_, with_bias_);
	}

	static size_t RegisterSingleTest(size_t M, size_t N, size_t K, bool with_threadpool, bool symmetric, bool with_bias) {
		if (!MlasIsLUTGemmAvailable(BlkBitWidth, BlkLen)) {
			return 0;
		}
		if (M < BlkLen || K < BlkLen || N < BlkLen) {
			return 0;
		}

		std::stringstream ss;
		ss << (with_threadpool ? "Threaded" : "SingleThread")
			 << "/isSymmetric" << symmetric
			 << "/M" << M << "xN" << N << "xK" << K
			 << "/hasBias" << with_bias;
		auto name = ss.str();

		testing::RegisterTest(
				MlasSQLutGemm2BitTest<BlkBitWidth, BlkLen>::GetTestSuiteName(),
				name.c_str(),
				nullptr,
				name.c_str(),
				__FILE__,
				__LINE__,
				[=]() -> MlasTestFixture<MlasSQLutGemm2BitTest<BlkBitWidth, BlkLen>>* {
					return new SQLutGemm2BitShortExecuteTest<BlkBitWidth, BlkLen>(M, N, K, with_threadpool, symmetric, with_bias);
				});
		return 1;
	}

	static size_t RegisterAll() {
		size_t count = 0;
		for (bool with_threadpool : {false, true}) {
			for (bool symmetric : {false, true}) {
				for (size_t b = 256; b <= 512; b <<= 1) {
					count += RegisterSingleTest(b, b, b, with_threadpool, symmetric, false);
					count += RegisterSingleTest(b, b, b, with_threadpool, symmetric, true);
				}
				count += RegisterSingleTest(64, 128, 128, with_threadpool, symmetric, false);
				count += RegisterSingleTest(128, 256, 256, with_threadpool, symmetric, true);
			}
		}
		return count;
	}

 private:
	size_t M_, N_, K_;
	bool with_threadpool_, symmetric_, with_bias_;
};

static size_t SQLutGemmRegisterAll() {
	size_t count = 0;
	// Instantiate only 2-bit for now
	count += SQLutGemm2BitShortExecuteTest<2, 16>::RegisterAll();
	count += SQLutGemm2BitShortExecuteTest<2, 32>::RegisterAll();
	count += SQLutGemm2BitShortExecuteTest<2, 64>::RegisterAll();
	count += SQLutGemm2BitShortExecuteTest<2, 128>::RegisterAll();
	return count;
}

static UNUSED_VARIABLE bool lut_added_to_main = AddTestRegister(
		[](bool is_short_execute) -> size_t {
			if (is_short_execute) {
				return SQLutGemmRegisterAll();
			}
			return 0;
		});
