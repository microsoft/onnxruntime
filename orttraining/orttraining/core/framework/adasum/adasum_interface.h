// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include <emmintrin.h>
#include <float.h>
#include <immintrin.h>

#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"

#include "orttraining/core/graph/optimizer_config.h"

namespace onnxruntime {
namespace training {

// Initial temp buffer size for Adasum allreduce
const size_t INITIAL_TEMP_BUFFER_SIZE = 64 * 1024 * 1024;

static inline bool IsPowerOfTwo(ulong x)
{
  return (x != 0) && ((x & (x - 1)) == 0);
}

static inline AdasumReductionType GetAdasumAlgo(int64_t reduce_algo)
{
  switch (reduce_algo) {
    case 0:
      return AdasumReductionType::CpuReduction;
    case 1:
      return AdasumReductionType::GpuHierarchical;
    default:
      ORT_THROW("Invalid Adasum reduction algorithm.");
  }
}

// Interface for Adasum algorithm
template <typename Communicator_type>
class AdasumInterface {
public:
  AdasumInterface() {
    // allocator_ = allocator;
    // current_recv_buffer_length = INITIAL_TEMP_BUFFER_SIZE;
    // // Pre-allocate a buffer.
    // recv_buffer_ = (uint8_t*)allocator_->Alloc(current_recv_buffer_length);
  };

  ~AdasumInterface() {
    // if (recv_buffer_ != nullptr) {
    //   allocator_->Free((void*)recv_buffer_);
    // }
  }

  Status DispatchFusedAllreduce(void* grad_buffer, void* recv_buffer,
                                std::vector<int>& tensor_counts, int start_level,
                                Communicator_type communicator, int tag,
                                Communicator_type* reduction_comms,
                                MLDataType data_type) {
    if (data_type == DataTypeImpl::GetType<MLFloat16>()) {
      FusedAllreduce((uint16_t*)grad_buffer, (uint16_t*)recv_buffer,
                     data_type, tensor_counts, start_level, communicator, tag,
                     reduction_comms);
    } else if (data_type == DataTypeImpl::GetType<float>()) {
      FusedAllreduce((float*)grad_buffer, (float*)recv_buffer,
                     data_type, tensor_counts, start_level, communicator, tag,
                     reduction_comms);
    } else if(data_type == DataTypeImpl::GetType<double>()) {
      FusedAllreduce((double*)grad_buffer, (double*)recv_buffer,
                     data_type, tensor_counts, start_level, communicator, tag,
                     reduction_comms);
    } else {
      // Shouldn't reach here
      ORT_THROW("Unsupported datatype for Adasum allreduce.");
    }
    return Status::OK();
  }
  
  // Get recv buffer
  uint8_t* GetRecvBuffer(int buffer_length) {
    return CheckBufferAndReallocate(&recv_buffer_, buffer_length,
                                    current_recv_buffer_length);
  }


  virtual bool IsAdasumInitialized() = 0;

  virtual void InitializeVHDDReductionComms() = 0;

  virtual Communicator_type* GetReductionComms() = 0;

protected:
  // Communication primitives required for Adasum algorithm
  virtual void PointToPointSendRecv(void* input_data_buffer,
                                    int64_t input_buffer_length,
                                    void* output_data_buffer,
                                    int64_t output_buffer_length,
                                    MLDataType data_type, int dst_src_rank,
                                    int tag, Communicator_type communicator) = 0;

  virtual void SumAllreduceWithComm(void* data, int num_elements,
                                    MLDataType data_type,
                                    Communicator_type comm) = 0;

  virtual int GetRankWithComm(Communicator_type communicator) = 0;

  virtual int GetSizeWithComm(Communicator_type communicator) = 0;

  virtual void DispatchComputeDotAndNormSqrds(const void* __restrict__ a,
                                              const void* __restrict__ b,
                                              MLDataType data_type,
                                              int count, double& dotProduct,
                                              double& anormsq, double& bnormsq) {
    if (data_type == DataTypeImpl::GetType<MLFloat16>()) {
      ComputeDotAndNormSqrdsfp16((uint16_t*)a, (uint16_t*)b, count, dotProduct,
                                 anormsq, bnormsq);
    } else if (data_type == DataTypeImpl::GetType<float>()) {
      ComputeDotAndNormSqrds((float*)a, (float*)b, count, dotProduct, anormsq,
                             bnormsq);
    } else if (data_type == DataTypeImpl::GetType<double>()) {
      ComputeDotAndNormSqrds((double*)a, (double*)b, count, dotProduct, anormsq,
                             bnormsq);
    } else {
      // Shouldn't reach here
      ORT_THROW("Unsupported datatype for Adasum allreduce.");
    }
  }

  virtual void DispatchScaledAdd(MLDataType data_type, int count,
                                 double acoeff, void* __restrict__ a,
                                 double bcoeff, void* __restrict__ b) {
    if (data_type == DataTypeImpl::GetType<MLFloat16>()) {
      ScaledAddfp16(count, acoeff, (uint16_t*)a, bcoeff, (uint16_t*)b);
    } else if (data_type == DataTypeImpl::GetType<float>()) {
      ScaledAdd(count, acoeff, (float*)a, bcoeff, (float*)b);
    } else if (data_type == DataTypeImpl::GetType<double>()) {
      ScaledAdd(count, acoeff, (double*)a, bcoeff, (double*)b);
    } else {
      // Shouldn't reach here
      ORT_THROW("Unsupported datatype for Adasum allreduce.");
    }
  }

  // Check buffer length and re-allocate if necessary
  virtual uint8_t* CheckBufferAndReallocate(uint8_t** buffer,
                                            uint64_t buffer_length,
                                            uint64_t& current_length) {
    if (buffer_length <= current_length) {
      return *buffer;
    }
    *buffer = (uint8_t*)realloc(*buffer, buffer_length);
    current_length = buffer_length;
    return *buffer;
  }

private:
  // Allocator for temporary buffer allocations
  AllocatorPtr allocator_ = nullptr;

  // Temp buffer used by Adasum operations
  uint8_t* recv_buffer_ = nullptr;

  // Keep track of current recv buffer length
  uint64_t current_recv_buffer_length = 0;

  // Perform Adasum allreduce using a vector-halving, distance-doubling (VHDD)
  // approach. grad_buffer: holds the data to reduce and will hold the result.
  // recv_buffer: must point to a buffer of the same size as grad_buffer.
  // horovod_datatype: the element type of grad_buffer.
  // tensor_counts: is a list of how many elements grad_buffer contains for each
  // tensor
  //                involved in the allreduce. It should contain a 0 if this
  //                rank holds no data for the tensor (see start_level below for
  //                when this can happen).
  // start_level: set to 1 to perform all levels of the operation. When set to
  // n>1 the
  //              first n-1 levels are skipped. This is useful when the
  //              communication inside the node is implemented using another
  //              reduce-scatter algorithm, e.g. the one in NCCL, which may be
  //              desireable on some hardware configurations. When
  //              start_level>1, tensor_counts must be set according to the
  //              slices owned by this rank.
  // communicator: the communicator to reduce with.
  // tag: a value used as the message tag for each send/recv in this algorithm.
  // This is
  //      useful for multithreaded scenarios. Remember to also create separate
  //      reduction_comms instances when running with multiple threads.
  // reduction_comms: pointer to an array of communicators for computing dot
  // products and
  //                  norms for Adasum. The communicators should include exactly
  //                  the ranks that this rank has either directly or indirectly
  //                  communicated with after each level of VHDD.
  template <typename T>
  void FusedAllreduce(T* grad_buffer, T* recv_buffer, MLDataType data_type,
                      std::vector<int>& tensor_counts, int start_level,
                      Communicator_type communicator, int tag,
                      Communicator_type* reduction_comms) {
    int per_element_size = data_type->Size();
    int rank = GetRankWithComm(communicator);
    int size = GetSizeWithComm(communicator);

    if (IsPowerOfTwo(size) == false) {
      ORT_THROW(
        "Adasum doesn't currently support reduction among non-power-of-2 number of ranks.");
    }

    std::vector<std::vector<int>> nghrCountVec;
    std::vector<double> normAndDots(tensor_counts.size() * 3 * 2);

    int nearest_power_2 = 1;
    for (nearest_power_2 = 1; (nearest_power_2 << 1) <= size;
         nearest_power_2 = (nearest_power_2 << 1)) {
    }
    int level;

    int nghrCountVec_index = 0;
    int orgSize = size;
    size = nearest_power_2;

    int total_counts_sum = 0;
    for (size_t i = 0; i < tensor_counts.size(); i++)
      total_counts_sum += tensor_counts[i];
    int myCount = total_counts_sum;
    int comm_index;
    for (level = 1, comm_index = 0; level < size;
         level = (level << 1), comm_index++) {
      if (level < start_level) {
        continue;
      }

      int neighbor_rank = rank ^ level;
      int nghrCount = 0;
      int sendOffset = 0;
      int recvOffset = 0;
      int firstHalfMyCount = (int)(myCount >> 1);
      int secondHalfMyCount = (int)myCount - firstHalfMyCount;

      nghrCountVec.emplace_back();
      nghrCountVec[nghrCountVec_index].resize(tensor_counts.size());

      int myCountSoFar = 0;
      int nghrCountSoFar = 0;
      if ((rank & level) != 0) {
        myCount = secondHalfMyCount;
        nghrCount = firstHalfMyCount;
        sendOffset = 0;
        recvOffset = nghrCount;

        for (size_t i = 0; i < tensor_counts.size(); i++) {
          if (nghrCountSoFar <= nghrCount) {
            if (nghrCountSoFar + tensor_counts[i] <= nghrCount) {
              nghrCountVec[nghrCountVec_index][i] = tensor_counts[i];
              tensor_counts[i] = 0;
            } else {
              nghrCountVec[nghrCountVec_index][i] =
                  nghrCount - nghrCountSoFar; // should not be negative
              tensor_counts[i] =
                  tensor_counts[i] -
                  (nghrCount - nghrCountSoFar); // should not be negative
            }
          } else {
            tensor_counts[i] = tensor_counts[i];
            nghrCountVec[nghrCountVec_index][i] = 0;
          }
          nghrCountSoFar += nghrCountVec[nghrCountVec_index][i];
          myCountSoFar += tensor_counts[i];
        }
      } else {
        myCount = firstHalfMyCount;
        nghrCount = secondHalfMyCount;
        sendOffset = myCount;
        recvOffset = 0;

        for (size_t i = 0; i < tensor_counts.size(); i++) {
          if (myCountSoFar <= myCount) {
            if (myCountSoFar + tensor_counts[i] <= myCount) {
              tensor_counts[i] = tensor_counts[i];
              nghrCountVec[nghrCountVec_index][i] = 0;
            } else {
              nghrCountVec[nghrCountVec_index][i] =
                  tensor_counts[i] -
                  (myCount - myCountSoFar); // should not be negative
              tensor_counts[i] =
                  myCount - myCountSoFar; // should not be negative
            }
          } else {
            nghrCountVec[nghrCountVec_index][i] = tensor_counts[i];
            tensor_counts[i] = 0;
          }
          nghrCountSoFar += nghrCountVec[nghrCountVec_index][i];
          myCountSoFar += tensor_counts[i];
        }
      }

      nghrCountVec_index++;

      this->PointToPointSendRecv(
          (uint8_t*)(&grad_buffer[sendOffset]), nghrCount * per_element_size,
          (uint8_t*)(&recv_buffer[recvOffset]), myCount * per_element_size,
          data_type, neighbor_rank, tag, communicator);
      if ((rank & level) != 0) {
        grad_buffer = &grad_buffer[nghrCount];
        recv_buffer = &recv_buffer[nghrCount];
      }
      FusedPairwiseReduceWithComm((uint8_t*)grad_buffer, (uint8_t*)recv_buffer,
          data_type, tensor_counts, reduction_comms[comm_index],
          (rank & level) == 0, normAndDots);
    }

    for (level = (size >> 1); level > 0; level = (level >> 1)) {
      if (level < start_level) {
        continue;
      }
      int neighbor_rank = rank ^ level;

      nghrCountVec_index--;
      int nghrCount = 0;
      for (size_t i = 0; i < tensor_counts.size(); i++) {
        nghrCount += nghrCountVec[nghrCountVec_index][i];
        tensor_counts[i] += nghrCountVec[nghrCountVec_index][i];
      }

      if ((rank & level) == 0) {
        recv_buffer = &grad_buffer[myCount];
      } else {
        recv_buffer = &grad_buffer[-nghrCount];
      }
      this->PointToPointSendRecv(grad_buffer, myCount * per_element_size,
                                 recv_buffer, nghrCount * per_element_size,
                                 data_type, neighbor_rank, tag,
                                 communicator);
      if ((rank & level) != 0) {
        grad_buffer = &grad_buffer[-nghrCount];
      }
      myCount += nghrCount;
    }
    size = orgSize;
  }

  void FusedPairwiseReduceWithComm(uint8_t* a, uint8_t* b,
                                   MLDataType data_type,
                                   std::vector<int>& tensor_counts,
                                   Communicator_type& comm, bool isLeftNeighbor,
                                   std::vector<double>& normAndDots) {
    static double sqrt_double_min = std::sqrt(DBL_MIN);
    int per_element_size = data_type->Size();
    int bytesSoFar = 0;
    for (size_t i = 0; i < tensor_counts.size(); i++) {
      double dotProduct = 0.;
      double anormsq = 0.;
      double bnormsq = 0.;

      DispatchComputeDotAndNormSqrds(&a[bytesSoFar], &b[bytesSoFar],
                                     data_type, tensor_counts[i],
                                     dotProduct, anormsq, bnormsq);
      normAndDots[i * 3] = dotProduct;
      if (isLeftNeighbor) {
        normAndDots[i * 3 + 1] = anormsq;
        normAndDots[i * 3 + 2] = bnormsq;
      } else {
        normAndDots[i * 3 + 1] = bnormsq;
        normAndDots[i * 3 + 2] = anormsq;
      }
      bytesSoFar += tensor_counts[i] * per_element_size;
    }

    SumAllreduceWithComm((void*)normAndDots.data(),
                         3 * tensor_counts.size(), DataTypeImpl::GetType<double>(),
                         comm);

    bytesSoFar = 0;
    for (size_t i = 0; i < tensor_counts.size(); i++) {
      double dotProduct = normAndDots[i * 3];
      double anormsq;
      double bnormsq;
      if (isLeftNeighbor) {
        anormsq = normAndDots[i * 3 + 1];
        bnormsq = normAndDots[i * 3 + 2];
      } else {
        bnormsq = normAndDots[i * 3 + 1];
        anormsq = normAndDots[i * 3 + 2];
      }

      double acoeff = 1;
      double bcoeff = 1;
      if (anormsq >= sqrt_double_min) {
        acoeff = 1.0 - dotProduct / anormsq * 0.5;
      }
      if (bnormsq >= sqrt_double_min) {
        bcoeff = 1.0 - dotProduct / bnormsq * 0.5;
      }

      DispatchScaledAdd(data_type, tensor_counts[i], acoeff,
                        &a[bytesSoFar], bcoeff, &b[bytesSoFar]);
      bytesSoFar += tensor_counts[i] * per_element_size;
    }
  }

  // Given two vectors compute their dot product and the squared norm for each.
  template <typename T>
  void ComputeDotAndNormSqrds(const T* __restrict__ a, const T* __restrict__ b,
                              int count, double& dotProduct, double& anormsq,
                              double& bnormsq) {
    dotProduct = 0.;
    anormsq = 0.;
    bnormsq = 0.;

    for (int i = 0; i < count; i++) {
      dotProduct += (double)a[i] * (double)b[i];
      anormsq += (double)a[i] * (double)a[i];
      bnormsq += (double)b[i] * (double)b[i];
    }
  }

  // Update a vector to a linear combination of itself and another vector.
  template <typename T>
  void ScaledAdd(int n, double acoeff, T* __restrict__ a, double bcoeff,
                 T* __restrict__ b) {
    for (int i = 0; i < n; i++) {
      a[i] = acoeff * a[i] + bcoeff * b[i];
    }
  }

  inline void ComputeDotAndNormSqrdsfp16(const uint16_t* __restrict__ a,
                                         const uint16_t* __restrict__ b,
                                         int len, double& dotProduct,
                                         double& anormsq, double& bnormsq) {
    int i;
    __m256d dotProductVec = _mm256_setzero_pd();
    __m256d anormVec = _mm256_setzero_pd();
    __m256d bnormVec = _mm256_setzero_pd();
    for (i = 0; i < len - 7; i += 8) {
      __m256 aVec = MmLoaduPh(&a[i]);
      __m256 bVec = MmLoaduPh(&b[i]);
      __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
      __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
      __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
      __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
      dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
      dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
      anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
      anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
      bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
      bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
    }
    if (i < len) {
      __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
      __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
      __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
      __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
      __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
      __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
      dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
      dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
      anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
      anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
      bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
      bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
    }

    dotProduct = Mm256ReductionPd(dotProductVec);
    anormsq = Mm256ReductionPd(anormVec);
    bnormsq = Mm256ReductionPd(bnormVec);
  }

  inline void ScaledAddfp16(int len, double acoeff, uint16_t* __restrict__ a,
                            double bcoeff, uint16_t* __restrict__ b) {
    int i;
    __m256 acoeffVec = _mm256_set1_ps((float)(acoeff));
    __m256 bcoeffVec = _mm256_set1_ps((float)bcoeff);
    for (i = 0; i < len - 7; i += 8) {
      __m256 aVec = MmLoaduPh(&a[i]);
      __m256 bVec = MmLoaduPh(&b[i]);
      aVec = _mm256_mul_ps(acoeffVec, aVec);
      MmStorePh(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec));
    }
    if (i < len) {
      __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
      __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
      aVec = _mm256_mul_ps(acoeffVec, aVec);
      MmStorePhPartial(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec), len - i);
    }
  }

  // reduce 4xfloat64 into one double
  inline double Mm256ReductionPd(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // reduce to scalar
  }

  // load 8 float16s from a and return the __m256 register
  inline __m256 MmLoaduPh(const uint16_t* a) {
    __m128i r = _mm_loadu_si128((__m128i*)(a));
    return _mm256_cvtph_ps(r);
  }

  // store 8 float16 from val into a
  inline void MmStorePh(uint16_t* a, __m256 val) {
    __m128i r = _mm256_cvtps_ph(val, 0);
    _mm_storeu_si128((__m128i*)a, r);
  }

  // load len (< 8) float16s from a, fill the rest with 0s, and return the
  // __m256 register
  inline __m256 MmLoaduPhPartial(const uint16_t* a, int len) {
    short e[8];
    std::memset(e, 0, sizeof(e));
    std::memcpy(e, a, std::min(len, 8) * sizeof(short));
    __m128i es = _mm_set_epi16(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
    return _mm256_cvtph_ps(es);
  }

  // store the first len (< 8) float16s from val and store into a
  inline void MmStorePhPartial(uint16_t* a, __m256 val, int len) {
    __m128i r = _mm256_cvtps_ph(val, 0);
    // for (int i = 0; i < std::min(len, 8); i++)
    //    a[i].value = _mm_extract_epi16(r, i);
    // but we cannot do this because the second argument to _mm_extract_epi16
    // has to be a compile time constant
    if (0 < len)
      a[0] = (short)_mm_extract_epi16(r, 0);
    if (1 < len)
      a[1] = (short)_mm_extract_epi16(r, 1);
    if (2 < len)
      a[2] = (short)_mm_extract_epi16(r, 2);
    if (3 < len)
      a[3] = (short)_mm_extract_epi16(r, 3);
    if (4 < len)
      a[4] = (short)_mm_extract_epi16(r, 4);
    if (5 < len)
      a[5] = (short)_mm_extract_epi16(r, 5);
    if (6 < len)
      a[6] = (short)_mm_extract_epi16(r, 6);
    if (7 < len)
      a[7] = (short)_mm_extract_epi16(r, 7);
  }
};

} // namespace training
} // namespace onnxruntime
