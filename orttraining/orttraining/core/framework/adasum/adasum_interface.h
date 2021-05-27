// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include <float.h>
#include <cmath>

#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#endif

#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/framework/distributed_run_context.h"

#ifdef ENABLE_CPU_FP16_TRAINING_OPS
#include "orttraining/core/framework/adasum/m256_utils.h"
#endif

namespace onnxruntime {
namespace training {

static inline bool IsPowerOfTwo(unsigned x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

// Interface for Adasum algorithm
template <typename Communicator_type>
class AdasumInterface {
 public:
  Status DispatchFusedAllreduce(void* grad_buffer, void* recv_buffer,
                                std::vector<int>& tensor_counts, int start_level,
                                Communicator_type communicator, int tag,
                                const Communicator_type* reduction_comms,
                                MLDataType data_type) {
    if (data_type == DataTypeImpl::GetType<MLFloat16>()) {
      FusedAllreduce((uint16_t*)grad_buffer, (uint16_t*)recv_buffer,
                     data_type, tensor_counts, start_level, communicator, tag,
                     reduction_comms);
    } else if (data_type == DataTypeImpl::GetType<float>()) {
      FusedAllreduce((float*)grad_buffer, (float*)recv_buffer,
                     data_type, tensor_counts, start_level, communicator, tag,
                     reduction_comms);
    } else if (data_type == DataTypeImpl::GetType<double>()) {
      FusedAllreduce((double*)grad_buffer, (double*)recv_buffer,
                     data_type, tensor_counts, start_level, communicator, tag,
                     reduction_comms);
    } else {
      // Shouldn't reach here
      ORT_THROW("Unsupported datatype for Adasum allreduce.");
    }
    return Status::OK();
  }

  virtual bool IsAdasumInitialized() = 0;

  virtual void InitializeVHDDReductionComms(WorkerGroupType worker_group) = 0;

  virtual const Communicator_type* GetReductionComms() = 0;

 protected:
  // Communication primitives required for Adasum algorithm
  virtual void PointToPointSendRecv(void* input_data_buffer,
                                    int64_t input_buffer_bytes,
                                    void* output_data_buffer,
                                    int64_t output_buffer_bytes,
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
  // data_type: the element type of grad_buffer.
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
                      const Communicator_type* reduction_comms) {
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

    size_t total_counts_sum = 0;
    for (size_t i = 0; i < tensor_counts.size(); i++)
      total_counts_sum += tensor_counts[i];
    size_t myCount = total_counts_sum;
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

      size_t myCountSoFar = 0;
      int nghrCountSoFar = 0;
      if ((rank & level) != 0) {
        myCount = (size_t)secondHalfMyCount;
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
                  nghrCount - nghrCountSoFar;  // should not be negative
              tensor_counts[i] =
                  tensor_counts[i] -
                  (nghrCount - nghrCountSoFar);  // should not be negative
            }
          } else {
            tensor_counts[i] = tensor_counts[i];
            nghrCountVec[nghrCountVec_index][i] = 0;
          }
          nghrCountSoFar += nghrCountVec[nghrCountVec_index][i];
          myCountSoFar += tensor_counts[i];
        }
      } else {
        myCount = (size_t)firstHalfMyCount;
        nghrCount = secondHalfMyCount;
        sendOffset = myCount;
        recvOffset = 0;

        for (size_t i = 0; i < tensor_counts.size(); i++) {
          if (myCountSoFar <= myCount) {
            if (myCountSoFar + tensor_counts[i] <= myCount) {
              tensor_counts[i] = tensor_counts[i];
              nghrCountVec[nghrCountVec_index][i] = 0;
            } else {
              assert((myCount - myCountSoFar) >= 0);
              nghrCountVec[nghrCountVec_index][i] =
                  tensor_counts[i] -
                  (myCount - myCountSoFar);  // should not be negative
              tensor_counts[i] =
                  myCount - myCountSoFar;  // should not be negative
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
                                   const Communicator_type& comm, bool isLeftNeighbor,
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
};

}  // namespace training
}  // namespace onnxruntime
