#include "packed_sparse_attention_indexer_merge_impl.h"

#include <climits>

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;

__device__ int HashSlot(int value, int capacity) {
  return static_cast<int>((static_cast<unsigned>(value) * 2654435761u) & (capacity - 1));
}

__device__ int FindSlot(const int* keys, int value, int capacity) {
  int slot = HashSlot(value, capacity);
  for (int probe = 0; probe < capacity; ++probe) {
    if (keys[slot] == value || keys[slot] == -1) return slot;
    slot = (slot + 1) & (capacity - 1);
  }
  return -1;
}

__device__ int InclusiveScan(int* flags, int value) {
  const int lane = threadIdx.x;
  flags[lane] = value;
  __syncthreads();
  for (int stride = 1; stride < kThreads; stride *= 2) {
    const int previous = lane >= stride ? flags[lane - stride] : 0;
    __syncthreads();
    flags[lane] += previous;
    __syncthreads();
  }
  return flags[lane];
}

__global__ void MergeRangeKernel(
    const int* base_indices, const int* base_counts, const int* base_rows,
    const int* starts, const int* ends, int* indices, int* counts, int* status,
    int* workspace, int cached_rows, int base_capacity, int output_capacity, int hash_capacity) {
  const int query = blockIdx.x;
  const int lane = threadIdx.x;
  int* output = indices + static_cast<int64_t>(query) * output_capacity;
  int* keys = workspace + static_cast<int64_t>(query) * hash_capacity * 2;
  int* first = keys + hash_capacity;
  __shared__ int flags[kThreads];
  __shared__ int failed;
  __shared__ int emitted;
  __shared__ int overlap;
  if (lane == 0) {
    failed = 0;
    emitted = 0;
    overlap = 0;
    counts[query] = 0;
    status[query] = 0;
  }
  for (int64_t column = lane; column < output_capacity; column += kThreads) output[column] = -1;
  __syncthreads();

  const int row = base_rows[query];
  const int start = starts[query];
  const int end = ends[query];
  if (row < 0 || row >= cached_rows || start < 0 || end < start) {
    if (lane == 0) status[query] = 1;
    return;
  }
  const int count = base_counts[row];
  if (count < 0 || count > base_capacity) {
    if (lane == 0) status[query] = 1;
    return;
  }
  const int* base = base_indices + static_cast<int64_t>(row) * base_capacity;
  for (int64_t column = lane; column < count; column += kThreads) {
    if (base[column] < 0) atomicExch(&failed, 1);
  }
  __syncthreads();
  if (failed) {
    if (lane == 0) status[query] = 1;
    return;
  }

  for (int64_t slot = lane; slot < hash_capacity; slot += kThreads) {
    keys[slot] = -1;
    first[slot] = INT_MAX;
  }
  __syncthreads();
  for (int64_t column = lane; column < count; column += kThreads) {
    const int value = base[column];
    int slot = HashSlot(value, hash_capacity);
    bool inserted = false;
    for (int probe = 0; probe < hash_capacity; ++probe) {
      const int previous = atomicCAS(keys + slot, -1, value);
      if (previous == -1 || previous == value) {
        atomicMin(first + slot, static_cast<int>(column));
        inserted = true;
        break;
      }
      slot = (slot + 1) & (hash_capacity - 1);
    }
    if (!inserted) atomicExch(&failed, 2);
  }
  __syncthreads();
  if (failed) {
    if (lane == 0) status[query] = 2;
    return;
  }

  for (int64_t chunk = 0; chunk < count; chunk += kThreads) {
    const int64_t column = chunk + lane;
    const int value = column < count ? base[column] : -1;
    const int slot = column < count ? FindSlot(keys, value, hash_capacity) : -1;
    const int unique = slot >= 0 && first[slot] == column;
    const int rank = InclusiveScan(flags, unique);
    if (unique) {
      if (static_cast<int64_t>(emitted) + rank <= output_capacity) output[emitted + rank - 1] = value;
      if (value >= start && value < end) atomicAdd(&overlap, 1);
    }
    __syncthreads();
    if (lane == 0) emitted += flags[kThreads - 1];
    __syncthreads();
  }

  const int64_t total = static_cast<int64_t>(emitted) + static_cast<int64_t>(end) - start - overlap;
  if (total > output_capacity) {
    for (int64_t column = lane; column < output_capacity; column += kThreads) output[column] = -1;
    if (lane == 0) status[query] = 2;
    return;
  }
  for (int64_t chunk = start; chunk < end; chunk += kThreads) {
    const int64_t value = chunk + lane;
    const int slot = value < end ? FindSlot(keys, static_cast<int>(value), hash_capacity) : -1;
    const int unique = value < end && (slot < 0 || keys[slot] != value);
    const int rank = InclusiveScan(flags, unique);
    if (unique) output[emitted + rank - 1] = static_cast<int>(value);
    __syncthreads();
    if (lane == 0) emitted += flags[kThreads - 1];
    __syncthreads();
  }
  if (lane == 0) counts[query] = emitted;
}

__global__ void MergeIndicesKernel(
    const int* base_indices, const int* base_counts, const int* base_rows,
    const int* additional_indices, const int* additional_counts,
    int* indices, int* counts, int* status, int* workspace,
    int cached_rows, int base_capacity, int additional_capacity, int output_capacity, int hash_capacity) {
  const int query = blockIdx.x;
  const int lane = threadIdx.x;
  int* output = indices + static_cast<int64_t>(query) * output_capacity;
  int* keys = workspace + static_cast<int64_t>(query) * hash_capacity * 2;
  int* first = keys + hash_capacity;
  __shared__ int flags[kThreads];
  __shared__ int failed;
  __shared__ int emitted;
  if (lane == 0) {
    failed = 0;
    emitted = 0;
    counts[query] = 0;
    status[query] = 0;
  }
  for (int column = lane; column < output_capacity; column += kThreads) output[column] = -1;
  __syncthreads();
  const int row = base_rows[query];
  const int added_count = additional_counts[query];
  if (row < 0 || row >= cached_rows || added_count < 0 || added_count > additional_capacity) {
    if (lane == 0) status[query] = 1;
    return;
  }
  const int base_count = base_counts[row];
  if (base_count < 0 || base_count > base_capacity) {
    if (lane == 0) status[query] = 1;
    return;
  }
  const int* base = base_indices + static_cast<int64_t>(row) * base_capacity;
  const int* added = additional_indices + static_cast<int64_t>(query) * additional_capacity;
  const int total = base_count + added_count;
  for (int column = lane; column < total; column += kThreads) {
    if ((column < base_count ? base[column] : added[column - base_count]) < 0) atomicExch(&failed, 1);
  }
  __syncthreads();
  if (failed) {
    if (lane == 0) status[query] = 1;
    return;
  }
  for (int slot = lane; slot < hash_capacity; slot += kThreads) {
    keys[slot] = -1;
    first[slot] = INT_MAX;
  }
  __syncthreads();
  for (int column = lane; column < total; column += kThreads) {
    const int value = column < base_count ? base[column] : added[column - base_count];
    int slot = HashSlot(value, hash_capacity);
    bool inserted = false;
    for (int probe = 0; probe < hash_capacity; ++probe) {
      const int previous = atomicCAS(keys + slot, -1, value);
      if (previous == -1 || previous == value) {
        atomicMin(first + slot, column);
        inserted = true;
        break;
      }
      slot = (slot + 1) & (hash_capacity - 1);
    }
    if (!inserted) atomicExch(&failed, 2);
  }
  __syncthreads();
  if (failed) {
    if (lane == 0) status[query] = 2;
    return;
  }
  for (int chunk = 0; chunk < total; chunk += kThreads) {
    const int column = chunk + lane;
    const int value = column < total ? (column < base_count ? base[column] : added[column - base_count]) : -1;
    const int slot = column < total ? FindSlot(keys, value, hash_capacity) : -1;
    const int unique = slot >= 0 && first[slot] == column;
    const int rank = InclusiveScan(flags, unique);
    if (unique && emitted + rank <= output_capacity) output[emitted + rank - 1] = value;
    __syncthreads();
    if (lane == 0) emitted += flags[kThreads - 1];
    __syncthreads();
  }
  if (emitted > output_capacity) {
    for (int column = lane; column < output_capacity; column += kThreads) output[column] = -1;
    if (lane == 0) status[query] = 2;
  } else if (lane == 0) {
    counts[query] = emitted;
  }
}

}  // namespace

cudaError_t LaunchPackedSparseAttentionIndexerMergeIndices(
    cudaStream_t stream, const int32_t* base_indices, const int32_t* base_counts,
    const int32_t* base_rows, const int32_t* additional_indices, const int32_t* additional_counts,
    int32_t* indices, int32_t* counts, int32_t* status, int32_t* workspace,
    int32_t cached_rows, int32_t base_capacity, int32_t additional_capacity,
    int32_t query_count, int32_t output_capacity, int32_t hash_capacity) {
  if (query_count == 0) return cudaSuccess;
  MergeIndicesKernel<<<query_count, kThreads, 0, stream>>>(
      base_indices, base_counts, base_rows, additional_indices, additional_counts,
      indices, counts, status, workspace, cached_rows, base_capacity, additional_capacity,
      output_capacity, hash_capacity);
  return cudaGetLastError();
}

cudaError_t LaunchPackedSparseAttentionIndexerMerge(
    cudaStream_t stream, const int32_t* base_indices, const int32_t* base_counts,
    const int32_t* base_rows, const int32_t* starts, const int32_t* ends,
    int32_t* indices, int32_t* counts, int32_t* status, int32_t* workspace,
    int32_t cached_rows, int32_t base_capacity, int32_t query_count,
    int32_t output_capacity, int32_t hash_capacity) {
  if (query_count == 0) return cudaSuccess;
  MergeRangeKernel<<<query_count, kThreads, 0, stream>>>(
      base_indices, base_counts, base_rows, starts, ends, indices, counts, status,
      workspace, cached_rows, base_capacity, output_capacity, hash_capacity);
  return cudaGetLastError();
}

}  // namespace onnxruntime::contrib::cuda