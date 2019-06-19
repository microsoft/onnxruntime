// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "non_max_suppression_impl.h"
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace cuda {

template <class... Args>
inline void DebugPring(const char* fmt, Args... args) {
  printf(fmt, std::forward<Args>(args)...);
}

#ifdef __CUDACC_DEBUG__
#define LocalAssert(cond)                 \
  if (!(cond)) {                          \
    DebugPring("Assert failed: ", #cond); \
  }
#else
#define LocalAssert(cond)
#endif

// XXX: Move to common?
template <class T>
class FixesSizeContainer {
  int alloc_size_; // Allocation size
  char* buffer_ = nullptr;
  int size_ = 0;  // Number of contained elements

  __device__ __host__ FixesSizeContainer() : alloc_size_(0), buffer_(nullptr), size_(0) {}

 public:
  __device__ __host__ explicit FixesSizeContainer(int alloc_size) : 
    FixesSizeContainer() {
    Allocate(alloc_size);
    alloc_size_ = alloc_size;
  }

  FixesSizeContainer(const FixesSizeContainer&) = delete;
  FixesSizeContainer& operator=(const FixesSizeContainer&) = delete;
  __device__ __host__ FixesSizeContainer(FixesSizeContainer&& o) noexcept
      : FixesSizeContainer() {
    *this = std::move(o);
  }
  __device__ __host__ FixesSizeContainer& operator=(FixesSizeContainer&& o) noexcept {
    if (this != &o) {
      if (buffer_ != nullptr) {
        array_destruct();
        delete[] buffer_;
        buffer_ = nullptr;
        alloc_size_ = 0;
      }
      std::swap(alloc_size_, o.alloc_size_);
      std::swap(buffer_, o.buffer_);
      std::swap(size_, o.size_);
    }
    return *this;
  }

  __device__ __host__ ~FixesSizeContainer() noexcept {
    array_destruct();
    delete[] buffer_;
  }

  __device__ __host__ int size() const { return size_; }

  __device__ __host__ T& operator[](int idx) noexcept {
    LocalAssert(idx < 0 || idx > size_ - 1);
    return *GetTyped()[idx];
  }

  __device__ __host__ const T& operator[](int idx) const noexcept {
    LocalAssert(idx < 0 || idx > size_ - 1);
    return *GetTyped()[idx];
  }

  __device__ __host__ void push_back(const T& v) {
    LocalAssert(size_ < alloc_size_);
    if (size_ < (alloc_size_)) {
      new (GetTyped(size_)) T(v);
      ++size_;
    }
  }
  __device__ __host__ T* begin() { return GetTyped(0); }
  __device__ __host__ const T* begin() const { return GetTyped(0); }
  __device__ __host__ T* end() { return GetTyped(size_); }
  __device__ __host__ const T* end() const { return GetTyped(size_); }

 private:
  __device__ __host__ T* GetTyped(int idx) const {
    return reinterpret_cast<T*>(buffer_) + idx;
  }

  __device__ __host__ void Allocate(int size) {
    buffer_ = new char[sizeof(T) * size];
  }

  __device__ __host__ void array_destruct() noexcept {
    T* p = GetTyped(0);
    while (size_-- > 0) {
      (p + size_)->~T();
    }
    size_ = 0;
  }
};

struct ScoreIndexPair {
  float score_{};
  int64_t index_{};

  ScoreIndexPair() = default;
  ~ScoreIndexPair() = default;
  __device__ explicit ScoreIndexPair(float score, int64_t idx) noexcept : score_(score), index_(idx) {}
  ScoreIndexPair(const ScoreIndexPair&) = default;
  ScoreIndexPair& operator=(const ScoreIndexPair&) = default;
  // We reverse the meaning so thrust::sort below sorts in descending order
  __device__ bool operator<(const ScoreIndexPair& rhs) const noexcept {
    return score_ > rhs.score_;
  }
};

// This structure has combined members so we can simplify shared
// memory calculation and allocation, so instead of two shared arrays
// we allocate onw array of structures.
struct ScoresAndIndecies {
  float    score_;  // Score value loaded from global memory
  int32_t  sorted_index_; // Box index in a score_sorted_order
  bool     is_selected_; // See if this was selected
};

// This kernel will be launched in the following configuration
// blocks number is a num_batches * num_classes * num_boxes and the amount
// of calculated shared memory
__global__ void NonMaxSuppressionImplDevice(const float* boxes_data, const float* scores_data,
                                            int64_t* output_data,
                                            int32_t num_batches, int32_t num_classes,
                                            int32_t num_boxes,
                                            int32_t center_point_box,
                                            int32_t max_output_boxes_per_class,
                                            const fast_divmod batches,
                                            const fast_divmod classes,
                                            bool has_score_threshold,
                                            float score_threshold,
                                            float iou_threshold, CUDA_LONG N) {
  // This is allocated by runtime when we launch the kernel. The lifespan of this memory is
  // per thread block
  extern __shared__ ScoresAndIndecies scores_indecies[];

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // Load the scores for each block in individual thread

  // N = num_batches * num_classes
  // batch_index = id % num_batches
  int batch_index = 0;
  int whole_batches = 0;
  batches.divmod(id, whole_batches, batch_index);

  // class_index = id % num_classes
  int whole_classes = 0;
  int class_index = 0;
  classes.divmod(id, whole_classes, class_index);

  FixesSizeContainer<ScoreIndexPair> selected_scores(num_boxes);
  int box_score_offset = (batch_index * num_classes + class_index) * num_boxes;
  const auto* class_scores = scores_data + box_score_offset;
  if (has_score_threshold) {
    for (int64_t box_index = 0; box_index < int64_t{num_boxes}; ++box_index, ++class_scores) {
      if (*class_scores > score_threshold) {
        selected_scores.push_back(ScoreIndexPair(*class_scores, box_index));
      }
    }
  } else {
    for (int64_t box_index = 0; box_index < int64_t{num_boxes}; ++box_index, ++class_scores) {
      selected_scores.push_back(ScoreIndexPair(*class_scores, box_index));
    }
  }
  // We lack priority queue
  // std::sort(selected_scores.begin(), selected_scores.end());

  // Compute for each of the classes/batches
  ScoreIndexPair next_top_score;
  FixesSizeContainer<int64_t> selected_indicies_inside_class(max_output_boxes_per_class);
  int box_offset = batch_index * num_classes * num_boxes * 4;

  // Get the next box with top score, filter by iou_threshold
  const float* class_boxes = boxes_data + box_offset;
  for (const ScoreIndexPair& top : selected_scores) {
    bool selected = true;
    // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
    for (int64_t selected_index : selected_indicies_inside_class) {
      if (nms_helpers::SuppressByIOU(class_boxes, selected_index, top.index_,
                                     center_point_box, iou_threshold)) {
        selected = false;
        break;
      }
    }

    if (selected) {
      if (max_output_boxes_per_class > 0 &&
          static_cast<int32_t>(selected_indicies_inside_class.size()) >= max_output_boxes_per_class) {
        break;
      }
      selected_indicies_inside_class.push_back(next_top_score.index_);
    }
  }  //for

  // Assign dynamically allocated memory to the shared array
  //

  // Increment the count
  // __threadfence();

  // __syncthreads();   // sync inside the block
  // One thread copies results into output
}

void NonMaxSuppressionImpl(const PrepareContext& pc, int64_t max_boxes_per_class, float score_threshold,
                           float iou_threshold, int64_t* output_data) {
  auto N = pc.num_batches_ * pc.num_classes_;
  fast_divmod batches(static_cast<int32_t>(pc.num_batches_));
  fast_divmod classes(static_cast<int32_t>(pc.num_classes_));

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  // XXX: How do we gather the input into one big buffer? Collect pairs of buffer/size?
  // NonMaxSuppressionImplDevice<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>
}

}  // namespace cuda
}  // namespace onnxruntime
