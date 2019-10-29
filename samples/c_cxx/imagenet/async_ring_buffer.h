// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iostream>
#include <cstddef>
#include <mutex>
#include "controller.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "single_consumer.h"
#include "runnable_task.h"

template <typename InputIterator>
class AsyncRingBuffer {
 private:
  static VOID NTAPI ThreadPoolEntry(_Inout_ PTP_CALLBACK_INSTANCE pci, _Inout_opt_ PVOID data, _Inout_ PTP_WORK work) {
    CloseThreadpoolWork(work);
    (*(RunnableTask*)data)(pci);
  }

  template <typename T = float>
  static size_t CalcItemSize(const std::vector<int64_t>& tensor_shape) {
    int64_t r = 1;
    for (int64_t i : tensor_shape) r *= i;
    return static_cast<size_t>(r) * sizeof(T);
  }

  enum class BufferState { EMPTY,
                           FILLING,
                           FULL,
                           TAKEN };
  const size_t batch_size_;
  using InputType = typename InputIterator::value_type;
  DataProcessing* p_;
  OutputCollector<InputType>* c_;
  size_t capacity_;
  struct QueueItem {
    Ort::Value value{nullptr};
    std::vector<InputType> taskid_list;

    QueueItem() = default;
    QueueItem(const QueueItem&) = delete;
    QueueItem& operator=(const QueueItem&) = delete;
  };
  //A list of tensors with equal tensor shape
  SingleConsumerFIFO<QueueItem> queue_;
  using TensorListEntry = typename SingleConsumerFIFO<QueueItem>::ListEntry;
  Controller& threadpool_;
  std::vector<int64_t> CreateTensorShapeWithBatchSize(const std::vector<int64_t>& input, size_t batch_size) {
    std::vector<int64_t> shape(input.size() + 1);
    shape[0] = batch_size;
    size_t len = shape.size();
    for (size_t i = 1; i != len; ++i) {
      shape[i] = input[i - 1];
    }
    return shape;
  }
  std::mutex m;

  /**
   * A collection of buffers with equal size.
   */
  struct BufferManager {
    size_t capacity_;
    size_t item_size_in_bytes_;
    size_t write_index_ = 0;
    std::vector<BufferState> buffer_state;
    std::vector<InputType> input_task_id_for_buffers_;

    // TODO: if there is an alignment requirement, this buffer need do padding between the tensors.
    std::vector<uint8_t> buffer_;

    BufferManager(size_t capacity, size_t item_size_in_bytes)
        : capacity_(capacity),
          item_size_in_bytes_(item_size_in_bytes),
          buffer_state(capacity, BufferState::EMPTY),
          input_task_id_for_buffers_(capacity),
          buffer_(item_size_in_bytes * capacity) {}

    size_t GetId(_In_ const uint8_t* p) const { return (p - buffer_.data()) / item_size_in_bytes_; }
    size_t GetItemSizeInBytes() const { return item_size_in_bytes_; }
    bool CompareAndSet(size_t i, BufferState old, BufferState new_state) {
      if (buffer_state[i] != old) return false;
      buffer_state[i] = new_state;
      return true;
    }

    bool CompareAndSet(size_t index, size_t index_end, BufferState old, BufferState new_state) {
      assert(index_end >= index);
      for (size_t i = index; i != index_end; ++i) {
        if (buffer_state[i] != old) return false;
      }
      for (size_t i = index; i != index_end; ++i) {
        buffer_state[i] = new_state;
      }
      return true;
    }

    bool TakeRange(size_t index, size_t index_end, std::vector<InputType>& task_id_list) {
      assert(index_end >= index);
      if (!CompareAndSet(index, index_end, BufferState::FULL, BufferState::TAKEN)) {
        return false;
      }
      auto* p = &input_task_id_for_buffers_[index];
      auto* p_end = p + (index_end - index);
      task_id_list.assign(p, p_end);
      return true;
    }

    _Success_(return ) bool TakeAllRemain(_Out_ uint8_t** begin, std::vector<InputType>& task_id_list) {
      auto iter =
          std::find_if(buffer_state.begin(), buffer_state.end(), [](BufferState s) { return s == BufferState::FULL; });
      if (iter == buffer_state.end()) return false;
      auto iter_end = std::find_if(iter, buffer_state.end(), [](BufferState s) { return s != BufferState::FULL; });

      *begin = &buffer_[iter - buffer_state.begin()];
      if (!TakeRange(iter - buffer_state.begin(), iter_end - buffer_state.begin(), task_id_list)) {
        throw std::runtime_error("internal error");
      }
      size_t remain = std::count_if(buffer_state.begin(), buffer_state.end(),
                                    [](BufferState s) { return s != BufferState::TAKEN && s != BufferState::EMPTY; });
      if (remain != 0) {
        throw std::runtime_error("the buffer contains multiple non-contiguous region");
      }
      return true;
    }

    uint8_t* Begin() { return buffer_.data(); }

    /*
     * Get a buffer pointer and set its state to FILLING
     * \param taskid
     * \return Pointer to the buffer
     */
    uint8_t* Next(InputType taskid) {
      for (size_t i = 0; i != capacity_; ++i) {
        size_t index = (write_index_ + i) % capacity_;
        if (buffer_state[i] == BufferState::EMPTY) {
          buffer_state[i] = BufferState::FILLING;
          input_task_id_for_buffers_[i] = taskid;
          return &buffer_[index * item_size_in_bytes_];
        }
      }
      return nullptr;
    }
  };
  BufferManager buffer_;
  InputIterator input_begin_;
  const InputIterator input_end_;
  // unsafe
  bool is_input_eof() const { return input_end_ == input_begin_; }
  size_t parallelism = 8;
  size_t current_running_downloders = 0;

  void ReturnAndTake(TensorListEntry*& input_tensor) {
    std::lock_guard<std::mutex> g(m);
    if (input_tensor != nullptr) {
      size_t tensor_id = queue_.Return(input_tensor);
      size_t buffer_id = tensor_id * batch_size_;
      if (!buffer_.CompareAndSet(buffer_id, buffer_id + batch_size_, BufferState::TAKEN, BufferState::EMPTY)) {
        throw std::runtime_error("ReturnAndTake: internal state error");
      }
    }
    input_tensor = queue_.Take();
  }

  void OnDownloadFinished(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci, const uint8_t* dest) {
    size_t buffer_id = buffer_.GetId(dest);
    TensorListEntry* input_tensor = nullptr;
    {
      std::lock_guard<std::mutex> g(m);
      --current_running_downloders;
      if (!buffer_.CompareAndSet(buffer_id, BufferState::FILLING, BufferState::FULL)) {
        throw std::runtime_error("ReturnAndTake: internal state error");
      }
      size_t tensor_id = buffer_id / batch_size_;
      std::vector<InputType> task_id_list;
      buffer_id = tensor_id * batch_size_;
      if (buffer_.TakeRange(buffer_id, buffer_id + batch_size_, task_id_list)) {
        queue_.Put(tensor_id, [&task_id_list](QueueItem& i) {
          i.taskid_list = task_id_list;
        });
        input_tensor = queue_.Take();
      }
    }

    bool eof = false;
    while (threadpool_.IsRunning()) {
      if (!eof) {
        int tasks = StartDownloadTasks();
        if (tasks < 0) {
          threadpool_.SetFailBit(pci, "Schedule download task failed");
          return;
        }
        if (tasks == 0) {
          threadpool_.SetEof(pci);
          eof = true;
        }
      }
      if (input_tensor == nullptr) {
        break;
      }
      (*c_)(input_tensor->value.taskid_list, input_tensor->value.value);
      ReturnAndTake(input_tensor);
    }
  }

  void Fail(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci, const char* errmsg) {
    threadpool_.SetFailBit(pci, errmsg);
  }

 public:
  AsyncRingBuffer(size_t batch_size, size_t capacity, Controller& threadpool, const InputIterator& input_begin,
                  const InputIterator& input_end, DataProcessing* p, OutputCollector<InputType>* c)
      : batch_size_(batch_size),
        p_(p),
        c_(c),
        capacity_((capacity + batch_size_ - 1) / batch_size_ * batch_size_),
        queue_(capacity_ / batch_size_),
        threadpool_(threadpool),
        buffer_(capacity_, CalcItemSize(p->GetOutputShape(1))),
        input_begin_(input_begin),
        input_end_(input_end) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    uint8_t* output_data = buffer_.Begin();
    std::vector<int64_t> input_shape = p_->GetOutputShape(batch_size_);
    size_t off = CalcItemSize(input_shape);
    queue_.Init([&memory_info, off, &output_data, &input_shape](QueueItem& e) {
      e.value = Ort::Value::CreateTensor(memory_info, reinterpret_cast<float*>(output_data), off, input_shape.data(), input_shape.size());
      output_data += off;
    });
  }

  void ProcessRemain() {
    queue_.Release();
    c_->ResetCache();

    uint8_t* output_data;
    std::vector<InputType> task_id_list;
    if (!buffer_.TakeAllRemain(&output_data, task_id_list)) return;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    size_t count = task_id_list.size();
    assert(count != 0);
    std::vector<int64_t> input_shape = p_->GetOutputShape(count);
    size_t len = CalcItemSize(input_shape);
    Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, reinterpret_cast<float*>(output_data), len, input_shape.data(), input_shape.size());
    (*c_)(task_id_list, input_tensor);
  }

  /**
   * call this function when a download task is just finished or any buffer became FREE.
   * \return 0 EOF. No more download task to schedule
   *         1 OK
   *         -1 ERROR
   */
  int StartDownloadTasks() {
    class DownloadTask : public RunnableTask {
     public:
      AsyncRingBuffer* requester;
      InputType source;
      uint8_t* dest;
      DownloadTask(AsyncRingBuffer* r, const InputType& s, uint8_t* d) : requester(r), source(s), dest(d) {}

      void operator()(_In_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci) noexcept override {
        AsyncRingBuffer* r = requester;
        InputType s = source;
        uint8_t* d = dest;
        delete this;
        try {
          (*r->p_)(&s, d, r->buffer_.GetItemSizeInBytes());
          r->OnDownloadFinished(pci, d);
        } catch (const std::exception& ex) {
          fprintf(stderr, "%s\n", ex.what());
          r->Fail(pci, ex.what());
        }
      }
    };

    // search empty slots, launch a download task for each of them
    std::vector<DownloadTask*> tasks_to_launch;
    bool is_eof = false;
    {
      std::lock_guard<std::mutex> g(m);
      // if we have
      // 1. cpu  (current_running_downloders < parallelism)
      // 2. memory (buffer available)
      // 3. input_task
      // then schedule a download task to the thread pool
      for (; current_running_downloders + tasks_to_launch.size() < parallelism && !is_input_eof();
           ++input_begin_, ++current_running_downloders) {
        uint8_t* b = buffer_.Next(*input_begin_);
        if (b == nullptr) break;  // no empty buffer
        tasks_to_launch.push_back(new DownloadTask(this, *input_begin_, b));
      }
      is_eof = is_input_eof();
    }

    for (DownloadTask* p : tasks_to_launch) {
      if (!threadpool_.RunAsync(ThreadPoolEntry, p)) {
        return -1;
      }
    }

    if (is_eof) {
      return 0;
    }
    return 1;
  }
};
