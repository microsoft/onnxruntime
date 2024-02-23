// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "contrib_ops/cpu/transformers/sequences.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

void Sequences::Init(gsl::span<int32_t> buffer, int batch_beam_size, int sequence_length, int max_length) {
  size_t sequences_size = SafeInt<size_t>(batch_beam_size) * max_length;
  assert(buffer.size() == sequences_size + sequences_size);

  sequences[0] = buffer.subspan(0, sequences_size);
  sequences[1] = buffer.subspan(sequences_size);

  current_sequences_buffer = 0;

  batch_beam_size_ = batch_beam_size;
  max_length_ = max_length;
  current_length_ = sequence_length;
}

void Sequences::InitDevice(gsl::span<int32_t> buffer) {
  device_sequences[0] = buffer.subspan(0, buffer.size() / 2);
  device_sequences[1] = buffer.subspan(buffer.size() / 2);
}

gsl::span<const int32_t> Sequences::GetSequence(int beam_index) const {
  gsl::span<const int32_t> buffer = sequences[current_sequences_buffer];
  return buffer.subspan(SafeInt<size_t>(beam_index) * max_length_, static_cast<gsl::index>(current_length_));
}

int Sequences::GetSequenceLength() const {
  return current_length_;
}

#ifdef DEBUG_GENERATION
void Sequences::PrintSequences(const IConsoleDumper* dumper) const {
  for (int i = 0; i < batch_beam_size_; i++) {
    gsl::span<const int32_t> sequence = GetSequence(i);
    dumper->Print("sequences", i, false);
    dumper->Print(nullptr, sequence.data(), 1, current_length_);
  }
}
#endif

void Sequences::AppendNextTokenToSequences(
    gsl::span<int32_t>& beam_indices,
    gsl::span<int32_t>& beam_next_tokens) {
  gsl::span<const int32_t> input = sequences[current_sequences_buffer];
  gsl::span<int32_t> output = sequences[current_sequences_buffer ^ 1];

  for (int i = 0; i < batch_beam_size_; i++) {
    int beam_index = beam_indices[i];
    gsl::span<const int32_t> source = input.subspan(SafeInt<size_t>(beam_index) * max_length_,
                                                    static_cast<gsl::index>(current_length_));
    gsl::span<int32_t> target = output.subspan(SafeInt<size_t>(i) * max_length_,
                                               static_cast<gsl::index>(current_length_));
    gsl::copy(source, target);

    // Append next token to each beam.
    output[SafeInt<size_t>(i) * max_length_ + current_length_] = beam_next_tokens[i];
  }

  ++current_length_;

  // Rotate buffer for next round.
  current_sequences_buffer ^= 1;
}

void Sequences::AppendNextTokenToSequences(gsl::span<int32_t>& next_tokens) {
  auto output = sequences[0];

  // Append next token to each sequence.
  for (int i = 0; i < batch_beam_size_; i++) {
    output[SafeInt<size_t>(i) * max_length_ + current_length_] = next_tokens[i];
  }

  ++current_length_;
}

void Sequences::AfterDeviceAppendedNextToken() {
  ++current_length_;
  current_sequences_buffer ^= 1;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
