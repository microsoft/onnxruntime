#pragma once

#include "gsl/gsl"
#include "beam_search_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// This class keeps track of sequences generated.
class Sequences : public ISequences {
 public:
  Sequences() {}

  // Initialize the sequence.
  void Init(gsl::span<int32_t> buffer, int batch_beam_size, int sequence_length, int max_length);

  // Returns a sequence of word IDs for a given beam index ( beam_index < batch_beam_size).
  gsl::span<const int32_t> GetSequence(int beam_index) const override;

  // Returns current sequence length.
  int GetSequenceLength() const override;

#ifdef DEBUG_BEAM_SEARCH
  // Print the sequences to StdOut in debug mode
  void PrintSequences(const IConsoleDumper* dumper) const;
#endif

  // Select sequences based on beam indices, then append next token to selected sequences.
  void AppendNextTokenToSequences(
      gsl::span<int32_t>& beam_indices,
      gsl::span<int32_t>& beam_next_tokens);

 private:
  // Two buffers of shape (batch_size, num_beams, max_seq_length) to store sequences.
  // At each time, there is only one buffer is active. The other one will be active in next token.
  // Each AppendNextTokenToSequences call will trigger a rotation of active buffer.
  gsl::span<int32_t> sequences[2];

  // Index (either 0 or 1) of two buffers that is currently is active.
  int current_sequences_buffer;

  int batch_beam_size_;
  int max_length_;
  int current_length_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime