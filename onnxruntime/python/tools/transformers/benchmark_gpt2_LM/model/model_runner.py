from logging import exception
import torch
import argparse

from generator import Generator
from wrapper.onnx import OnnxModelWrapper
from tokenizer import Tokenizer

class ModelRunner:
    def __init__(self) -> None:
        pass

    def _process_results(
        self, output_ids: torch.Tensor, output_probs: torch.Tensor, tokenizer: Tokenizer,
        last_complete_word_position: int):

        try:
            #TODO why is this expanding to a 3D tensor
            if output_ids.dim() == 2: # one output
                output_ids = output_ids[:, None]

            returns = []
            returns_probs = []
            for i, output_i in enumerate(output_ids):
                returns_i = []
                probs_i = []
                for j, output_ij in enumerate(output_i):
                    # TODO give an option to user args.sequence_prob_threshold
                    # This can be used to filter out results with very less probability instead of zero
                    if returns_i and output_probs[i, j] < 0:
                        break
                    output_ij = output_ij.tolist()
                    for k in range(len(output_ij)):
                        if output_ij[k] == tokenizer.eos_token_id:
                            output_ij = output_ij[:k]
                            break

                    output_ij = output_ij[last_complete_word_position:]
                    output_text = tokenizer.decode(output_ij).rstrip()
                    if output_text not in returns_i:
                        returns_i.append(output_text)
                        probs_i.append(output_probs[i, j].item())

                returns.append(returns_i)
                returns_probs.append(probs_i)
        except Exception as e:
            raise("Caught exception during processing results:" + str(e))

        return returns, returns_probs


    @torch.no_grad()
    def autocomplete(
        self, args : argparse, model : OnnxModelWrapper, tokenizer : Tokenizer,
        input_text : str, pad_token_id : int, is_onnx_model = False):
        """Entry point for model runner, this kicks of beam search and processes the results
        """

        try:
            input_ids, first_token_masks, last_complete_word_position = tokenizer.encode_text_with_partial_word(input_text, pad_token_id, args.device)

            # This is created for each query, currently there is no performance impact for this.
            generator = Generator(
                max_length=input_ids.size(1) + args.num_words, num_return_sequences=args.num_suggestions, num_beams=args.num_beams,
                pad_token_id=tokenizer.eos_token_id, eos_token_ids=[tokenizer.eos_token_id], length_penalty=args.length_penalty,
                is_onnx_model=is_onnx_model, tokenizer = tokenizer)

            #TODO, currently ids and probs are generated in case of beam search
            # need to handle the case of beam search op which probably only gives output_ids
            output_ids, output_probs = generator.generate(model, input_ids, first_token_masks=first_token_masks)

            outputs, output_probs = self._process_results(output_ids, output_probs, tokenizer, last_complete_word_position)
        except Exception as e:
            raise e
        
        #After this output probs doesn't really have an importance, just returning outputs
        return outputs
