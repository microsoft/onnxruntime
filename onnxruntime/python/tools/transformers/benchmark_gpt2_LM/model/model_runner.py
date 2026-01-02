from logging import exception
import torch
import argparse

from generator import Generator
from wrapper.base import BaseModelWrapper
from tokenizer import Tokenizer

GPT2_MAX_ENCODING_LENGTH=1024

class ModelRunner:
    def __init__(self) -> None:
        self._MAX_ENCODING_LENGTH = GPT2_MAX_ENCODING_LENGTH

    def _process_results(
        self, output_ids: torch.Tensor, output_probs: torch.Tensor, tokenizer: Tokenizer,
        last_complete_word_position: int):
        """
        takes the output ids and output probabilities and decodes them. output probability is only needed
        to see if it has a negative probability to not include the result.
        """
        try:
            #TODO why is this expanding to a 3D tensor
            if output_ids.dim() == 2: # one output
                output_ids = output_ids[:, None]

            returns = []
            for i, output_i in enumerate(output_ids):
                returns_i = []
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

                returns.append(returns_i)
        except Exception as e:
            raise Exception(f"Caught exception during processing results: {str(e)}")

        return returns


    @torch.no_grad()
    def run_beam_search_to_extract_suggestions(
        self, args : argparse, model : BaseModelWrapper, tokenizer : Tokenizer,
        input_text : str, pad_token_id : int, is_data_str: True):
        """Entry point for model runner, this kicks of beam search and processes the results
        """

        try:
            input_ids, first_token_masks, last_complete_word_position = tokenizer.encode_text_with_partial_word(input_text, pad_token_id, args.device)

            if len(input_ids) > self._MAX_ENCODING_LENGTH:
                raise Exception(f"Input string is tokenized into {len(input_ids)} more than accepted value {self._MAX_ENCODING_LENGTH}")

            # This is created for each query, currently there is no performance impact for this.
            generator = Generator(
                max_length=input_ids.size(1) + args.num_words, num_return_sequences=args.num_suggestions, num_beams=args.num_beams,
                pad_token_id=tokenizer.eos_token_id, eos_token_ids=[tokenizer.eos_token_id], length_penalty=args.length_penalty,
                tokenizer = tokenizer, device = args.device)

            #TODO, currently ids and probs are generated in case of beam search
            # need to handle the case of beam search op which probably only gives output_ids
            output_ids, output_probs = generator.generate(model, input_ids, first_token_masks=first_token_masks)

            outputs = self._process_results(output_ids, output_probs, tokenizer, last_complete_word_position)
        except Exception as e:
            raise e
        
        #After this output probs doesn't really have an importance, just returning outputs
        return outputs[0]

    def run_model(
        self, args : argparse, model : BaseModelWrapper, tokenizer : Tokenizer,
        input_text : str, pad_token_id : int, is_data_str: True):
        """
        this needs changes to handle the onnx beam search op version
        """
        try:
            input_ids, first_token_masks, last_complete_word_position = tokenizer.encode_text_with_partial_word(input_text, pad_token_id, args.device)

            output_ids, output_probs = model.run(
                input_ids = input_ids,
                max_length= input_ids.size(1) + args.num_words,
                num_sequences = args.num_sequences,
                num_beams = args.num_beams,
                eos_token_id= tokenizer.eos_token_id,
                vocab_mask= first_token_masks,
                temperature = args.temparature,
                length_penalty = args.length_penalty,
                repetition_penalty =args.repetition_penalty)

            outputs = self._process_results(output_ids, output_probs, tokenizer, last_complete_word_position)
        except Exception as e:
            raise e

        return outputs

