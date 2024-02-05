import argparse

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="model direcotory, including config.json and tokenizer.model",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="prompt string for the model to generate text from. e.g. 'question: What is the lightest element?'",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_dir = args.model_dir

    model = ORTModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ort_llama2_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda:0")

    sequences = ort_llama2_generator(
        args.prompt,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=256,
        return_full_text=False,
        repetition_penalty=1.1,
    )
    print(sequences[0]["generated_text"])


if __name__ == "__main__":
    main()
