from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# prompt = "My favourite condiment is"
model.to(device)

print("model loaded")

while True:
    prompt = input()
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True)
    print(tokenizer.batch_decode(generated_ids)[0])
