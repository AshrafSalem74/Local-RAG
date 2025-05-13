from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_answer(context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    output = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)[0]["generated_text"]
    return output.split("Answer:")[-1].strip()
