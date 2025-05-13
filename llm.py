from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_answer(context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Encode the input text
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + 100,  # Generate up to 100 new tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        do_sample=True
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split("Answer:")[-1].strip()
