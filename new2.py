from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = input("ENTER THE TEXT")

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=100,             # Generate up to 100 tokens
    top_k=50,                   # Consider top 50 tokens for diversity
    top_p=0.9,                  # Use nucleus sampling for balanced randomness
    temperature=0.8,            # Add randomness to token selection
    repetition_penalty=1.2,     # Penalize repetitive patterns
    no_repeat_ngram_size=2,     # Avoid repeating n-grams of size 2
    early_stopping=True         # Stop once the model generates an EOS token
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)
