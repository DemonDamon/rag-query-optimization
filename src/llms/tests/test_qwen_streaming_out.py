# Date    : 2024/7/9 13:56
# File    : test_qwen_streaming_out.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# model_name = "Qwen/Qwen2-7B-Instruct"
model_name = "Qwen/Qwen2-0.5B"
device = "cpu"
# device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(
    **model_inputs,
    streamer=streamer,
    max_new_tokens=512,
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("Generated text:")
for text in streamer:
    print(text, end="", flush=True)