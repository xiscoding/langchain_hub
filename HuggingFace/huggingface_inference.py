from transformers import AutoTokenizer
import transformers
import torch
from transformers import PreTrainedTokenizerFast

#PATH TO MODEL: 
"""
Review Model ID: Ensure the model_id variable in your code matches 
the exact directory name where the downloaded model files are located.
"""
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2"
# model_name = "Meta-Llama-3-8B-Instruct-8"
# Path to tokenizer files (replace with your actual paths)
tokenizer_config_path = "/home/xdoestech/llama3_hub/LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2/tokenizer_config.json"
tokenizer_vocab_path = "/home/xdoestech/llama3_hub/LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2/tokenizer.json"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

