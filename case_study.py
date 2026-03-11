import torch
import torch.nn as nn
from transformers import LlamaTokenizer
from models.modeling_aoe_dive import PrunedLlamaSMoEForCausalLM
import csv

model_dir = "/data9/fengyuchen/new_pruned_smoe_models/tinyllama_dive/dive_8_1_0.5"
tokenizer = LlamaTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = PrunedLlamaSMoEForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
model = nn.DataParallel(model).to(device)

prompt = "Neuronal signals via the hepatic vagus nerve contribute to the development of steatohepatitis and protection against obesity in HFD fed Pemt(-/-) mice."
inputs = tokenizer(prompt, return_tensors="pt")

inputs = {key: value.to(device) for key, value in inputs.items()}

tokens = tokenizer.encode(prompt)
tokens_str = tokenizer.convert_ids_to_tokens(tokens)

with open('/data9/fengyuchen/AoE-DIVE/aoe_cases.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    file.seek(0, 2)
    if file.tell() == 0:
        writer.writerow(['Original Text', 'Token ID', 'Token'])

    for token_id, token in zip(tokens, tokens_str):
        writer.writerow([prompt, token_id, token])

model(**inputs)

print("Tokenized input is saved.")
