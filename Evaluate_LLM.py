import os
import re
from tqdm import tqdm
import transformers, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GenerationConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-2b")

model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-2b",
                                              trust_remote_code=True,
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True)

model.to(DEVICE)
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
