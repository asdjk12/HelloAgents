from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests

model_name = r"D:\hf_models\modelscope\hub\models\Qwen\Qwen3-0___6B"

# 1. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 2. 构造输入
prompt = "请解释一下什么是费马定理. 说明该定理旨在解决什么事情。该定理的公式是什么。"

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 3. 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,      # 开启采样
    temperature=0.7,
    top_p=0.8,
    top_k=20
)

# 4. 截掉输入，只保留新生成内容
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(response)

"""
class ToolCalling():
    def __init__(self) -> None:
        pass

    def arXiv(self, search_element:str, page:int=0, rank:int=5):
        url = "http://export.arxiv.org/api/query"
        
        params = {
            "search_query": search_element,
            "start": page,
            "max_results": rank
        }

        resp = requests.get(url, params, timeout=30)
        print(resp.status_code)
        print(resp.text[:1000])"""

