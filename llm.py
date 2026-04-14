import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def model_download():
    # 指定模型ID
    model_id = "Qwen/Qwen1.5-0.5B-Chat"

    # 指定下载路径
    cache_dir = r"D:\hf_models"
    os.makedirs(cache_dir, exist_ok=True)

    # 设置设备，优先使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)

    # 加载模型，并将其移动到指定设备
    model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir=cache_dir).to(device)

    print("模型和分词器加载完成！")
    return tokenizer, model, device

def token_msg_example(device):
    # 准备对话输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍你自己。"}
    ]

    # 使用分词器的模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    print("编码后的输入文本:")
    print(model_inputs)
    return model_inputs

def llm_response(tokenizer, model,model_inputs):
    # 基于input生成output
    generate_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    # 截断回答，只保留output部分
    generate_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generate_ids)
    ]

    # 解码: token -> 人类文字
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

    print("\n模型的回答:")
    print(response)

if __name__ == "__main__":
    tokenizer, model,device = model_download()
    model_inputs = token_msg_example(device=device)
    llm_response(tokenizer, model,model_inputs)
    
