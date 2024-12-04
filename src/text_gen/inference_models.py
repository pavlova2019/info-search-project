import os, csv
os.environ['TRANSFORMERS_CACHE'] = "../../hfcache"
os.environ['HF_HOME'] = "../../hfcache"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    GenerationConfig, PreTrainedModel, PreTrainedTokenizer
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cache_dir = "../../hfcache"
data_dir = "../../data"


def preparing_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False
) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        cache_dir=cache_dir
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir)
    config = GenerationConfig.from_pretrained(model_name)
    return model, tokenizer, config


if __name__ == "__main__":
    dataset = load_dataset("TitanMLData/arxiv_qa", cache_dir=cache_dir)["train"].take(400)
    model_names = {
        "nvidia/Llama3-ChatQA-1.5-8B": "nvidia_llama3_8b",  # context 8k
        "Qwen/Qwen2.5-7B-Instruct": "qwen2.5_7b",  # context 32k
        "microsoft/Phi-3.5-mini-instruct": "phi3.5_mini",  # context 128k
        "mistralai/Mistral-Nemo-Instruct-2407": "mistral_nemo_12b"  # context 128k
    }

    system_prompt = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."
    
    system_conversation = {"role": "system", "content": system_prompt}
    for model_name, title in model_names.items():
        model, tokenizer, config = preparing_model(model_name, )

        if not config.max_new_tokens:
            config.max_new_tokens = 512
        
        filename = os.path.join(data_dir, f"{title}_gen.csv")
        with open(filename, "w", encoding="UTF-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "True", "Gen"])
            
            for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
                if item["Text"] and item["Question"]:
                    query = item["Text"] + " " + instruction + " " + item["Question"]
                    conversation_prompt = [
                        system_conversation, 
                        {"role": "user", "content": query}
                    ]
    
                    prompt = tokenizer.apply_chat_template(
                        conversation_prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    )
    
                    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                    with torch.no_grad():
                        output_ids = model.generate(**data, generation_config=config)[0]
                        
                    output_ids = output_ids[data.input_ids.shape[-1]:]
                    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    writer.writerow([idx, item["Response"], output])

                    del output_ids, data
                    gc.collect()
                    torch.cuda.empty_cache()  
                
        del model
        gc.collect()
        torch.cuda.empty_cache()

        

