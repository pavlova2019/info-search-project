import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class LLMModelConfig:
    model_name: str
    system_prompt: str
    instruction: str
    context_window: int
    model_kwargs: dict
    generate_config: dict
    template: str

    def __post_init__(self):
        self.template = self.template.format(system_prompt=self.system_prompt, instruction=self.instruction,
                                             context_str='{context_str}', query_str='{query_str}')


nvidia_llama3_8b_config = LLMModelConfig(
    model_name="nvidia/Llama3-ChatQA-2-8B",
    system_prompt="System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.",
    instruction="Please give a full and complete answer for the question.",
    template="<|begin_of_text|>{system_prompt}\n\n{context_str}\n\nUser: {instruction} {query_str}\n\nAssistant:",
    context_window=131072,
    model_kwargs = {
        'torch_dtype': torch.float16
    },
    generate_config={
        # 'temperature': 1.0,
        # 'top_k': 50,
        # 'top_p': 1.0,
        # 'do_sample': False,
        # 'repetition_penalty': 1.0,
        # 'length_penalty': 1.0,
        # 'num_beams': 1,
    }
)


mistral_nemo_12b_inst_config = LLMModelConfig(
    model_name="mistralai/Mistral-Nemo-Instruct-2407",
    system_prompt="System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.",
    instruction="Please give a full and complete answer for the question just based on the previous context.",
    template="<s>[INST]{system_prompt}\n\n{context_str}\n\nUser: {instruction} {query_str}\n\nAssistant:[/INST]",
    context_window=131072,
    model_kwargs = {
        'torch_dtype': torch.bfloat16
    },
    generate_config={}
)


qwen_25_7b_inst_config = LLMModelConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    system_prompt="system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    instruction="Please give a full and complete answer for the question. For the answer, be sure to use only the presented context",
    template="<|im_start|>{system_prompt}<|im_end|>\n<|im_start|>user\n{context_str}\n{instruction} {query_str}<|im_end|>\n<|im_start|>assistant\n",
    context_window=32768,
    model_kwargs = {
        'torch_dtype': torch.bfloat16
    },
    generate_config={}
)


models: List[LLMModelConfig] = [nvidia_llama3_8b_config, qwen_25_7b_inst_config, mistral_nemo_12b_inst_config]

models_config = {
    config.model_name: config for config in models
}
