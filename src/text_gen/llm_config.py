import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class LLMModelConfig:
    model_name: str
    system_prompt: str
    instruction: str
    text_qa_template: str
    generate_config: Dict

    def __post_init__(self):
        self.text_qa_template = self.text_qa_template.format(
            instruction=self.instruction,
            context_str='{context_str}',
            query_str='{query_str}'
        )


nvidia_llama3_8b_config = LLMModelConfig(
    model_name = "nvidia/Llama3-ChatQA-2-8B",
    system_prompt = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.",
    instruction = "Please give a full and complete answer for the question just based on the context for answer. Give me structured answers.",
    text_qa_template = (
        "{instruction}"
        " Context for answer: "
        "{context_str}. "
        "{query_str}"
    ),
    generate_config = {
        'temperature': 0.2,
        'top_k': 30,
        'top_p': 0.9,
        'repetition_penalty': 1.12,
        'presence_penalty': 1.0,
        'frequency_penalty': 1.0,
    }
)


mistral_nemo_12b_inst_config = LLMModelConfig(
    model_name="mistralai/Mistral-Nemo-Instruct-2407",
    system_prompt="System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.",
    instruction = "Please give a full and complete answer for the question just based on the context for answer.",
    text_qa_template = (
        "{instruction}"
        " Context for answer: "
        "{context_str}. "
        "{query_str}"
    ),
    generate_config = {
        'temperature': 0.2,
        'top_k': 30,
        'top_p': 0.9,
        'repetition_penalty': 1.12,
        'presence_penalty': 1.0,
        'frequency_penalty': 1.0,
    }
)


qwen_25_7b_inst_config = LLMModelConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    system_prompt="system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    instruction="Please give a full and complete answer for the question. For the answer, be sure to use only the presented context",
    text_qa_template = (
        "{instruction}"
        " Context for answer: "
        "{context_str}. "
        "{query_str}"
    ),
    generate_config = {
        'temperature': 0.2,
        'top_k': 30,
        'top_p': 0.9,
        'repetition_penalty': 1.12,
        'presence_penalty': 1.0,
        'frequency_penalty': 1.0,
    }
)


gemma_2_9b_inst_config = LLMModelConfig(
    model_name="google/gemma-2-9b-it",
    system_prompt="System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.",
    instruction="Please give a full and complete answer for the question. For the answer, be sure to use only the presented context",
    text_qa_template = (
        "{instruction}"
        " Context for answer: "
        "{context_str}. "
        "{query_str}"
    ),
    generate_config = {
        'temperature': 0.2,
        'top_k': 30,
        'top_p': 0.9,
        'repetition_penalty': 1.12,
        'presence_penalty': 1.0,
        'frequency_penalty': 1.0,
    }
)


models: List[LLMModelConfig] = [
    nvidia_llama3_8b_config,
    qwen_25_7b_inst_config,
    mistral_nemo_12b_inst_config,
    gemma_2_9b_inst_config
]

models_config = {
    config.model_name: config for config in models
}

vllm_config = {
    "swap_space": 1,
    "gpu_memory_utilization": 0.9,
    'device':'cuda',
    'enable_prefix_caching': True,
    'enforce_eager': True,
    'max_model_len': 32000,
}
