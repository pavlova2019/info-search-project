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
    template: str = field(init=False)

    def __post_init__(self):
        self.template = self.system_prompt + "\n\n{context_str}\n\nUser: " + self.instruction + " {query_str}\n\nAssistant:"


llama3_config = LLMModelConfig(
    model_name="nvidia/Llama3-ChatQA-1.5-8B",
    system_prompt="System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.",
    instruction="Please give a full and complete answer for the question just based on the previous context.",
    context_window=8192,
    model_kwargs = {
        'torch_dtype': torch.bfloat16
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

models: List[LLMModelConfig] = [llama3_config]

models_config = {
    config.model_name: config for config in models
}
