import src.config as cfg
from src.text_gen.llm_config import models_config
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


def load_llm(model_name: cfg.LLM_MODEL = cfg.llm_model_name, max_new_tokens: int = 512):
    if model_name not in models_config:
        raise ValueError(f"{model_name} is not supported.")

    query_wrapper_prompt = PromptTemplate(models_config[model_name].template)
    
    return HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        max_new_tokens=max_new_tokens,
        context_window=models_config[model_name].context_window,
        generate_kwargs=models_config[model_name].generate_config,
        model_kwargs=models_config[model_name].model_kwargs,
        query_wrapper_prompt=query_wrapper_prompt,
        device_map='auto',
    )
