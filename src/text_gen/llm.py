from typing import Any, Callable, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
)
from llama_index.llms.vllm import Vllm
from llama_index.core import PromptTemplate
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from vllm import SamplingParams
from vllm.sequence import RequestMetrics
from src.text_gen.llm_config import models_config, vllm_config
from src.db.db import save_logs


class CustomVllm(Vllm):
    
    repetition_penalty: float = Field(
        default=1.0,
        description="Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.",
    )

    custom_chat_system_prompt: float = Field(
        default=None,
        description="Use it to generate with calling vllm.chat. system_prompt must be ''",
    )
    
    custom_chat_user_instruction: float = Field(
        default=None,
        description="Use it to precalculate KV-cache for user_instruction when custom_chat is used.",
    )
    
    def __init__(
        self,
        model: str,
        logs_path: str,
        temperature: float = 1.0,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        ignore_eos: bool = False,
        max_new_tokens: int = 512,
        logprobs: Optional[int] = None,
        dtype: str = "auto",
        download_dir: Optional[str] = None,
        vllm_kwargs: Dict[str, Any] = {},
        api_url: Optional[str] = "",
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        custom_chat_system_prompt: Optional[str] = None,
        custom_chat_user_instruction: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            ignore_eos=ignore_eos,
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            dtype=dtype,
            download_dir=download_dir,
            vllm_kwargs=vllm_kwargs,
            api_url=api_url,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code
        )

        self.repetition_penalty = repetition_penalty

        self.custom_chat_system_prompt = custom_chat_system_prompt
        self.custom_chat_user_instruction = custom_chat_user_instruction
        
        # warmup so that the shared prompt's KV cache is computed.
        if vllm_kwargs.get("enable_prefix_caching"):
            self.precompute_kvcache()

        self._logs_path = logs_path

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "best_of": self.best_of,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop,
            "logprobs": self.logprobs,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return {**base_kwargs}

    def sampling_params(self, **kwargs: Any):
        # build sampling parameters
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}
        return SamplingParams(**params)

    @staticmethod
    def get_messages(system_content: str, user_content: str) -> List[Dict[str, str]]:
        return [{
            "role": "system",
            "content": system_content
        }, {
            "role": "user",
            "content": user_content
        }]

    def precompute_kvcache(self):
        sampling_params = self.sampling_params()
        if self.custom_chat_system_prompt:
            messages = self.get_messages(self.custom_chat_system_prompt,
                                         self.custom_chat_user_instruction)
            self._client.chat(messages, sampling_params, use_tqdm=False)
        else:
            self._client.generate(self.system_prompt, sampling_params, use_tqdm=False)

    def _save_metrics(self, metrics: RequestMetrics, prompt: str):
        save_logs("llm", metrics.finished_time - metrics.arrival_time, self._logs_path)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        sampling_params = self.sampling_params(**kwargs)
        
        if self.custom_chat_system_prompt:
            messages = self.get_messages(self.custom_chat_system_prompt, prompt)
            outputs = self._client.chat(messages, sampling_params, use_tqdm=False)
        else:
            outputs = self._client.generate([prompt], sampling_params, use_tqdm=False)
        self._save_metrics(outputs[0].metrics, outputs[0].prompt)
        return CompletionResponse(text=outputs[0].outputs[0].text)


def load_llm_and_qa_tmpl(model_name: str, max_new_tokens: int, cache_dir: str, logs_path: str):
    if model_name not in models_config:
        raise ValueError(f"{model_name} is not supported.")

    qa_prompt_tmpl = PromptTemplate(models_config[model_name].text_qa_template)
    
    llm = CustomVllm(
        model=model_name,
        custom_chat_system_prompt=models_config[model_name].system_prompt,
        custom_chat_user_instruction=models_config[model_name].instruction,
        system_prompt='',
        max_new_tokens=max_new_tokens,
        vllm_kwargs=vllm_config,
        download_dir=cache_dir,
        trust_remote_code=True,
        tensor_parallel_size=1,
        logs_path=logs_path,
        **models_config[model_name].generate_config
    )
    
    return llm, qa_prompt_tmpl
