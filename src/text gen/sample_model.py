from langchain_core.language_models.llms import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from util.key_imports import *


class LanguageModel(LLM):
    device: str
    model: Any
    version: Optional[str] = None
    generate_config: dict
    tokenizer: Any

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.version = Path(kwargs["model_path"]).name
        self.device = kwargs["device"]
        self.generate_config = kwargs["generate_config"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            kwargs["model_path"], legacy=True, trust_remote_code=True)
        self.generate_config.update({
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        })
        self.model = AutoModelForCausalLM.from_pretrained(
            kwargs["model_path"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

    @property
    def _llm_type(self) -> str:
        return self.version
    
    def _answer(self, prompt: str) -> str:
        pass

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        pass
