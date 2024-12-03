import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)

from pathlib import Path
import torch

from typing import Any, List, Optional

EmbeddingVector = List[float]
