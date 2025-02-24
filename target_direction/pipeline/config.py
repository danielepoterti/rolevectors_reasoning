
import os

from dataclasses import dataclass, field
from typing import Tuple, List
from dotenv import load_dotenv

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train: int = 128
    n_test: int = 100
    max_new_tokens: int = 512
    model_test: str = "anthropic/claude-3.5-haiku"
    providers_test: List[str] = field(default_factory=lambda: ['Anthropic'])
    temperature_test: float = 0.0
    role = "chemist"
    test = "natural_science"
    coeff = +1.0
    batch = 32
    offline = True

    def openrouter_key(self) -> str:
        load_dotenv()
        key = os.getenv("OPENROUTER_KEY")
        if not key and not self.offline:
            raise KeyError("OPENROUTER_KEY not found in environment")
        return key or ""
        
    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)