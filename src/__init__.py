"""
AdaptEval — LoRA/QLoRA Fine-Tuning and Evaluation Framework
src package initializer. Exposes core modules for external access.
"""

from src.load_base_model import load_base_model
from src.tokenize_dataset import tokenize_dataset
from src.inject_lora import inject_lora_adapters
from src.train import run_training
from src.evaluate import run_evaluation
from src.merge_adapter import merge_and_save

__all__ = [
    "load_base_model",
    "tokenize_dataset",
    "inject_lora_adapters",
    "run_training",
    "run_evaluation",
    "merge_and_save",
]