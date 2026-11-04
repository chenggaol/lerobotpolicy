from lerobot.processor import make_default_pre_post_processors

def make_mine_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)

""" # processor_my_policy.py
from typing import Any
import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor import make_default_pre_post_processors

def make_my_policy_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
 preprocessor = ...   # build your PolicyProcessorPipeline for inputs
    postprocessor = ...  # build your PolicyProcessorPipeline for outputs
    return preprocessor, postprocessor """