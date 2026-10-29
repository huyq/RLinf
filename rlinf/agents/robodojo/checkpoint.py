# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export full FSDP weights with the original Qwen processor and configuration."""

from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)


def export_checkpoint(weights: str, model_path: str, output_dir: str) -> None:
    """Write an evaluable HF checkpoint without allocating a second model copy.

    Args:
        weights: RLinf actor/model_state_dict/full_weights.pt.
        model_path: Original checkpoint containing model config and processor.
        output_dir: New HF checkpoint directory for evaluation or serving.
    """
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Expected an empty export directory, got {destination}.")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise ValueError(f"Expected qwen3_5 model config, got {config.model_type!r}.")
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    if (Path(model_path) / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(
            model_path, local_files_only=True
        )
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.save_pretrained(destination)
