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
"""Vision-preserving FSDP2 updates for RoboDojo tool trajectories."""

import copy

import torch
import torch.nn.functional as F

from rlinf.utils.data_iter_utils import get_seqlen_balanced_partitions
from rlinf.workers.actor.ma_fsdp_actor_worker import MAFSDPActor


def vision_logprobs(
    model: torch.nn.Module,
    batch: dict,
    temperature: float,
    calculate_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Score exact prompt+response IDs using their original images and model mRoPE.

    Micro-batches contain one turn. Each forward computes positional information
    afresh, because frames and history can differ between consecutive turns.
    """
    if batch["input_ids"].shape[0] != 1 or len(batch["multi_modal_inputs"]) != 1:
        raise ValueError("RoboDojo vision scoring requires one turn per micro-batch.")
    device = batch["input_ids"].device
    image_inputs = {
        key: value.to(device) for key, value in batch["multi_modal_inputs"][0].items()
    }
    length = int(batch["attention_mask"][0].sum())
    prompt_length = int(batch["prompt_lengths"][0]) if "prompt_lengths" in batch else 1
    response_length = length - prompt_length
    ids = batch["input_ids"][:, :length]
    if "mm_token_type_ids" in image_inputs:
        types = image_inputs["mm_token_type_ids"]
        image_inputs["mm_token_type_ids"] = F.pad(types, (0, length - types.shape[-1]))
    outputs = model(
        input_ids=ids,
        attention_mask=batch["attention_mask"][:, :length],
        position_ids=None,
        use_cache=False,
        logits_to_keep=response_length + 1,
        **image_inputs,
    )
    logits = outputs.logits[:, :-1].float() / temperature
    distribution = F.log_softmax(logits, dim=-1)
    values = distribution.gather(-1, ids[:, prompt_length:, None]).squeeze(-1)
    padding = (prompt_length, batch["input_ids"].shape[1] - length)
    logprobs = F.pad(values, padding)
    if calculate_entropy:
        entropy = F.pad(-(distribution.exp() * distribution).sum(-1), padding)
        return logprobs, entropy
    return logprobs


def partition_vision_batches(batches: list[dict], world_size: int) -> list[dict]:
    """Balance whole image-bearing turns, padding with zero-loss real examples.

    A padded rank still traverses the vision encoder, so every FSDP rank enters
    the same sharded modules. Tensors and variable-size image dictionaries keep
    the same ordering through gather, padding, and repartitioning.
    """
    keys = batches[0].keys()
    if any(batch.keys() != keys for batch in batches):
        raise ValueError("Expected identical vision batch keys across actor ranks.")
    merged = {}
    for key in keys:
        if isinstance(batches[0][key], torch.Tensor):
            merged[key] = torch.cat([batch[key] for batch in batches])
        elif isinstance(batches[0][key], list):
            merged[key] = [value for batch in batches for value in batch[key]]
        else:
            raise TypeError(
                f"Unsupported vision batch field {key}: {type(batches[0][key]).__name__}."
            )
    count = len(merged["input_ids"])
    if not count:
        raise ValueError("Expected at least one camera-conditioned turn.")
    padding = (-count) % world_size
    for key, values in merged.items():
        if isinstance(values, list):
            values.extend(copy.deepcopy(values[:1]) * padding)
        else:
            fill = values[:1].clone()
            if key in ("response_mask", "loss_scales", "advantages", "rewards"):
                fill.zero_()
            merged[key] = torch.cat([values, *[fill] * padding])
    lengths = (merged["prompt_lengths"] + merged["response_lengths"]).tolist()
    partitions = get_seqlen_balanced_partitions(lengths, world_size, equal_size=True)
    return [
        {
            key: values[indices]
            if isinstance(values, torch.Tensor)
            else [values[i] for i in indices]
            for key, values in merged.items()
        }
        for indices in partitions
    ]


class RoboDojoFSDPActor(MAFSDPActor):
    """Train image-conditioned tool turns with the existing multi-turn GRPO losses."""

    def forward_batch(
        self, m_batch: dict, calculate_entropy: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score one camera-conditioned turn, optionally returning token entropy."""
        with self.amp_context:
            if calculate_entropy and not self.cfg.algorithm.calculate_entropy:
                scores = vision_logprobs(
                    self.model, m_batch, self.cfg.algorithm.sampling_params.temperature
                )
                return scores, torch.zeros_like(scores)
            return vision_logprobs(
                self.model,
                m_batch,
                self.cfg.algorithm.sampling_params.temperature,
                calculate_entropy,
            )

    def _dp_load_balance_dynamic(self, batch, batch_pad, split_fix_chunk):
        if split_fix_chunk != 1:
            raise ValueError(
                "RoboDojo vision training requires actor.micro_batch_size=1."
            )
        local = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
        result = partition_vision_batches(gathered, len(gathered))[
            torch.distributed.get_rank()
        ]
        from rlinf.scheduler import Worker

        return {
            key: value.to(Worker.torch_device_type)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in result.items()
        }
