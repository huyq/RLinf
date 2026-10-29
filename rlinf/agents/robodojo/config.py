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
"""Validate the supported camera-conditioned RoboDojo training configuration."""

from omegaconf import DictConfig, OmegaConf


def validate_robodojo_cfg(cfg: DictConfig) -> None:
    """Reject settings that would drop images or misalign multi-turn GRPO updates."""
    expected = {
        "actor.training_backend": "fsdp",
        "actor.fsdp_config.strategy": "fsdp2",
        "actor.pack_traj": False,
        "runner.enable_dynamic_batch_size": False,
        "actor.enable_dp_load_balance": True,
        "agentloop.is_dynamic_rollout_batch": True,
        "rollout.rollout_backend": "sglang",
        "rollout.detokenize": True,
        "rollout.return_logprobs": True,
        "algorithm.recompute_logprobs": False,
        "algorithm.kl_beta": 0.0,
        "algorithm.n_minibatches": 1,
        "algorithm.training_batch_size_per_gpu": 1,
        "algorithm.sampling_params.temperature": 1.0,
        "algorithm.sampling_params.top_p": 1.0,
        "algorithm.sampling_params.top_k": -1,
        "rollout.model.model_type": "qwen3_5",
        "actor.model.model_type": "qwen3_5",
        "algorithm.calculate_entropy": False,
        "algorithm.entropy_bonus": 0.0,
        "algorithm.adv_type": "grpo_dynamic",
        "algorithm.advantage_mode": "trajectory",
        "algorithm.normalize_advantages": False,
    }
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key)
        if actual != value:
            raise ValueError(
                f"RoboDojo camera training requires {key}={value!r}, got {actual!r}."
            )
    minimum_group = 1 if cfg.runner.task_type == "reasoning_eval" else 2
    if cfg.algorithm.group_size < minimum_group:
        raise ValueError(
            f"Expected algorithm.group_size >= {minimum_group} for {cfg.runner.task_type}."
        )
    if not cfg.data.layout_seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in cfg.data.layout_seeds
    ):
        raise ValueError(
            "Expected nonnegative data.layout_seeds for saved RoboDojo layouts."
        )
    for key in (
        "agentloop.max_turns",
        "algorithm.group_size",
        "runner.seq_length",
        "data.max_prompt_length",
        "data.repeat",
        "data.rollout_batch_size",
        "algorithm.sampling_params.max_new_tokens",
    ):
        value = OmegaConf.select(cfg, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Expected positive integer {key}, got {value!r}.")
    if cfg.agentloop.image_size < 32 or cfg.agentloop.image_size % 32:
        raise ValueError(
            "Expected agentloop.image_size to be a positive multiple of 32."
        )
    if cfg.runner.seq_length <= cfg.data.max_prompt_length:
        raise ValueError("Expected runner.seq_length > data.max_prompt_length.")

    model_path = cfg.rollout.model.model_path
    if (
        cfg.actor.model.model_path != model_path
        or cfg.actor.tokenizer.tokenizer_model != model_path
    ):
        raise ValueError(
            "Expected actor, rollout, and tokenizer to use the same model_path."
        )
    if cfg.data.repeat * len(cfg.data.layout_seeds) < cfg.data.rollout_batch_size:
        raise ValueError(
            "Expected at least one complete rollout batch in the layout dataset."
        )
