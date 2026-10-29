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
"""RLinf agent worker for camera-conditioned RoboDojo trajectories."""

import asyncio

from omegaconf import DictConfig
from transformers import AutoProcessor

from rlinf.agents.robodojo.rollout import pack_episodes, run_episode
from rlinf.agents.robodojo.service import RoboDojoClient
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import MultiAgentLoopOutput, MultiAgentLoopWorker


class RoboDojoAgentLoopWorker(MultiAgentLoopWorker):
    """Generate robot tool calls with a VLM and emit image-bearing GRPO groups."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        super().__init__(cfg, placement)
        self.processor = AutoProcessor.from_pretrained(
            cfg.rollout.model.model_path,
            trust_remote_code=cfg.rollout.model.trust_remote_code,
        )
        pixels = cfg.agentloop.image_size**2
        self.processor.image_processor.size = {
            "shortest_edge": pixels,
            "longest_edge": pixels,
        }

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict | None = None,
        rollout_name: str | None = None,
        image_data: list | None = None,
    ) -> dict:
        """Bound inference waits so engine failures release simulator sessions."""
        return await asyncio.wait_for(
            super().generate(
                prompt_ids,
                sampling_params=sampling_params,
                rollout_name=rollout_name,
                image_data=image_data,
            ),
            timeout=self.cfg.robodojo.request_timeout,
        )

    async def run_one_query(
        self, input_dict: list[int], answer: dict
    ) -> MultiAgentLoopOutput:
        """Use dataset layout metadata to reset every member of a GRPO group identically."""
        async with RoboDojoClient(
            self.cfg.robodojo.server_url, self.cfg.robodojo.request_timeout
        ) as client:
            return await run_episode(
                client,
                self.processor,
                self.generate,
                seed=answer["seed"],
                max_turns=self.cfg.agentloop.max_turns,
                max_prompt_length=self.cfg.data.max_prompt_length,
                seq_length=self.cfg.runner.seq_length,
                max_new_tokens=self.cfg.algorithm.sampling_params.max_new_tokens,
                return_logprobs=self.return_logprobs,
            )

    def gen_extra_fields(
        self, task_results: list[MultiAgentLoopOutput], answer: dict
    ) -> tuple:
        """Return trajectory outcomes and the shared saved-layout seed."""
        return (
            None,
            {
                key: [result.extra_fields[key] for result in task_results]
                for key in ("task_return", "success")
            },
            {"seed": answer["seed"]},
            {},
        )

    def get_rollout_result(
        self, task_results: list[MultiAgentLoopOutput], *extra_fields
    ) -> DynamicRolloutResult:
        """Preserve per-turn image inputs while packing a trajectory group."""
        result = pack_episodes(task_results)
        (
            result.extra_fields_turn,
            result.extra_fields_traj,
            result.extra_fields_group,
            result.extra_fields_train,
        ) = extra_fields
        return result

    def get_rollout_metrics(self, rollout_result: DynamicRolloutResult) -> dict:
        """Return weighted success, return, and turn metrics for distributed reduction."""
        successes = rollout_result.extra_fields_traj["success"]
        returns = rollout_result.extra_fields_traj["task_return"]
        return {
            "__mean__/robodojo/success_rate": (sum(successes), len(successes)),
            "__mean__/robodojo/return": (sum(returns), len(returns)),
            "__mean__/robodojo/turns": (
                rollout_result.num_sequence,
                rollout_result.group_size,
            ),
        }
