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
"""Saved-layout success metrics for the camera-conditioned RoboDojo agent."""

import json
from pathlib import Path
from typing import Any

from rlinf.runners.agent_eval_runner import AgentEvalRunner
from rlinf.scheduler import Channel
from rlinf.utils.logging import get_logger


class RoboDojoEvalRunner(AgentEvalRunner):
    """Evaluate a fixed layout set and save task success separately from returns."""

    def pre_process(self) -> dict:
        """Create the saved-layout evaluation result collection."""
        return {"episodes": []}

    def update(
        self,
        context: dict,
        eval_pbar: Any,
        input_channel: Channel,
        batch_idx: int,
        batch: dict,
    ) -> None:
        """Collect one validation batch of per-layout outcomes."""
        received = 0
        while received < self.total_batch_size:
            result = input_channel.get()
            for index in range(result.group_size):
                context["episodes"].append(
                    {
                        "seed": result.extra_fields_group["seed"],
                        "success": result.extra_fields_traj["success"][index],
                        "return": result.extra_fields_traj["task_return"][index],
                        "turns": result.idx_to_traj.count(index),
                    }
                )
            received += result.group_size
            eval_pbar.update(result.group_size)

    def post_process(self, context: dict) -> dict:
        """Write outcomes and log the aggregate task success rate and return."""
        episodes = context["episodes"]
        metrics = {
            "eval/success_rate": sum(episode["success"] for episode in episodes)
            / len(episodes),
            "eval/return": sum(episode["return"] for episode in episodes)
            / len(episodes),
        }
        directory = Path(self.cfg.runner.output_dir) / self.cfg.runner.experiment_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "results.json"
        path.write_text(json.dumps({"metrics": metrics, **context}, indent=2))
        self.metric_logger.log(metrics, 0)
        get_logger().info("Saved RoboDojo evaluation to %s", path)
        return metrics
