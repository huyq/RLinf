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
"""Evaluate a Qwen camera/tool policy on saved RoboDojo layouts."""

import hydra
from omegaconf import DictConfig

from rlinf.agents.robodojo.config import validate_robodojo_cfg


@hydra.main(version_base="1.3", config_path="config", config_name="qwen3_8_27b_eval")
def main(cfg: DictConfig) -> None:
    """Launch rollout and agent workers against the same simulator session service."""
    validate_robodojo_cfg(cfg)
    from rlinf.agents.robodojo.agent_loop import RoboDojoAgentLoopWorker
    from rlinf.agents.robodojo.dataset import RoboDojoDataset
    from rlinf.agents.robodojo.eval_runner import RoboDojoEvalRunner
    from rlinf.config import validate_cfg
    from rlinf.models.tokenization.hf import hf_tokenizer
    from rlinf.scheduler import Cluster, NodePlacementStrategy
    from rlinf.utils.placement import ModelParallelEvalComponentPlacement
    from rlinf.workers.rollout.utils import get_rollout_backend_worker

    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelEvalComponentPlacement(cfg, cluster)
    strategy = placement.get_strategy("rollout")
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement, weight_reload=None)
        .launch(cluster, name=cfg.rollout.group_name, placement_strategy=strategy)
    )
    agent_strategy = NodePlacementStrategy(
        [item.cluster_node_rank for item in strategy.get_placement(cluster)]
    )
    agent = RoboDojoAgentLoopWorker.create_group(cfg, placement).launch(
        cluster, name=cfg.agentloop.group_name, placement_strategy=agent_strategy
    )
    dataset = RoboDojoDataset(
        list(cfg.data.layout_seeds), cfg.data.repeat, hf_tokenizer(cfg.model_path)
    )
    runner = RoboDojoEvalRunner(
        cfg=cfg,
        placement=placement,
        val_dataset=dataset,
        rollout=rollout,
        reward=None,
        agent_loop=agent,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
