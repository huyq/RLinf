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
"""Train Qwen3.8-27B from camera observations and RoboDojo task rewards."""

import hydra
from omegaconf import DictConfig

from rlinf.agents.robodojo.config import validate_robodojo_cfg


@hydra.main(version_base="1.3", config_path="config", config_name="qwen3_8_27b_grpo")
def main(cfg: DictConfig) -> None:
    """Launch the FSDP2 actor, SGLang rollout, and image-conditioned agent workers."""
    validate_robodojo_cfg(cfg)
    from rlinf.agents.robodojo.actor import RoboDojoFSDPActor
    from rlinf.agents.robodojo.agent_loop import RoboDojoAgentLoopWorker
    from rlinf.agents.robodojo.dataset import RoboDojoDataset
    from rlinf.config import validate_cfg
    from rlinf.models.tokenization.hf import hf_tokenizer
    from rlinf.runners.agent_runner import AgentRunner
    from rlinf.scheduler import Cluster, NodePlacementStrategy
    from rlinf.utils.logging import get_logger
    from rlinf.utils.placement import ModelParallelComponentPlacement
    from rlinf.workers.rollout.utils import get_rollout_backend_worker

    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelComponentPlacement(cfg, cluster)
    if not placement.is_collocated:
        raise ValueError(
            "RoboDojo FSDP2 training requires collocated actor and rollout placement."
        )
    actor_strategy = placement.get_strategy("actor")
    actor_world_size = len(actor_strategy.get_placement(cluster))
    if cfg.data.rollout_batch_size % actor_world_size:
        raise ValueError(
            f"Expected data.rollout_batch_size divisible by actor world size {actor_world_size}."
        )
    rollout_strategy = placement.get_strategy("rollout")
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement)
        .launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_strategy
        )
    )
    agent_strategy = NodePlacementStrategy(
        [item.cluster_node_rank for item in rollout_strategy.get_placement(cluster)]
    )
    agent = RoboDojoAgentLoopWorker.create_group(cfg, placement).launch(
        cluster, name=cfg.agentloop.group_name, placement_strategy=agent_strategy
    )
    actor = RoboDojoFSDPActor.create_group(cfg, placement).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_strategy
    )
    tokenizer = hf_tokenizer(
        cfg.actor.tokenizer.tokenizer_model,
        trust_remote_code=cfg.actor.tokenizer.trust_remote_code,
    )
    dataset = RoboDojoDataset(list(cfg.data.layout_seeds), cfg.data.repeat, tokenizer)
    get_logger().info(
        "Starting camera-conditioned RoboDojo GRPO with an external simulator service."
    )
    runner = AgentRunner(
        cfg=cfg,
        placement=placement,
        train_dataset=dataset,
        val_dataset=None,
        rollout=rollout,
        inference=None,
        actor=actor,
        reward=None,
        agent_loop=agent,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
