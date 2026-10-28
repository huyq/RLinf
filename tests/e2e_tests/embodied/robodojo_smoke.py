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

"""Exercise RoboDojo agent tools, observations, and cleanup without a model."""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.agents.robodojo.tools import RoboDojoAgentTools
from rlinf.envs import get_env_cls
from rlinf.envs.sim.robodojo.actions import EefTarget
from rlinf.utils.logging import get_logger


def main() -> None:
    """Run a short hold-position episode on an installed NVIDIA simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", default=os.environ.get("ROBODOJO_ASSETS_ROOT"))
    parser.add_argument("--task", default="put_bottles_into_dustbin")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--interface", choices=("agent", "joint", "eef"), default="agent"
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        parser.error("--steps and --num-envs must be positive")
    if not args.assets_root or not (Path(args.assets_root) / "Assets").is_dir():
        parser.error(
            "--assets-root must name a directory containing RoboDojo's Assets/"
        )

    root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(
        root / "examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml"
    )
    cfg.assets_path = str(Path(args.assets_root).resolve())
    cfg.task_config.task_name = args.task
    env = get_env_cls(cfg.env_type)(cfg, args.num_envs, 0, 1, None)
    try:
        obs, _ = env.reset(env_seeds=[args.seed] * args.num_envs)
        assert obs["main_images"].shape[0] == args.num_envs
        assert obs["main_images"].dtype == torch.uint8
        assert obs["states"].shape == (args.num_envs, 14)
        assert all(text.strip() for text in obs["task_descriptions"])
        if args.interface == "agent":
            agent = RoboDojoAgentTools(env)
            assert any(
                tool["function"]["name"] == "move_joints" for tool in agent.schemas()
            )
            for index in range(args.num_envs):
                for control_index in range(args.steps):
                    current = agent.observe(index)
                    if current["terminated"] or current["truncated"]:
                        break
                    before = env.elapsed_steps.clone()
                    motion = "move_joints" if control_index % 2 == 0 else "move_eef"
                    targets = (
                        {"left_j0": current["joints"]["left_j0"]}
                        if motion == "move_joints"
                        else {"left": current["eef_poses"]["left"]}
                    )
                    obs, reward, _, _, info = agent.step(
                        {
                            "name": motion,
                            "arguments": {
                                "targets": targets,
                                "note": "Check the control path while holding the current pose.",
                            },
                        },
                        env_idx=index,
                    )
                    assert info["agent"]["ok"]
                    assert info["agent"]["executed_steps"] > 0
                    untouched = torch.arange(args.num_envs) != index
                    torch.testing.assert_close(
                        env.elapsed_steps[untouched], before[untouched]
                    )
                    assert (
                        torch.isfinite(obs["states"]).all()
                        and torch.isfinite(reward).all()
                    )
                current = agent.observe(index)
                if not current["terminated"] and not current["truncated"]:
                    _, reward, terminated, _, info = agent.call(
                        "done", {"summary": "Control smoke finished."}, env_idx=index
                    )
                    assert info["agent"]["ok"] and terminated[index]
                    assert not reward.any()
        elif args.interface == "eef":
            for index in range(args.num_envs):
                for _ in range(args.steps):
                    current = env.observe(index)
                    if current["terminated"] or current["truncated"]:
                        break
                    obs, reward, _, _, info = env.execute(
                        EefTarget(left=tuple(current["eef_poses"]["left"])),
                        env_idx=index,
                    )
                    assert info["execution"]["executed_steps"] > 0
                    assert (
                        torch.isfinite(obs["states"]).all()
                        and torch.isfinite(reward).all()
                    )
        else:
            for _ in range(args.steps):
                obs, reward, terminated, truncated, info = env.step(obs["states"])
                assert torch.isfinite(obs["states"]).all()
                assert torch.isfinite(reward).all()
                assert "executed_steps" in info
                if (terminated | truncated).all():
                    break
        if args.num_envs > 1:
            unaffected = obs["states"][1:].clone()
            obs, _ = env.reset(env_idx=0, env_seeds=[args.seed] * args.num_envs)
            torch.testing.assert_close(obs["states"][1:], unaffected)
        get_logger().info(
            f"RoboDojo {args.interface} smoke passed; task success was not evaluated."
        )
    finally:
        env.close()
        env.close()


if __name__ == "__main__":
    main()
