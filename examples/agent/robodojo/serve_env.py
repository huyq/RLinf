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
"""Serve RoboDojo cameras and robot tools from a dedicated simulator environment."""

import argparse
from pathlib import Path

from aiohttp import web
from omegaconf import OmegaConf

from rlinf.agents.robodojo.service import RoboDojoSessions, create_app
from rlinf.envs import get_env_cls


def main() -> None:
    """Own simulator startup and teardown on the main thread."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml"
        ),
    )
    parser.add_argument("--assets-root")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--session-timeout", type=float, default=3600)
    parser.add_argument(
        "overrides", nargs="*", help="Environment overrides in key=value form."
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    cfg = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)
    )
    if args.assets_root:
        cfg.assets_path = args.assets_root
    if (
        cfg.auto_reset
        or cfg.ignore_terminations
        or cfg.group_size != 1
        or cfg.task_config.random
    ):
        parser.error(
            "Session serving requires explicit resets, group_size=1, and saved layouts (task_config.random=False)."
        )
    env = get_env_cls(cfg.env_type)(cfg, args.num_envs, 0, 1, None)
    try:
        # Fill every slot before partial resets so native camera/state reads are valid.
        env.reset(env_seeds=[0] * args.num_envs)
        app = create_app(
            RoboDojoSessions(env, args.max_turns, args.image_size, args.session_timeout)
        )
        web.run_app(
            app, host=args.host, port=args.port, print=None, handler_cancellation=True
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
