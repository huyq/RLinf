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

"""CPU simulator double for the optional native runtime and its robot manager."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


class NativeEnv:
    """Deterministic joints, translated/rotated bases, IK failure, and latched success."""

    def __init__(self, task_config):
        self.action_dims = tuple(task_config["action_dims"])
        self.task_config = task_config
        self.closed = False
        root = np.sqrt(0.5)
        self.robot_list = [
            SimpleNamespace(
                type="target",
                arm_name="left_arm",
                entity_origin_pose=[10, 0, 0, root, 0, 0, root],
            ),
            SimpleNamespace(
                type="target",
                arm_name="right_arm",
                entity_origin_pose=[0, 10, 0, 1, 0, 0, 0],
            ),
        ]
        self.env = SimpleNamespace(robot_manager=self)
        self.reset(0)

    def reset(self, seed=None):
        self.seed = seed
        self.count = 0
        self.state = np.zeros(sum(self.action_dims), dtype=float)
        self.state[-2:] = 0.5
        self.done = False
        self.last_actions = None

    def get_obs(self):
        if self.task_config.get("broken_sensor"):
            self.state[0] = np.nan
        return {
            "state": self.state.copy(),
            "instruction": f"task seed {self.seed}",
            "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def get_real_endpose(self, robot, env_idx_list=None, is_relative=False):
        side = robot.arm_name.split("_")[0]
        offset = 0 if side == "left" else 6
        base = np.array(robot.entity_origin_pose)
        rotation = Rotation.from_quat(base[[4, 5, 6, 3]])
        local = Rotation.from_euler("ZYX", self.state[offset + 3 : offset + 6])
        xyz = rotation.apply(self.state[offset : offset + 3]) + base[:3]
        quat = (rotation * local).as_quat()[[3, 0, 1, 2]]
        return {0: np.r_[xyz, quat]}

    def solve_ik(self, target_pose, env_idx, robot):
        world = np.asarray(target_pose)
        base = np.array(robot.entity_origin_pose)
        rotation = Rotation.from_quat(base[[4, 5, 6, 3]])
        xyz = rotation.inv().apply(world[:3] - base[:3])
        if np.max(np.abs(xyz)) > 1:
            return {"status": "Failed"}
        orientation = rotation.inv() * Rotation.from_quat(world[[4, 5, 6, 3]])
        return {
            "status": "Success",
            "joint_value": np.r_[xyz, orientation.as_euler("ZYX")],
        }

    def step(self, actions):
        if self.task_config.get("step_fault"):
            raise ValueError("simulator fault")
        executed = 0
        score = 0.0
        self.last_actions = np.array(actions)
        for action in actions:
            if self.done or self.count >= self.task_config["max_episode_steps"]:
                break
            self.state = np.array(action, dtype=float)
            self.count += 1
            executed += 1
            if self.count >= self.task_config.get("finish_at", 1000):
                self.done = True
                score = 1.0
        return (
            self.get_obs(),
            score,
            self.done,
            self.count >= self.task_config["max_episode_steps"],
            {"success": self.done, "executed_steps": executed},
        )

    def check_seeds(self, seeds):
        return [seed >= 0 for seed in seeds]

    def close(self, clear_cache=True):
        self.closed = True
