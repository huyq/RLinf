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

"""Named robot tools following inspect-robots' partial-target motion interface."""

import json
import math
from numbers import Real
from typing import Any

import numpy as np
import torch

from rlinf.envs.sim.robodojo.actions import (
    EefTarget,
    InvalidAction,
    JointControls,
    JointTarget,
)
from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidAction(f"Expected a finite number for {name}, got {value!r}.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise InvalidAction(f"Expected a finite number for {name}.") from exc
    if not math.isfinite(result):
        raise InvalidAction(f"Expected a finite number for {name}, got {value!r}.")
    return result


def _euler_pose(pose: np.ndarray) -> list[float]:
    x, y, z, w, qx, qy, qz = np.asarray(pose, dtype=float)
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    pitch = math.asin(float(np.clip(2 * (w * qy - qz * qx), -1, 1)))
    roll = math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    return [float(v) for v in (x, y, z, yaw, pitch, roll)]


class _ToolParser:
    """Validate tool arguments and map named targets to physical actions.

    Args:
        action_dims: Native left arm, right arm, left gripper, right gripper widths.
        config: Motion/turn budgets and optional named joint limits.
            Arm angles are radians; grippers are fractions.
    """

    def __init__(self, action_dims: tuple[int, ...], config: dict) -> None:
        self.labels = tuple(
            [f"left_j{i}" for i in range(action_dims[0])]
            + [f"right_j{i}" for i in range(action_dims[1])]
            + ["left_gripper"]
            + ["right_gripper"]
        )
        if action_dims[2:] != (1, 1):
            raise InvalidAction("Agent tools require one scalar gripper per arm.")
        self.max_motion_steps = config.get("max_motion_steps", 100)
        self.max_turns = config.get("max_turns", 64)
        for name in ("max_motion_steps", "max_turns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InvalidAction(f"Expected positive integer {name}, got {value!r}.")
        self.bounds = {"left_gripper": (0.0, 1.0), "right_gripper": (0.0, 1.0)}
        for name, limits in config.get("joint_limits", {}).items():
            if name not in self.labels[:-2] or len(limits) != 2:
                raise InvalidAction(
                    f"Expected an arm joint and [lower, upper] limits, got {name!r}: {limits!r}."
                )
            low, high = (_number(value, name) for value in limits)
            if low > high:
                raise InvalidAction(
                    f"Expected ordered limits for {name}, got {limits!r}."
                )
            self.bounds[name] = (low, high)

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-tool schemas with native dimension names."""
        note = {
            "type": "string",
            "minLength": 1,
            "description": "What you observe and why you chose this action.",
        }
        dimensions = {}
        for name in self.labels:
            spec = {"type": "number"}
            if name in self.bounds:
                spec.update(minimum=self.bounds[name][0], maximum=self.bounds[name][1])
            dimensions[name] = spec

        def tool(name, description, properties, required):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }

        return [
            tool(
                "move_joints",
                (
                    "Move to named absolute joint targets (radians) or gripper opening fractions [0,1]. "
                    "Unnamed dimensions hold their observed values. The controller interpolates targets "
                    f"over at most {self.max_motion_steps} controls; large moves must be split. "
                    "Read actual execution and target residuals in the result."
                ),
                {
                    "targets": {
                        "type": "object",
                        "properties": dimensions,
                        "additionalProperties": False,
                        "minProperties": 1,
                    },
                    "note": note,
                },
                ["targets", "note"],
            ),
            tool(
                "move_eef",
                "Move absolute EE poses in each arm's own base frame. Each pose is [x,y,z,yaw,pitch,roll] in metres and radians, using Rz(yaw) Ry(pitch) Rx(roll). Unspecified arms hold measured joints. Grippers are opening fractions [0,1]. IK failure executes no motion.",
                {
                    "targets": {
                        "type": "object",
                        "additionalProperties": False,
                        "minProperties": 1,
                        "properties": {
                            **{
                                side: {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 6,
                                    "maxItems": 6,
                                }
                                for side in ("left", "right")
                            },
                            **{
                                side + "_gripper": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                }
                                for side in ("left", "right")
                            },
                        },
                    },
                    "note": note,
                },
                ["targets", "note"],
            ),
            tool(
                "wait",
                "Hold current joint targets for a bounded number of controls.",
                {
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.max_motion_steps,
                    },
                    "note": note,
                },
                ["steps", "note"],
            ),
            tool(
                "take_pic",
                "Refresh camera observations and named robot state without advancing physics.",
                {"note": note},
                ["note"],
            ),
            tool(
                "done",
                "End this episode. Only the simulator's task predicate determines success.",
                {
                    "summary": {"type": "string", "minLength": 1},
                    "hindsight": {"type": "string"},
                },
                ["summary"],
            ),
            tool(
                "give_up",
                "End this episode and record why the agent stopped.",
                {
                    "reason": {"type": "string", "minLength": 1},
                    "hindsight": {"type": "string"},
                },
                ["reason"],
            ),
        ]

    def plan(
        self, name: str, arguments: dict | str, current: np.ndarray
    ) -> tuple[
        np.ndarray | JointTarget | EefTarget | None, dict[str, float] | None, dict
    ]:
        """Validate a tool call and return an action, named targets, and arguments.

        Invalid model arguments raise ValueError. Invalid simulator state raises
        RuntimeError so sensor failures remain distinct from correctable calls.
        """
        current = np.asarray(current, dtype=np.float64)
        if current.shape != (len(self.labels),) or not np.isfinite(current).all():
            raise RuntimeError("Expected finite native joint state for agent control.")
        specifications = {
            tool["function"]["name"]: tool["function"]["parameters"]
            for tool in self.schemas()
        }
        if not isinstance(name, str) or name not in specifications:
            raise InvalidAction(
                f"Unknown tool {name!r}; expected one of {list(specifications)}."
            )
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError) as exc:
                raise InvalidAction(
                    "Expected tool arguments to be a JSON object."
                ) from exc
        if not isinstance(arguments, dict):
            raise InvalidAction("Expected tool arguments to be a JSON object.")
        spec = specifications[name]
        if set(arguments) - set(spec["properties"]):
            raise InvalidAction(
                f"Unexpected arguments for {name}: {sorted(set(arguments) - set(spec['properties']))}."
            )
        for required in spec["required"]:
            value = arguments.get(required)
            if required in ("note", "summary", "reason") and (
                not isinstance(value, str) or not value.strip()
            ):
                raise InvalidAction(f"Expected a non-empty {required} for {name}.")
        if "hindsight" in arguments and not isinstance(arguments["hindsight"], str):
            raise InvalidAction("Expected hindsight to be a string.")
        if name in ("take_pic", "done", "give_up"):
            return None, None, arguments
        if name == "wait":
            steps = arguments.get("steps")
            if (
                isinstance(steps, bool)
                or not isinstance(steps, int)
                or not 1 <= steps <= self.max_motion_steps
            ):
                raise InvalidAction(
                    f"Expected wait steps in [1, {self.max_motion_steps}]."
                )
            return (
                np.repeat(
                    np.r_[current[:-2], np.clip(current[-2:], 0, 1)][None],
                    steps,
                    axis=0,
                ),
                None,
                arguments,
            )

        if name == "move_eef":
            values = arguments.get("targets")
            allowed = {"left", "right", "left_gripper", "right_gripper"}
            if not isinstance(values, dict) or not values or set(values) - allowed:
                raise InvalidAction(
                    "Expected EEF targets containing left/right poses or grippers."
                )
            poses, parsed = {}, {}
            for key, value in values.items():
                if key in ("left", "right"):
                    if not isinstance(value, (list, tuple)) or len(value) != 6:
                        raise InvalidAction(
                            "Expected [x,y,z,yaw,pitch,roll] for each requested arm."
                        )
                    xyz = [_number(v, key) for v in value]
                    yaw, pitch, roll = np.array(xyz[3:]) / 2
                    cy, cp, cr = np.cos([yaw, pitch, roll])
                    sy, sp, sr = np.sin([yaw, pitch, roll])
                    poses[key] = tuple(xyz[:3]) + (
                        cr * cp * cy + sr * sp * sy,
                        sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                    )
                    parsed[key] = xyz
                else:
                    grip = _number(value, key)
                    if not 0 <= grip <= 1:
                        raise InvalidAction("Expected gripper opening in [0, 1].")
                    poses[key] = parsed[key] = grip
            return EefTarget(**poses), None, {**arguments, "targets": parsed}

        values = arguments.get("targets")
        if not isinstance(values, dict) or not values:
            raise InvalidAction(
                "Expected a non-empty targets object of joint names and values."
            )
        target = current.copy()
        # Measured openings may overshoot; commanded openings remain fractions.
        target[-2:] = np.clip(target[-2:], 0, 1)
        targets = {}
        for label, value in values.items():
            if label not in self.labels:
                raise InvalidAction(
                    f"Unknown joint {label!r}; expected one of {list(self.labels)}."
                )
            value = _number(value, label)
            if label in self.bounds:
                low, high = self.bounds[label]
                if not low <= value <= high:
                    raise InvalidAction(
                        f"Expected {label} in [{low}, {high}], got {value}."
                    )
            target[self.labels.index(label)] = value
            targets[label] = value
        return JointTarget(target), targets, {**arguments, "targets": targets}


class RoboDojoAgentTools:
    """Bind LLM tools to a physical environment, owning agent episode state.

    Args:
        env: A resettable RoboDojo environment owned by the caller.
        config: Agent settings: max_turns and optional named joint_limits.
            Physical motion limits are read from env.cfg.controller.

    Resetting an environment slot also starts a new agent episode on that slot.
    This adapter never closes the environment or invokes its private methods.
    """

    def __init__(self, env: RoboDojoEnv, config: dict | None = None) -> None:
        self.env = env
        from omegaconf import OmegaConf

        physical = OmegaConf.to_container(
            env.cfg.get("controller", OmegaConf.create({}))
        )
        agent = dict(config or {})
        unexpected = set(agent) - {"max_turns", "joint_limits"}
        if unexpected:
            raise InvalidAction(f"Unknown agent settings: {sorted(unexpected)}.")
        self.parser = _ToolParser(env.action_dims, {**physical, **agent})
        self.turns = torch.zeros(env.num_envs, dtype=torch.long)
        self.terminated = torch.zeros(env.num_envs, dtype=torch.bool)
        self.truncated = torch.zeros(env.num_envs, dtype=torch.bool)
        self.episode_ids = env.episode_ids.clone()

    def schemas(self) -> list[dict[str, Any]]:
        """Return the function schemas offered to the LLM."""
        return self.parser.schemas()

    def observe(self, env_idx: int = 0) -> dict[str, Any]:
        """Format native state and agent budgets as JSON-compatible context."""
        raw = self.env.observe(env_idx)
        changed = self.episode_ids != self.env.episode_ids
        self.turns[changed] = 0
        self.terminated[changed] = False
        self.truncated[changed] = False
        self.episode_ids = self.env.episode_ids.clone()
        state = np.asarray(raw["state"], dtype=float)
        if state.shape != (len(self.parser.labels),) or not np.isfinite(state).all():
            raise RuntimeError("Expected finite native joint state for agent control.")
        return {
            "instruction": raw["instruction"],
            "joints": dict(zip(self.parser.labels, state.tolist())),
            "eef_poses": {
                side: _euler_pose(pose)
                for side, pose in raw.get("eef_poses", {}).items()
            },
            "cameras": [
                name
                for key, name in (
                    ("full_image", "head"),
                    ("left_wrist_image", "left_wrist"),
                    ("right_wrist_image", "right_wrist"),
                )
                if raw.get(key) is not None
            ],
            "elapsed_steps": raw["elapsed_steps"],
            "remaining_steps": raw["remaining_steps"],
            "turns": int(self.turns[env_idx]),
            "remaining_turns": max(0, self.parser.max_turns - int(self.turns[env_idx])),
            "success": raw["success"],
            "terminated": raw["terminated"] or bool(self.terminated[env_idx]),
            "truncated": raw["truncated"] or bool(self.truncated[env_idx]),
        }

    def step(self, call: dict, *, env_idx: int = 0) -> tuple:
        """Execute {name, arguments} or an OpenAI function envelope."""
        envelope = call.get("function", call) if isinstance(call, dict) else {}
        if not isinstance(envelope, dict):
            envelope = {}
        return self.call(
            envelope.get("name"), envelope.get("arguments", {}), env_idx=env_idx
        )

    def call(self, name: str, arguments: dict | str, *, env_idx: int = 0) -> tuple:
        """Execute one tool call; argument/IK errors consume a turn and return feedback.

        Native runtime faults propagate. done/give_up end the agent episode;
        simulator success alone supplies rewards. All transitions stay batched.
        """
        current = self.observe(env_idx)
        feedback = {
            "name": name if isinstance(name, str) else "",
            "env_idx": env_idx,
            "ok": False,
            "requested_steps": 0,
            "executed_steps": 0,
        }
        result = None
        targets = None
        if current["terminated"] or current["truncated"]:
            feedback["error"] = (
                "Episode ended; reset this slot before another tool call."
            )
        else:
            self.turns[env_idx] += 1
            try:
                action, targets, parsed = self.parser.plan(
                    name, arguments, list(current["joints"].values())
                )
                if name in ("done", "give_up"):
                    self.terminated[env_idx] = True
                    feedback["stop_reason"] = name
                elif name == "take_pic":
                    self.env.observe(env_idx, refresh=True)
                elif action is not None:
                    if isinstance(action, np.ndarray):
                        action = JointControls(action)
                    result = self.env.execute(action, env_idx=env_idx)
                    feedback.update(
                        {
                            key: result[-1]["execution"][key]
                            for key in ("requested_steps", "executed_steps")
                        }
                    )
                feedback.update(ok=True, arguments=parsed)
            except InvalidAction as error:
                feedback["error"] = str(error)
            if (
                self.turns[env_idx] >= self.parser.max_turns
                and not self.observe(env_idx)["terminated"]
            ):
                self.truncated[env_idx] = True
        if result is None:
            result = (
                self.env.snapshot(),
                torch.zeros(self.env.num_envs),
                torch.tensor(
                    [
                        self.env.observe(i)["terminated"]
                        for i in range(self.env.num_envs)
                    ]
                ),
                torch.tensor(
                    [self.env.observe(i)["truncated"] for i in range(self.env.num_envs)]
                ),
                {
                    "executed_steps": [0] * self.env.num_envs,
                    "valid_action_mask": torch.zeros(
                        (self.env.num_envs, 1), dtype=torch.bool
                    ),
                    "success": [
                        self.env.observe(i)["success"] for i in range(self.env.num_envs)
                    ],
                },
            )
        obs, reward, term, trunc, infos = result
        term, trunc = term | self.terminated, trunc | self.truncated
        feedback["observation"] = self.observe(env_idx)
        if targets is not None:
            measured = feedback["observation"]["joints"]
            feedback["targets"] = targets
            feedback["residuals"] = {
                label: measured[label] - target for label, target in targets.items()
            }
        infos["agent"] = feedback
        return obs, reward, term, trunc, infos
