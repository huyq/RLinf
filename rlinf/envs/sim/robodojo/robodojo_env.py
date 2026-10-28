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

"""Adapt the optional RoboDojo runtime to RLinf's vector environment contract."""

import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from rlinf.envs.sim.robodojo.actions import EefTarget, JointControls, JointTarget
from rlinf.envs.sim.robodojo.runtime import VectorRuntime
from rlinf.envs.utils import (
    center_crop_image,
    list_of_dict_to_dict_of_list,
    partition_success_seeds,
    valid_action_mask_from_counts,
)
from rlinf.utils.logging import get_logger

__all__ = ["RoboDojoEnv"]

_NATIVE_ACTION_LABELS = ("left_arm", "right_arm", "left_gripper", "right_gripper")
_ALOHA_ACTION_LABELS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
_JOINT_KEYS = (
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_joint_state",
    "right_ee_joint_state",
)

# Settling the same saved layout can fail nondeterministically.
UNSTABLE_RESET_ATTEMPTS = 8


def _is_unstable_scene(exc: BaseException) -> bool:
    """Whether a bridge error is the runtime's layout settle failure.

    A single in-process slot raises ``UnStableError`` itself; a remote slot
    re-raises it as a ``RuntimeError`` that carries the traceback text.
    """
    return type(exc).__name__ == "UnStableError" or "UnStableError" in str(exc)


def state_action_permutations(
    action_order: str, action_dims: list[int] | None
) -> tuple[list[int] | None, list[int] | None]:
    """Return the index lists that convert between the runtime and model layouts.

    The runtime emits ``[left arm, right arm, left gripper, right gripper]``.
    OpenPI Aloha checkpoints (including RoboDojo's official Pi_05) are trained
    on ``[left arm, left gripper, right arm, right gripper]``. ``native`` keeps
    the runtime layout; ``aloha`` returns ``(env_to_model, model_to_env)``.
    """
    if action_order == "native":
        return None, None
    if action_order != "aloha":
        raise ValueError(
            f"state_action_order must be 'native' or 'aloha', got {action_order!r}."
        )
    if action_dims is None or len(action_dims) != 4:
        raise ValueError(
            "state_action_order='aloha' requires task_config.action_dims with "
            f"four widths, got {action_dims!r}."
        )
    widths = {
        label: int(width) for label, width in zip(_NATIVE_ACTION_LABELS, action_dims)
    }
    if any(width <= 0 for width in widths.values()):
        raise ValueError(f"action_dims must be positive, got {action_dims!r}.")

    starts = {}
    offset = 0
    for label in _NATIVE_ACTION_LABELS:
        starts[label] = offset
        offset += widths[label]

    native_to_model = [
        index
        for label in _ALOHA_ACTION_LABELS
        for index in range(starts[label], starts[label] + widths[label])
    ]
    model_to_native = [0] * len(native_to_model)
    for model_index, native_index in enumerate(native_to_model):
        model_to_native[native_index] = model_index
    return native_to_model, model_to_native


class RoboDojoEnv(gym.Env):
    """Own isolated RoboDojo scenes and return batched CPU observations.

    Args:
        cfg: Environment settings and runtime task_config.
        num_envs: Number of local vector slots, divisible by group_size.
        seed_offset: Worker seed offset.
        total_num_processes: Number of environment processes for seed partitioning.
        worker_info: Scheduler metadata retained for the worker.
        record_metrics: Whether to include episode summaries in infos.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = cfg.seed + seed_offset
        self.base_seed = cfg.seed
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.group_size = cfg.group_size
        if self.group_size <= 0 or num_envs <= 0 or num_envs % self.group_size:
            raise ValueError("num_envs must be positive and divisible by group_size.")
        if total_num_processes <= 0 or not 0 <= seed_offset < total_num_processes:
            raise ValueError(
                "Expected 0 <= seed_offset < positive total_num_processes, "
                f"got {seed_offset=} and {total_num_processes=}."
            )
        if cfg.get("enable_offload", False):
            raise ValueError(
                "RoboDojo requires enable_offload=False; close is terminal."
            )
        self.action_dims = tuple(cfg.task_config.get("action_dims", [6, 6, 1, 1]))
        if len(self.action_dims) != 4 or any(
            not isinstance(width, int) or width <= 0 for width in self.action_dims
        ):
            raise ValueError(
                f"Expected four positive action widths, got {self.action_dims}."
            )
        if cfg.max_episode_steps <= 0:
            raise ValueError(
                f"Expected positive max_episode_steps, got {cfg.max_episode_steps}."
            )
        runtime_limit = cfg.task_config.get("max_episode_steps", cfg.max_episode_steps)
        if runtime_limit != cfg.max_episode_steps:
            raise ValueError(
                "task_config.max_episode_steps must equal max_episode_steps."
            )
        self.num_group = num_envs // self.group_size
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.use_rel_reward = cfg.use_rel_reward
        self.use_custom_reward = cfg.use_custom_reward
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.task_name = cfg.task_config.task_name
        self.center_crop = cfg.get("center_crop", False)
        self.state_action_order = str(cfg.get("state_action_order", "native")).lower()
        self._state_perm, self._action_perm = state_action_permutations(
            self.state_action_order, cfg.task_config.get("action_dims")
        )
        self.record_metrics = record_metrics
        self._is_start = True
        self._closed = False
        self._generator = torch.Generator().manual_seed(self.seed)
        self.success_seeds = None
        self._current_seed_index = 0
        if cfg.get("seeds_path") is not None:
            with open(cfg.seeds_path) as stream:
                seeds = json.load(stream)[self.task_name].get("success_seeds")
            if seeds is not None:
                self.success_seeds = partition_success_seeds(
                    torch.as_tensor(seeds, dtype=torch.long),
                    base_seed=self.base_seed,
                    seed_offset=seed_offset,
                    total_num_processes=total_num_processes,
                    num_group=self.num_group,
                )
                if not self.success_seeds.numel():
                    raise ValueError(
                        "Not enough success seeds for this worker's groups."
                    )
        self.update_reset_state_ids()
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
        self.prev_step_reward = torch.zeros(num_envs)
        self.episode_ids = torch.zeros(num_envs, dtype=torch.long)
        self._terminated = torch.zeros(num_envs, dtype=torch.bool)
        self._truncated = torch.zeros(num_envs, dtype=torch.bool)
        self._success = torch.zeros(num_envs, dtype=torch.bool)
        self._raw_obs: list[dict] | None = None
        if record_metrics:
            self._init_metrics()
        self._init_env()

    def _init_env(self) -> None:
        if self.cfg.get("assets_path") is not None:
            os.environ["ROBODOJO_ASSETS_ROOT"] = str(self.cfg.assets_path)
        try:
            import robodojo_runtime.bridge  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RoboDojo requires its simulator runtime. Install it in a separate "
                "environment with: bash requirements/install.sh embodied --env robodojo"
            ) from exc

        task_config = OmegaConf.to_container(self.cfg.task_config, resolve=True)
        task_config["max_episode_steps"] = self.cfg.max_episode_steps
        if task_config.get("save_dir"):
            task_config["save_dir"] = str(
                Path(task_config["save_dir"]) / f"worker_{self.seed_offset}"
            )
        self.venv = VectorRuntime(
            task_config=task_config,
            n_envs=self.num_envs,
            env_seeds=self.reset_state_ids.tolist(),
            controller=OmegaConf.to_container(
                self.cfg.get("controller", OmegaConf.create({}))
            ),
        )

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return submitted control-step counts for each slot."""
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        """Whether this wrapper has not yet been reset."""
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs)

    def _reset_metrics(self, env_idx: Any = None) -> None:
        index = slice(None) if env_idx is None else env_idx
        self._elapsed_steps[index] = 0
        self.prev_step_reward[index] = 0
        self.episode_ids[index] += 1
        self._terminated[index] = False
        self._truncated[index] = False
        self._success[index] = False
        if self.record_metrics:
            self.success_once[index] = False
            self.returns[index] = 0

    def _record_metrics(self, reward: torch.Tensor, infos: dict) -> None:
        self.returns += reward
        episode = {
            "return": self.returns.clone(),
            "episode_len": self.elapsed_steps.clone(),
            "reward": self.returns / self.elapsed_steps.clamp_min(1),
        }
        if "success" in infos:
            infos["success"] = torch.as_tensor(
                infos["success"], dtype=torch.bool
            ).reshape(-1)
            self.success_once |= infos["success"]
            episode["success_once"] = self.success_once.clone()
            if self.ignore_terminations:
                episode["success_at_end"] = infos["success"].clone()
        infos["episode"] = episode

    def _extract_obs_image(self, raw_obs: list[dict]) -> dict:
        if len(raw_obs) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} observations, got {len(raw_obs)}."
            )
        camera_keys = tuple(
            key
            for key in ("left_wrist_image", "right_wrist_image")
            if raw_obs[0].get(key) is not None
        )
        if any(
            tuple(
                key
                for key in ("left_wrist_image", "right_wrist_image")
                if obs.get(key) is not None
            )
            != camera_keys
            for obs in raw_obs
        ):
            raise ValueError("Expected the same wrist cameras in every RoboDojo slot.")

        def image_tensor(image: np.ndarray) -> torch.Tensor:
            image = Image.fromarray(np.asarray(image)).convert("RGB")
            if self.center_crop:
                image = center_crop_image(image)
            return torch.from_numpy(np.array(image))

        wrists = [
            [
                image_tensor(obs[key])
                for key in ("left_wrist_image", "right_wrist_image")
                if obs.get(key) is not None
            ]
            for obs in raw_obs
        ]
        states = torch.stack(
            [torch.as_tensor(obs["state"], dtype=torch.float32) for obs in raw_obs]
        )
        if states.shape != (self.num_envs, sum(self.action_dims)):
            raise ValueError(
                f"Expected state shape ({self.num_envs}, {sum(self.action_dims)}), "
                f"got {tuple(states.shape)}."
            )
        if self._state_perm is not None:
            states = states[..., self._state_perm]
        return {
            "main_images": torch.stack(
                [image_tensor(obs["full_image"]) for obs in raw_obs]
            ),
            "wrist_images": torch.stack([torch.stack(images) for images in wrists])
            if wrists[0]
            else None,
            "states": states,
            "task_descriptions": [obs["instruction"] for obs in raw_obs],
        }

    def reset(
        self, env_idx: int | list[int] | None = None, env_seeds: list[int] | None = None
    ) -> tuple[dict, dict]:
        """Reset selected slots and return all slots' observations and empty infos."""
        self._ensure_open()
        if env_idx is not None:
            env_idx = [env_idx] if isinstance(env_idx, int) else list(env_idx)
            if len(set(env_idx)) != len(env_idx) or any(
                index < 0 or index >= self.num_envs for index in env_idx
            ):
                raise IndexError(
                    f"Expected distinct slot IDs in [0, {self.num_envs}), got {env_idx}."
                )
        seeds = self.reset_state_ids.tolist() if env_seeds is None else list(env_seeds)
        if len(seeds) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} globally indexed seeds, got {len(seeds)}."
            )
        for attempt in range(UNSTABLE_RESET_ATTEMPTS):
            try:
                self.venv.reset(env_idx=env_idx, env_seeds=seeds)
                break
            except Exception as exc:
                if (
                    not _is_unstable_scene(exc)
                    or attempt == UNSTABLE_RESET_ATTEMPTS - 1
                ):
                    self.close()
                    raise
                get_logger().warning(
                    "RoboDojo layout did not settle; retrying the requested seeds."
                )
        try:
            raw_obs = self.venv.get_obs()
            observations = self._extract_obs_image(raw_obs)
        except BaseException:
            self.close()
            raise
        indices = slice(None) if env_idx is None else env_idx
        self.reset_state_ids[indices] = torch.as_tensor(seeds, dtype=torch.long)[
            indices
        ]
        self._reset_metrics(env_idx)
        self._raw_obs = raw_obs
        self._is_start = False
        return observations, {}

    def step(
        self,
        actions: torch.Tensor | np.ndarray | dict,
        auto_reset: bool = True,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute batched joint controls.

        Arrays use (N, A)/(N, H, A); dictionaries contain native joint groups.
        """
        self._ensure_open()
        native_dict = isinstance(actions, dict) and "actions" not in actions
        if isinstance(actions, dict):
            if native_dict:
                parts = []
                for key, width in zip(_JOINT_KEYS, self.action_dims):
                    part = actions[key]
                    if isinstance(part, torch.Tensor):
                        part = (
                            part.detach().to(device="cpu", dtype=torch.float32).numpy()
                        )
                    part = np.asarray(part)
                    if part.ndim not in (2, 3) or part.shape[-1] != width:
                        raise ValueError(
                            f"Expected {key} shape (N, [H,] {width}), got {part.shape}."
                        )
                    parts.append(part)
                actions = np.concatenate(parts, axis=-1)
            else:
                actions = actions["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        actions = np.asarray(actions)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[1] == 0
            or actions.shape[2] != sum(self.action_dims)
        ):
            raise ValueError(
                f"Expected actions shape ({self.num_envs}, positive horizon, "
                f"{sum(self.action_dims)}), got {actions.shape}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("Expected finite RoboDojo action values.")
        if self._action_perm is not None and not native_dict:
            actions = actions[..., self._action_perm]
        raw_obs, reward, terminated, truncated, info_list = self.venv.step(actions)
        return self._finish_step(
            raw_obs,
            reward,
            terminated,
            truncated,
            info_list,
            horizon=actions.shape[1],
            auto_reset=auto_reset,
        )

    def _finish_step(
        self,
        raw_obs: list[dict],
        reward: Any,
        terminated: Any,
        truncated: Any,
        info_list: list[dict],
        *,
        horizon: int,
        auto_reset: bool,
        step_counts: torch.Tensor | None = None,
        respect_terminations: bool = False,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs = self._extract_obs_image(raw_obs)
        self._raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_list)
        terminated = (
            torch.as_tensor(terminated, dtype=torch.bool).reshape(self.num_envs).clone()
        )
        truncated = (
            torch.as_tensor(truncated, dtype=torch.bool).reshape(self.num_envs).clone()
        )
        reward = torch.as_tensor(reward, dtype=torch.float32).reshape(self.num_envs)
        self._success = (
            torch.as_tensor(infos.get("success", terminated), dtype=torch.bool)
            .reshape(self.num_envs)
            .clone()
        )
        if self.use_custom_reward:
            current = self.cfg.reward_coef * self._success.float()
            reward = current - self.prev_step_reward if self.use_rel_reward else current
            self.prev_step_reward = current
        if step_counts is not None:
            reward = reward * (step_counts > 0)
        executed = torch.as_tensor(
            infos.get("executed_steps", [horizon] * self.num_envs),
            dtype=torch.long,
        ).reshape(self.num_envs)
        if ((executed < 0) | (executed > horizon)).any():
            raise ValueError(
                f"Expected executed_steps in [0, {horizon}], got {executed.tolist()}."
            )
        # Latched slots still consume rollout positions until the batch resets.
        self._elapsed_steps += horizon if step_counts is None else step_counts
        infos["valid_action_mask"] = valid_action_mask_from_counts(executed, horizon)
        truncated |= self.elapsed_steps >= self.cfg.max_episode_steps
        self._terminated = terminated.clone()
        self._truncated = truncated.clone()
        if self.record_metrics:
            self._record_metrics(reward, infos)
        if self.ignore_terminations and not respect_terminations:
            terminated.zero_()
        dones = terminated | truncated
        if auto_reset and self.auto_reset and dones.any():
            final_obs, final_info = obs, infos
            env_idx = dones.nonzero(as_tuple=True)[0].tolist()
            if self.cfg.get("is_eval", False):
                self.update_reset_state_ids(env_idx)
            obs, infos = self.reset(env_idx)
            infos.update(
                final_observation=final_obs,
                final_info=final_info,
                _final_observation=dones,
                _final_info=dones,
                _elapsed_steps=dones,
                valid_action_mask=final_info["valid_action_mask"],
            )
        return obs, reward, terminated, truncated, infos

    def snapshot(self) -> dict:
        """Return the cached batched observation without advancing physics."""
        self._check_slot(0)
        return self._extract_obs_image(self._raw_obs)

    def observe(self, env_idx: int = 0, *, refresh: bool = False) -> dict:
        """Read one slot's native state, cameras, and physical episode status.

        Joint state always uses native ordering. EEF poses use metres and wxyz
        quaternions in each arm's base frame. Returned arrays are copies.
        """
        self._check_slot(env_idx)
        if refresh:
            self._raw_obs[env_idx] = self.venv.observe(env_idx)
        import copy

        return {
            **copy.deepcopy(self._raw_obs[env_idx]),
            "elapsed_steps": int(self.elapsed_steps[env_idx]),
            "remaining_steps": max(
                0, self.cfg.max_episode_steps - int(self.elapsed_steps[env_idx])
            ),
            "success": bool(self._success[env_idx]),
            "terminated": bool(self._terminated[env_idx]),
            "truncated": bool(self._truncated[env_idx]),
        }

    def _check_slot(self, env_idx: int) -> None:
        self._ensure_open()
        if (
            isinstance(env_idx, bool)
            or not isinstance(env_idx, int)
            or not 0 <= env_idx < self.num_envs
        ):
            raise IndexError(
                f"Expected env_idx in [0, {self.num_envs}), got {env_idx!r}."
            )
        if self._raw_obs is None:
            raise RuntimeError("Call reset before controlling RoboDojo.")

    def execute(
        self, action: JointControls | JointTarget | EefTarget, *, env_idx: int = 0
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute a joint target, joint trajectory, or EEF target on one slot.

        Only actual controls consume the episode budget. Other slots stay idle;
        completed slots require reset. Tool calls and JSON are handled by agents.
        """
        self._check_slot(env_idx)
        if self._terminated[env_idx] or self._truncated[env_idx]:
            raise RuntimeError("Reset completed episodes before executing controls.")
        if not isinstance(action, (JointControls, JointTarget, EefTarget)):
            raise TypeError(
                f"Expected JointControls, JointTarget, or EefTarget, got {type(action).__name__}."
            )
        remaining = self.cfg.max_episode_steps - int(self.elapsed_steps[env_idx])
        raw, score, done, timeout, native_info = self.venv.execute(
            env_idx, action, remaining
        )
        self._raw_obs[env_idx] = raw
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        rewards[env_idx] = score
        terminations, truncations = self._terminated.clone(), self._truncated.clone()
        terminations[env_idx], truncations[env_idx] = done, timeout
        self._success[env_idx] = native_info["success"]
        counts = torch.zeros(self.num_envs, dtype=torch.long)
        counts[env_idx] = native_info["executed_steps"]
        infos = [
            {"success": bool(success), "executed_steps": int(count)}
            for success, count in zip(self._success, counts)
        ]
        result = self._finish_step(
            self._raw_obs,
            rewards,
            terminations,
            truncations,
            infos,
            horizon=max(1, native_info["requested_steps"]),
            auto_reset=False,
            step_counts=counts,
            respect_terminations=True,
        )
        result[-1]["execution"] = native_info
        return result

    def chunk_step(
        self, chunk_actions: torch.Tensor | np.ndarray
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        """Return one final observation and place aggregate results at chunk end."""
        if chunk_actions.ndim != 3 or chunk_actions.shape[1] == 0:
            raise ValueError("Chunk actions must have shape (N, positive horizon, A).")
        obs, reward, terminated, truncated, infos = self.step(chunk_actions)
        shape = (self.num_envs, chunk_actions.shape[1])
        rewards = torch.zeros(shape, dtype=torch.float32)
        terminations = torch.zeros(shape, dtype=torch.bool)
        truncations = torch.zeros(shape, dtype=torch.bool)
        rewards[:, -1] = reward
        terminations[:, -1] = terminated
        truncations[:, -1] = truncated
        return [obs], rewards, terminations, truncations, [infos]

    def update_reset_state_ids(self, env_idx: list[int] | None = None) -> None:
        """Select grouped seeds, preserving unselected slots and fixed reset IDs."""
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return
        if self.success_seeds is None:
            seeds = torch.randint(
                10000, 200000, (self.num_group,), generator=self._generator
            )
        else:
            indices = (
                torch.arange(self.num_group) + self._current_seed_index
            ) % self.success_seeds.numel()
            seeds = self.success_seeds[indices]
            self._current_seed_index = (
                self._current_seed_index + self.num_group
            ) % self.success_seeds.numel()
        seeds = seeds.repeat_interleave(self.group_size)
        if env_idx is None:
            self.reset_state_ids = seeds
        else:
            self.reset_state_ids[env_idx] = seeds[env_idx]

    def check_seeds(self, seeds: list[int]) -> list[bool]:
        """Delegate seed validity checks to the runtime."""
        self._ensure_open()
        return self.venv.check_seeds(seeds)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "RoboDojo environment is closed; construct a new instance."
            )

    def close(self) -> None:
        """Release simulator resources; subsequent calls are harmless."""
        self.offload()

    def offload(self, clear_cache: bool = True) -> None:
        """Release the owned bridge once; repeated calls are harmless."""
        if not self._closed:
            self._closed = True
            self.venv.close(clear_cache)
