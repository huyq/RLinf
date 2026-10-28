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

"""Slot isolation and Cartesian control over the optional native runtime."""

import multiprocessing
import traceback
from pathlib import Path

import numpy as np

from rlinf.envs.sim.robodojo.actions import (
    EefTarget,
    InvalidAction,
    JointControls,
    JointTarget,
    interpolate_joints,
)


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def transform_pose(
    base: np.ndarray, pose: np.ndarray, *, inverse: bool = False
) -> np.ndarray:
    """Compose a base pose with a local pose, or express a world pose in base."""
    base, pose = np.asarray(base, dtype=float), np.asarray(pose, dtype=float)
    if (
        base.shape != (7,)
        or pose.shape != (7,)
        or not np.isfinite(base).all()
        or not np.isfinite(pose).all()
    ):
        raise RuntimeError("Expected finite 7D robot base and end-effector poses.")
    q = base[3:]
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise RuntimeError("Expected a unit robot base quaternion.")
    if inverse:
        q = q * [1, -1, -1, -1]
        point = pose[:3] - base[:3]
    else:
        point = pose[:3]
    rotated = _multiply(_multiply(q, np.r_[0, point]), q * [1, -1, -1, -1])[1:]
    return np.r_[rotated if inverse else rotated + base[:3], _multiply(q, pose[3:])]


class SceneController:
    """Own native execution, IK, coordinate conversion, and motion limits."""

    def __init__(self, native, settings: dict) -> None:
        self.native = native
        self.settings = settings

    def observe(self) -> dict:
        """Return native observations with per-arm base-frame end-effector poses."""
        return self._with_eef(self.native.get_obs())

    def _with_eef(self, observation: dict) -> dict:
        obs = dict(observation)
        manager = self.native.env.robot_manager
        poses = {}
        for robot in manager.robot_list:
            if robot.type == "target":
                side = robot.arm_name.split("_")[0]
                world = manager.get_real_endpose(
                    robot, env_idx_list=[0], is_relative=False
                )[0]
                poses[side] = transform_pose(
                    robot.entity_origin_pose, world, inverse=True
                )
        obs["eef_poses"] = poses
        return obs

    def step(self, controls: np.ndarray) -> tuple:
        """Execute native controls and enrich the returned observation."""
        obs, reward, done, timeout, info = self.native.step(controls)
        return self._with_eef(obs), reward, done, timeout, info

    def execute(
        self, action: JointControls | JointTarget | EefTarget, remaining: int
    ) -> tuple:
        """Execute a joint trajectory or solve an EEF target before bounded motion."""
        if isinstance(action, JointControls):
            controls = action.values
        elif isinstance(action, JointTarget):
            if action.values.shape != (sum(self.native.action_dims),):
                raise InvalidAction(
                    f"Expected {sum(self.native.action_dims)} native target dimensions, got {action.values.shape}."
                )
            current = np.asarray(self.native.get_obs()["state"], dtype=float)
            controls = interpolate_joints(
                current,
                action.values,
                self.settings["max_joint_step"],
                self.settings["max_gripper_step"],
                self.settings["max_motion_steps"],
            )
        elif isinstance(action, EefTarget):
            current = np.asarray(self.native.get_obs()["state"], dtype=float)
            target = current.copy()
            target[-2:] = np.clip(target[-2:], 0, 1)
            manager = self.native.env.robot_manager
            dims = self.native.action_dims
            if dims[2:] != (1, 1):
                raise InvalidAction("EEF control requires one scalar gripper per arm.")
            robots = {
                robot.arm_name.split("_")[0]: robot
                for robot in manager.robot_list
                if robot.type == "target"
            }
            for side, offset, width in (
                ("left", 0, dims[0]),
                ("right", dims[0], dims[1]),
            ):
                pose = getattr(action, side)
                if pose is None:
                    continue
                if side not in robots:
                    raise InvalidAction(f"No target robot for {side} arm.")
                robot = robots[side]
                world = transform_pose(robot.entity_origin_pose, pose)
                result = manager.solve_ik(
                    target_pose=world.tolist(), env_idx=0, robot=robot
                )
                if result["status"] != "Success":
                    raise InvalidAction(
                        f"IK failed for {side} arm; no controls executed."
                    )
                joints = np.asarray(result["joint_value"], dtype=float).reshape(-1)
                if joints.shape != (width,) or not np.isfinite(joints).all():
                    raise RuntimeError(
                        f"Expected {width} finite IK joints for {side} arm."
                    )
                target[offset : offset + width] = joints
            for i, side in enumerate(("left", "right")):
                grip = getattr(action, side + "_gripper")
                if grip is not None:
                    target[-2 + i] = grip
            controls = interpolate_joints(
                current,
                target,
                self.settings["max_joint_step"],
                self.settings["max_gripper_step"],
                self.settings["max_motion_steps"],
            )
        else:
            raise TypeError(f"Unsupported physical action: {type(action).__name__}.")
        if controls.shape[1] != sum(self.native.action_dims):
            raise InvalidAction(
                f"Expected {sum(self.native.action_dims)} native action dimensions, got {controls.shape[1]}."
            )
        requested = len(controls)
        obs, reward, done, timeout, info = self.step(controls[: max(0, remaining)])
        return (
            obs,
            reward,
            done,
            timeout or remaining <= 0,
            {**info, "requested_steps": requested},
        )

    def call(self, method: str, *args):
        """Dispatch simulator operations on its owning process thread."""
        if method in ("observe", "execute", "step"):
            return getattr(self, method)(*args)
        return getattr(self.native, method)(*args)


def _serve(connection, config: dict, settings: dict) -> None:
    scene = None
    try:
        from robodojo_runtime.bridge import NativeEnv

        scene = SceneController(NativeEnv(config), settings)
        connection.send((True, None))
        while True:
            method, args = connection.recv()
            try:
                connection.send((True, scene.call(method, *args)))
            except InvalidAction as error:
                connection.send((False, ("value", str(error))))
            except Exception:
                connection.send((False, ("runtime", traceback.format_exc())))
            if method == "close":
                break
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send((False, ("runtime", traceback.format_exc())))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        try:
            if scene is not None:
                scene.native.close(False)
        finally:
            connection.close()


class _RemoteScene:
    def __init__(self, config: dict, settings: dict) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, config, settings))
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.connection.close()
            self.process.join()
            raise

    def _receive(self):
        ok, result = self.connection.recv()
        if not ok:
            kind, message = result
            raise InvalidAction(message) if kind == "value" else RuntimeError(message)
        return result

    def call(self, method: str, *args):
        self.connection.send((method, args))
        return self._receive()

    def close(self, clear_cache: bool = True) -> None:
        try:
            self.call("close", clear_cache)
        finally:
            self.connection.close()
            self.process.join()


class VectorRuntime:
    """Own one native scene per slot, using spawned processes for multiple slots."""

    def __init__(
        self,
        task_config: dict,
        n_envs: int,
        env_seeds: list[int],
        controller: dict | None = None,
    ) -> None:
        from robodojo_runtime.bridge import NativeEnv

        self.settings = {
            "max_joint_step": 0.05,
            "max_gripper_step": 0.05,
            "max_motion_steps": 100,
            **(controller or {}),
        }
        for key in ("max_joint_step", "max_gripper_step", "max_motion_steps"):
            value = self.settings[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Expected positive finite {key}, got {value!r}.")
        if not isinstance(self.settings["max_motion_steps"], int):
            raise ValueError("Expected integer max_motion_steps.")
        self.env_seeds = list(env_seeds)
        self.scenes = []
        self.closed = False
        try:
            for i in range(n_envs):
                config = {
                    **task_config,
                    "save_dir": str(
                        Path(task_config.get("save_dir", ".")) / f"env_{i}"
                    ),
                }
                self.scenes.append(
                    SceneController(NativeEnv(config), self.settings)
                    if n_envs == 1
                    else _RemoteScene(config, self.settings)
                )
        except BaseException:
            self.close(False)
            raise

    def reset(self, env_idx=None, env_seeds=None) -> None:
        indices = range(len(self.scenes)) if env_idx is None else env_idx
        seeds = self.env_seeds if env_seeds is None else env_seeds
        for i in indices:
            self.scenes[i].call("reset", seeds[i])
            self.env_seeds[i] = seeds[i]

    def get_obs(self) -> list[dict]:
        return [scene.call("observe") for scene in self.scenes]

    def observe(self, index: int) -> dict:
        return self.scenes[index].call("observe")

    def execute(
        self,
        index: int,
        action: JointControls | JointTarget | EefTarget,
        remaining: int,
    ) -> tuple:
        return self.scenes[index].call("execute", action, remaining)

    def step(self, actions: np.ndarray) -> tuple:
        results = [
            scene.call("step", controls)
            for scene, controls in zip(self.scenes, actions)
        ]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return list(obs), rewards, terminated, truncated, list(infos)

    def check_seeds(self, seeds: list[int]) -> list[bool]:
        return self.scenes[0].call("check_seeds", seeds)

    def close(self, clear_cache: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        errors = []
        for scene in self.scenes:
            try:
                if isinstance(scene, _RemoteScene):
                    scene.close(clear_cache)
                else:
                    scene.native.close(clear_cache)
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
