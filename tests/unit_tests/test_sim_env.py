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

"""Simulator contracts exercised through the optional runtime boundary."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.agents.robodojo.tools import RoboDojoAgentTools
from rlinf.envs import get_env_cls
from rlinf.envs.sim.robodojo.actions import EefTarget, JointControls
from rlinf.envs.sim.robodojo.runtime import transform_pose

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = (
    _ROOT / "examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml"
)


@pytest.fixture
def make_env(monkeypatch):
    monkeypatch.syspath_prepend(str(_ROOT / "tests/robot_mocks"))
    for name in ("robodojo_runtime", "robodojo_runtime.bridge"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    envs = []

    def create(num_envs=1, **overrides):
        cfg = OmegaConf.merge(OmegaConf.load(_CONFIG), overrides)
        env = get_env_cls("robodojo")(cfg, num_envs, 0, 1, None)
        envs.append(env)
        return env

    yield create
    for env in envs:
        env.close()
    for name in ("robodojo_runtime", "robodojo_runtime.bridge"):
        sys.modules.pop(name, None)


def test_robodojo_native_observations_and_array_layout(make_env):
    env = make_env(state_action_order="aloha")
    obs, info = env.reset(env_seeds=[11])
    assert obs["main_images"].shape == (1, 8, 8, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["wrist_images"].shape == (1, 2, 8, 8, 3)
    assert obs["task_descriptions"] == ["task seed 11"]
    assert info == {}
    assert obs["states"][0, 6] == 0.5
    action = torch.zeros(1, 14, dtype=torch.bfloat16)
    action[0, 6], action[0, 7], action[0, 13] = 0.3, 0.1, 0.8
    env.step(action)
    raw = env.observe()
    np.testing.assert_allclose(
        raw["state"][[12, 6, 13]], [0.30078125, 0.10009765625, 0.80078125]
    )
    assert raw["elapsed_steps"] == 1
    assert not hasattr(env, "step_agent") and not hasattr(env, "agent_tools")


def test_robodojo_native_dictionary_is_independent_of_aloha(make_env):
    env = make_env(state_action_order="aloha")
    env.reset()
    env.step(
        {
            "left_arm_joint_state": np.zeros((1, 6)),
            "right_arm_joint_state": np.full((1, 6), 0.1),
            "left_ee_joint_state": np.full((1, 1), 0.2),
            "right_ee_joint_state": np.full((1, 1), 0.8),
        }
    )
    np.testing.assert_allclose(env.observe()["state"], [0] * 6 + [0.1] * 6 + [0.2, 0.8])


def test_robodojo_chunk_autoreset_and_prefix_mask(make_env):
    env = make_env(auto_reset=True, task_config={"finish_at": 1})
    env.reset()
    obs, reward, term, trunc, infos = env.chunk_step(torch.zeros(1, 3, 14))
    assert len(obs) == len(infos) == 1
    assert reward.tolist() == [[0, 0, 1]] and term.tolist() == [[False, False, True]]
    assert not trunc.any()
    assert infos[0]["valid_action_mask"].tolist() == [[True, False, False]]
    assert infos[0]["final_info"]["episode"]["success_once"].all()
    assert not env.elapsed_steps.any()


@pytest.mark.parametrize(
    "action",
    [
        np.zeros((2, 1, 14)),
        np.zeros((1, 0, 14)),
        np.zeros((1, 1, 7)),
        np.full((1, 1, 14), np.nan),
        {"name": "move_joints", "arguments": {}},
    ],
)
def test_robodojo_rejects_nonphysical_array_actions(make_env, action):
    env = make_env()
    env.reset()
    with pytest.raises((ValueError, KeyError)):
        env.step(action)
    assert not env.elapsed_steps.any()


def test_robodojo_physical_ee_control_and_base_frames(make_env):
    env = make_env()
    env.reset()
    np.testing.assert_allclose(
        env.observe()["eef_poses"]["left"], [0, 0, 0, 1, 0, 0, 0], atol=1e-6
    )
    _, reward, term, trunc, info = env.execute(
        EefTarget(left=(0.12, 0.03, 0, 1, 0, 0, 0), left_gripper=0.6)
    )
    assert info["execution"]["executed_steps"] == 3
    np.testing.assert_allclose(
        env.observe()["eef_poses"]["left"][:3], [0.12, 0.03, 0], atol=1e-6
    )
    np.testing.assert_allclose(env.observe()["state"][6:12], 0)
    assert env.observe()["state"][-2] == pytest.approx(0.6)
    assert not reward.any() and not term.any() and not trunc.any()


def test_robodojo_ik_failure_is_atomic_for_both_arms(make_env):
    env = make_env()
    env.reset()
    before = env.observe()["state"].copy()
    with pytest.raises(ValueError, match="IK failed for right"):
        env.execute(
            EefTarget(left=(0.1, 0, 0, 1, 0, 0, 0), right=(2, 0, 0, 1, 0, 0, 0))
        )
    np.testing.assert_array_equal(env.observe()["state"], before)
    assert not env.elapsed_steps.any()


def test_robodojo_spawned_slot_isolation_partial_reset_and_remote_ik(make_env):
    env = make_env(num_envs=2)
    env.reset(env_seeds=[11, 22])
    env.execute(EefTarget(left=(0.08, 0, 0, 1, 0, 0, 0)), env_idx=1)
    assert env.elapsed_steps.tolist() == [0, 2]
    assert env.observe(0)["state"][0] == 0
    assert env.observe(1)["state"][0] == pytest.approx(0.08)
    with pytest.raises(ValueError, match="IK failed"):
        env.execute(EefTarget(right=(2, 0, 0, 1, 0, 0, 0)), env_idx=1)
    env.reset(env_idx=0, env_seeds=[33, 99])
    assert env.observe(0)["instruction"] == "task seed 33"
    assert env.observe(1)["instruction"] == "task seed 22"
    assert env.elapsed_steps.tolist() == [0, 2]
    env.close()
    env.close()


def test_robodojo_physical_budget_and_terminal_latching(make_env):
    env = make_env(max_episode_steps=2, task_config={"max_episode_steps": 2})
    env.reset()
    _, _, _, trunc, info = env.execute(JointControls(np.zeros((5, 14))))
    assert trunc.tolist() == [True]
    assert info["execution"]["requested_steps"] == 5
    assert info["execution"]["executed_steps"] == 2
    with pytest.raises(RuntimeError, match="Reset completed"):
        env.execute(JointControls(np.zeros((1, 14))))


def test_robodojo_pose_transform_roundtrip():
    q = np.sqrt(0.5)
    base = np.array([10, 20, 30, q, 0, 0, q])
    local = np.array([1, 2, 3, q, q, 0, 0])
    world = transform_pose(base, local)
    np.testing.assert_allclose(world[:3], [8, 21, 33])
    np.testing.assert_allclose(
        transform_pose(base, world, inverse=True), local, atol=1e-6
    )


def test_robodojo_agent_joint_tools_and_prefix_feedback(make_env):
    env = make_env(task_config={"finish_at": 1})
    env.reset()
    tools = RoboDojoAgentTools(env)
    _, reward, term, _, info = tools.call(
        "move_joints", {"targets": {"left_j0": 0.12}, "note": "Approach."}
    )
    assert info["agent"]["requested_steps"] == 3
    assert info["agent"]["executed_steps"] == 1
    assert info["agent"]["residuals"]["left_j0"] == pytest.approx(-0.08)
    assert reward.tolist() == [1] and term.tolist() == [True]
    assert tools.observe()["success"]
    _, reward, _, _, _ = tools.call("done", {"summary": "Finished."})
    assert not reward.any()


def test_robodojo_agent_eef_tool_and_json_envelope(make_env):
    import json

    env = make_env()
    env.reset()
    tools = RoboDojoAgentTools(env)
    assert {item["function"]["name"] for item in tools.schemas()} == {
        "move_joints",
        "move_eef",
        "wait",
        "take_pic",
        "done",
        "give_up",
    }
    _, _, _, _, info = tools.step(
        {
            "type": "function",
            "function": {
                "name": "move_eef",
                "arguments": '{"targets":{"left":[0.08,0,0,0,0,0]},"note":"Reach."}',
            },
        }
    )
    assert info["agent"]["ok"]
    assert tools.observe()["joints"]["left_j0"] == pytest.approx(0.08)
    json.dumps(info["agent"], allow_nan=False)
    _, _, _, _, info = tools.call(
        "move_eef", {"targets": {"right": [2, 0, 0, 0, 0, 0]}, "note": "Reach."}
    )
    assert not info["agent"]["ok"] and "IK failed" in info["agent"]["error"]
    assert env.elapsed_steps.tolist() == [2]


@pytest.mark.parametrize(
    "arguments",
    [
        "not JSON",
        "[]",
        {"targets": {"left_j0": 0.1}},
        {"targets": {"bad": 0.1}, "note": "Move."},
        {"targets": {"left_j0": True}, "note": "Move."},
        {"targets": {"left_j0": float("nan")}, "note": "Move."},
        {"targets": {"left_j0": 10**400}, "note": "Move."},
        {"targets": {"left_gripper": 2}, "note": "Open."},
        {"targets": {"left_j0": 1000}, "note": "Move."},
    ],
)
def test_robodojo_agent_invalid_calls_consume_turn_without_motion(make_env, arguments):
    env = make_env()
    env.reset()
    tools = RoboDojoAgentTools(env, {"max_turns": 2})
    _, reward, _, _, info = tools.call("move_joints", arguments)
    assert not info["agent"]["ok"]
    assert tools.observe()["turns"] == 1
    _, _, _, trunc, _ = tools.call("unknown", {})
    assert trunc.tolist() == [True]
    assert not env.observe()["truncated"]
    assert not reward.any() and not env.elapsed_steps.any()


@pytest.mark.parametrize(
    "name,arguments",
    [("done", {"summary": "Finished."}), ("give_up", {"reason": "Unreachable."})],
)
def test_robodojo_agent_stop_and_reset_are_separate_from_physics(
    make_env, name, arguments
):
    env = make_env(auto_reset=True, ignore_terminations=True, use_rel_reward=False)
    env.reset()
    tools = RoboDojoAgentTools(env)
    _, reward, term, _, info = tools.call(name, arguments)
    assert term.tolist() == [True] and not reward.any()
    assert not env.observe()["terminated"] and not tools.observe()["success"]
    assert info["agent"]["stop_reason"] == name
    _, _, _, _, info = tools.call("wait", {"steps": 1, "note": "Wait."})
    assert "reset" in info["agent"]["error"]
    env.reset()
    assert tools.observe()["remaining_turns"] == 64
    tools.call("wait", {"steps": 1, "note": "Wait."})
    assert env.elapsed_steps.tolist() == [1]


def test_robodojo_agent_camera_refresh_and_joint_limits(make_env):
    env = make_env()
    tools = RoboDojoAgentTools(env, {"joint_limits": {"left_j0": [-0.1, 0.1]}})
    with pytest.raises(RuntimeError, match="reset"):
        tools.observe()
    env.reset()
    _, _, _, _, info = tools.call(
        "move_joints", {"targets": {"left_j0": 0.2}, "note": "Move."}
    )
    assert not info["agent"]["ok"]
    tools.call("take_pic", {"note": "Inspect."})
    assert not env.elapsed_steps.any()
    assert tools.observe()["cameras"] == ["head", "left_wrist", "right_wrist"]


def test_robodojo_seed_groups_and_cleanup(make_env):
    env = make_env(num_envs=2, group_size=2, use_fixed_reset_state_ids=True)
    assert env.reset_state_ids[0] == env.reset_state_ids[1]
    with pytest.raises(IndexError):
        env.reset(env_idx=[0, 0])
    with pytest.raises(ValueError, match="globally indexed"):
        env.reset(env_idx=0, env_seeds=[1])
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros((2, 1, 14)))


def test_robodojo_missing_runtime_is_lazy(monkeypatch):
    monkeypatch.setitem(sys.modules, "robodojo_runtime", None)
    monkeypatch.delitem(sys.modules, "robodojo_runtime.bridge", raising=False)
    with pytest.raises(ImportError, match="embodied --env robodojo"):
        get_env_cls("robodojo")(OmegaConf.load(_CONFIG), 1, 0, 1, None)


def test_robodojo_agent_eef_orientation_uses_yaw_pitch_roll(make_env):
    env = make_env()
    env.reset()
    agent = RoboDojoAgentTools(env)
    target = [0.08, 0.03, 0.02, 0.3, -0.2, 0.1]
    _, _, _, _, info = agent.call(
        "move_eef", {"targets": {"left": target}, "note": "Orient the gripper."}
    )
    assert info["agent"]["ok"]
    np.testing.assert_allclose(agent.observe()["eef_poses"]["left"], target, atol=1e-6)


def test_robodojo_runtime_faults_are_not_tool_feedback(make_env):
    env = make_env(task_config={"step_fault": True})
    env.reset()
    agent = RoboDojoAgentTools(env)
    with pytest.raises(ValueError, match="simulator fault"):
        agent.call("wait", {"steps": 1, "note": "Wait."})


def test_robodojo_agent_partial_reset_preserves_other_stop_state(make_env):
    env = make_env(num_envs=2)
    env.reset()
    agent = RoboDojoAgentTools(env)
    agent.call("done", {"summary": "Stop this slot."}, env_idx=1)
    env.reset(env_idx=0)
    assert agent.observe(0)["remaining_turns"] == 64
    assert agent.observe(1)["terminated"]
    assert agent.observe(1)["turns"] == 1
    env.reset(env_idx=1)
    assert not agent.observe(1)["terminated"]
    assert agent.observe(1)["remaining_turns"] == 64


def test_robodojo_unstable_reset_retries_same_layout_then_closes(make_env, monkeypatch):
    env = make_env(use_fixed_reset_state_ids=True)
    from robodojo_runtime.bridge import NativeEnv

    reset = NativeEnv.reset
    attempts = []

    def unstable_reset(native, seed=None):
        attempts.append(seed)
        if len(attempts) < 3:
            raise RuntimeError("UnStableError: layout did not settle")
        return reset(native, seed)

    monkeypatch.setattr(NativeEnv, "reset", unstable_reset)
    env.reset(env_seeds=[11])
    assert attempts == [11, 11, 11]

    def always_unstable(native, seed=None):
        attempts.append(seed)
        raise RuntimeError("UnStableError: layout did not settle")

    monkeypatch.setattr(NativeEnv, "reset", always_unstable)
    attempts.clear()
    with pytest.raises(RuntimeError, match="UnStableError"):
        env.reset()
    assert attempts == [11] * 8
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros((1, 1, 14)))


def test_robodojo_saved_seed_groups_disjoint_across_workers(make_env, tmp_path):
    import json

    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps({"put_bottles_into_dustbin": {"success_seeds": list(range(32))}})
    )
    cfg = OmegaConf.merge(
        OmegaConf.load(_CONFIG), {"seeds_path": str(path), "group_size": 2}
    )
    cls = get_env_cls("robodojo")
    first = cls(cfg, 2, 0, 2, None)
    try:
        second = cls(cfg, 2, 1, 2, None)
        try:
            assert not set(first.success_seeds.tolist()) & set(
                second.success_seeds.tolist()
            )
            assert first.reset_state_ids[0] == first.reset_state_ids[1]
            assert second.reset_state_ids[0] == second.reset_state_ids[1]
        finally:
            second.close()
    finally:
        first.close()


def test_robodojo_agent_holds_grippers_with_sensor_overshoot(make_env):
    env = make_env()
    env.reset()
    env.venv.scenes[0].native.state[-2:] = [-0.0001, 1.0001]
    env.observe(refresh=True)
    agent = RoboDojoAgentTools(env)
    _, _, _, _, info = agent.call("wait", {"steps": 1, "note": "Hold position."})
    assert info["agent"]["ok"]
    np.testing.assert_allclose(env.observe()["state"][-2:], [0, 1])
