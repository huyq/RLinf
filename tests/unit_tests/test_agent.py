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
"""RoboDojo agent sessions, camera trajectories, and real visual policy updates."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
import torch
from aiohttp.test_utils import TestServer
from omegaconf import OmegaConf

from rlinf.agents.robodojo.actor import partition_vision_batches, vision_logprobs
from rlinf.agents.robodojo.config import validate_robodojo_cfg
from rlinf.agents.robodojo.rollout import pack_episodes, parse_action, run_episode
from rlinf.agents.robodojo.service import RoboDojoClient, RoboDojoSessions, create_app
from rlinf.algorithms.advantages import compute_grpo_dynamic_advantages
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.envs import get_env_cls

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "examples/agent/robodojo/config/qwen3_8_27b_grpo.yaml"


@pytest.fixture
def robot_env(monkeypatch):
    monkeypatch.syspath_prepend(str(_ROOT / "tests/robot_mocks"))
    for name in ("robodojo_runtime", "robodojo_runtime.bridge"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    cfg = OmegaConf.load(
        _ROOT / "examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml"
    )
    cfg.task_config.success_target = 0.1
    env = get_env_cls("robodojo")(cfg, 1, 0, 1, None)
    env.reset(env_seeds=[0])
    yield env
    env.close()
    for name in ("robodojo_runtime", "robodojo_runtime.bridge"):
        sys.modules.pop(name, None)


def tiny_qwen_processor(schemas):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    tokens = [
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    corpus = (
        json.dumps(schemas)
        + " "
        + (_ROOT / "rlinf/agents/robodojo/rollout.py").read_text()
    )
    backend.train_from_iterator(
        [corpus],
        trainers.BpeTrainer(
            vocab_size=1024,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=tokens,
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=tokens[1:],
    )
    template = "{% for message in messages %}<|im_start|>{{message.role}}\n{% if message.content is string %}{{message.content}}{% else %}{% for part in message.content %}{% if part.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{part.text}}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tokenizer.chat_template = template
    processor = Qwen3VLProcessor(
        video_processor=Qwen3VLVideoProcessor(),
        image_processor=Qwen2VLImageProcessor(
            size={"shortest_edge": 1024, "longest_edge": 1024},
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
        ),
        tokenizer=tokenizer,
        chat_template=template,
    )
    return processor


def tiny_qwen_model(processor):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    tokenizer = processor.tokenizer
    config = Qwen3_5Config(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 2],
                "partial_rotary_factor": 1.0,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    config._attn_implementation = "eager"
    torch.manual_seed(11)
    return Qwen3_5ForConditionalGeneration(config).eval()


@pytest.mark.parametrize("tool_format", ["json", "qwen_xml"])
def test_robodojo_camera_grpo_updates_real_qwen_vision_and_language(
    robot_env, tmp_path, tool_format
):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    async def run():
        sessions = RoboDojoSessions(robot_env, max_turns=4, image_size=32)
        processor = tiny_qwen_processor(sessions.agent.schemas())
        model = tiny_qwen_model(processor)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        server = TestServer(create_app(sessions))
        await server.start_server()
        seen_pixels = []

        def engine(actions):
            iterator = iter(actions)

            async def generate(prompt_ids, image_data, sampling_params):
                assert (
                    len(image_data) == 1
                    and image_data[0]["format"] == "processor_output"
                )
                mm = {
                    key: value
                    for key, value in image_data[0].items()
                    if key != "format"
                }
                seen_pixels.append(mm["pixel_values"].clone())
                action = next(iterator)
                if tool_format == "qwen_xml":
                    body = f"<function={action['name']}>\n"
                    for name, value in action["arguments"].items():
                        value = value if isinstance(value, str) else json.dumps(value)
                        body += f"<parameter={name}>\n{value}\n</parameter>\n"
                    body += "</function>"
                else:
                    body = json.dumps(action)
                text = "<tool_call>" + body + "</tool_call>"
                response = processor.tokenizer.encode(
                    text, add_special_tokens=False
                ) + [processor.tokenizer.eos_token_id]
                assert len(response) <= sampling_params["max_new_tokens"]
                ids = torch.tensor([prompt_ids + response])
                batch = {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids, dtype=torch.bool),
                    "multi_modal_inputs": [
                        {
                            **mm,
                            "mm_token_type_ids": (
                                ids == model.config.image_token_id
                            ).long(),
                        }
                    ],
                    "prompt_lengths": torch.tensor([len(prompt_ids)]),
                }
                with torch.no_grad():
                    scores = vision_logprobs(model, batch, 1.0)[
                        0, len(prompt_ids) :
                    ].tolist()
                return {
                    "output_ids": response,
                    "logprobs": scores,
                    "finish_reason": "stop",
                }

            return generate

        async def trajectory(actions):
            async with RoboDojoClient(str(server.make_url(""))) as client:
                return await run_episode(
                    client,
                    processor,
                    engine(actions),
                    seed=7,
                    max_turns=4,
                    max_prompt_length=4096,
                    seq_length=4608,
                    max_new_tokens=512,
                )

        try:
            success = [
                {"name": "wait", "arguments": {"steps": 1, "note": "Inspect cameras."}},
                {
                    "name": "move_eef",
                    "arguments": {
                        "targets": {"left": [0.12, 0, 0, 0, 0, 0]},
                        "note": "Reach the target.",
                    },
                },
            ]
            failure = [
                {
                    "name": "done",
                    "arguments": {"summary": "Stop without completing the task."},
                }
            ]
            episodes = await asyncio.gather(trajectory(success), trajectory(failure))
            assert sessions.free.qsize() == 1 and not sessions.sessions
            assert [episode.extra_fields["task_return"] for episode in episodes] == [
                1,
                0,
            ]
            result = pack_episodes(episodes)
            assert result.idx_to_traj == [0, 0, 1]
            assert result.rewards == [1, 1, 0]
            assert result.is_end == [False, True, True]
            batch = result.to_actor_batch(
                4608, processor.tokenizer.pad_token_id, device="cpu"
            )
            assert batch["multi_modal_inputs"][0]["image_grid_thw"].shape == (3, 3)
            assert not batch["response_mask"][
                batch["input_ids"] == model.config.image_token_id
            ].any()
            assert not torch.equal(
                result.multi_modal_inputs[0]["pixel_values"],
                result.multi_modal_inputs[1]["pixel_values"],
            )
            for expected, actual in zip(
                seen_pixels,
                [
                    result.multi_modal_inputs[0],
                    result.multi_modal_inputs[1],
                    result.multi_modal_inputs[2],
                ],
            ):
                torch.testing.assert_close(expected, actual["pixel_values"])
            advantages, _ = compute_grpo_dynamic_advantages(
                batch["rewards"][:, None],
                batch["response_mask"].T,
                2,
                result.idx_to_traj,
                advantage_mode="trajectory",
            )
            batch["advantages"] = advantages.T
            batch["loss_scales"] = batch["response_mask"].float()
            partitions = partition_vision_batches([batch], 2)
            assert [len(part["input_ids"]) for part in partitions] == [2, 2]
            assert sum(int(part["response_mask"].sum()) for part in partitions) == int(
                batch["response_mask"].sum()
            )
            assert all(part["multi_modal_inputs"] for part in partitions)
            visual = next(
                parameter
                for name, parameter in model.named_parameters()
                if "visual" in name
            )
            language = next(
                parameter
                for name, parameter in model.named_parameters()
                if "embed_tokens" in name
            )
            visual_before, language_before = (
                visual.detach().clone(),
                language.detach().clone(),
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
            optimizer.zero_grad()
            for index in range(result.num_sequence):
                micro = {
                    key: value[index : index + 1]
                    for key, value in batch.items()
                    if isinstance(value, (torch.Tensor, list))
                }
                scores = vision_logprobs(model, micro, 1.0)
                loss, _ = compute_ppo_actor_loss(
                    scores,
                    micro["rollout_logprobs"],
                    0.2,
                    0.2,
                    micro["advantages"].float(),
                    loss_mask=micro["response_mask"],
                )
                (loss / result.num_sequence).backward()
            padded = next(
                {key: value[index : index + 1] for key, value in part.items()}
                for part in partitions
                for index in range(len(part["input_ids"]))
                if not part["response_mask"][index].any()
            )
            padded_scores = vision_logprobs(model, padded, 1.0)
            padded_loss, _ = compute_ppo_actor_loss(
                padded_scores,
                padded["rollout_logprobs"],
                0.2,
                0.2,
                padded["advantages"].float(),
                loss_mask=padded["response_mask"],
                fast_path_zero_loss_mask=True,
            )
            assert padded_loss.requires_grad and padded_loss.item() == 0
            padded_loss.backward()
            assert torch.isfinite(visual.grad).all() and visual.grad.abs().sum() > 0
            assert torch.isfinite(language.grad).all() and language.grad.abs().sum() > 0
            optimizer.step()
            assert not torch.equal(visual_before, visual)
            assert not torch.equal(language_before, language)
            from transformers import AutoModelForImageTextToText, AutoProcessor

            from rlinf.agents.robodojo.checkpoint import export_checkpoint

            base = tmp_path / "original"
            model.save_pretrained(base)
            processor.save_pretrained(base)
            weights = tmp_path / "full_weights.pt"
            torch.save(model.state_dict(), weights)
            exported = tmp_path / "exported"
            export_checkpoint(str(weights), str(base), str(exported))
            reloaded = AutoModelForImageTextToText.from_pretrained(exported)
            assert reloaded.config.model_type == "qwen3_5"
            for name, value in model.state_dict().items():
                torch.testing.assert_close(reloaded.state_dict()[name], value)
            assert (
                AutoProcessor.from_pretrained(exported).tokenizer.eos_token_id
                == processor.tokenizer.eos_token_id
            )
            split = DynamicRolloutResult.split_results(result, [2, 1])
            merged = DynamicRolloutResult.merge_result_list(split)
            for left, right in zip(
                merged.multi_modal_inputs, result.multi_modal_inputs
            ):
                torch.testing.assert_close(left["pixel_values"], right["pixel_values"])
        finally:
            await server.close()
            torch.set_num_threads(old_threads)

    asyncio.run(run())


def test_robodojo_http_session_cleanup_after_rollout_failure(robot_env):
    async def run():
        sessions = RoboDojoSessions(robot_env, image_size=32)
        server = TestServer(create_app(sessions))
        await server.start_server()
        try:
            with pytest.raises(RuntimeError, match="engine failed"):
                async with RoboDojoClient(str(server.make_url(""))) as client:
                    state = await client.start(3)
                    assert len(state["images"]) == 3
                    raise RuntimeError("engine failed")
            assert sessions.free.qsize() == 1 and not sessions.sessions
            async with RoboDojoClient(str(server.make_url(""))) as client:
                state = await client.start(4)
                assert "seed 4" in state["observation"]["instruction"]
                bad = await client.step(
                    {
                        "name": "move_eef",
                        "arguments": {
                            "targets": {"left": [2, 0, 0, 0, 0, 0]},
                            "note": "Unreachable.",
                        },
                    }
                )
                assert (
                    not bad["feedback"]["ok"]
                    and "IK failed" in bad["feedback"]["error"]
                )
                assert not bad["reward"]
        finally:
            await server.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor.pack_traj", True),
        ("runner.enable_dynamic_batch_size", True),
        ("algorithm.recompute_logprobs", True),
        ("algorithm.kl_beta", 0.01),
        ("rollout.detokenize", False),
        ("rollout.rollout_backend", "vllm"),
        ("algorithm.training_batch_size_per_gpu", 2),
        ("agentloop.image_size", 127),
    ],
)
def test_robodojo_camera_configuration_rejects_incompatible_paths(path, value):
    cfg = OmegaConf.load(_CONFIG)
    validate_robodojo_cfg(cfg)
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        validate_robodojo_cfg(cfg)


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "not JSON",
        '<tool_call>{"name":"done","arguments":{}}</tool_call><tool_call>{"name":"wait","arguments":{}}</tool_call>',
    ],
)
def test_robodojo_rejects_ambiguous_model_actions(text):
    with pytest.raises(ValueError):
        parse_action(text)


@pytest.mark.parametrize("note", ["123", "false", '{"observation": "target"}'])
def test_robodojo_qwen_xml_executes_typed_tool_arguments(robot_env, note):
    agent = RoboDojoSessions(robot_env, image_size=32).agent
    text = (
        "Reach the target.\n<tool_call>\n<function=move_eef>\n"
        '<parameter=targets>\n{"left": [0.12, 0, 0, 0, 0, 0]}\n</parameter>\n'
        f"<parameter=note>\n{note}\n</parameter>\n</function>\n</tool_call>"
    )
    call = parse_action(text, agent.schemas())
    assert call["arguments"]["note"] == note
    _, reward, terminated, _, _ = agent.step(call, env_idx=0)
    assert reward[0] > 0 and terminated[0]


@pytest.mark.parametrize(
    "body",
    [
        "<function=done><parameter=note>Missing close",
        "<function=done><parameter=note>A</parameter><parameter=note>B</parameter></function>",
        "<function=done>Unexpected text</function>",
        "<function=done></function><function=give_up></function>",
    ],
)
def test_robodojo_rejects_incomplete_or_ambiguous_qwen_xml(body):
    with pytest.raises(ValueError):
        parse_action(f"<tool_call>{body}</tool_call>")


def test_robodojo_cancelled_acquisition_does_not_leak_a_slot(robot_env):
    async def run():
        sessions = RoboDojoSessions(robot_env, image_size=32)
        active = await sessions.start(1)
        waiting = asyncio.create_task(sessions.start(2, "waiting"))
        await asyncio.sleep(0)
        assert "waiting" in sessions.pending
        await sessions.end("waiting")
        assert waiting.cancelled()
        assert not sessions.pending and sessions.free.qsize() == 0
        await sessions.end(active["session_id"])
        await sessions.end(active["session_id"])
        assert sessions.free.qsize() == 1

    asyncio.run(run())


def test_robodojo_never_executes_a_tool_inside_reasoning():
    with pytest.raises(ValueError):
        parse_action(
            '<think><tool_call>{"name":"done","arguments":{}}</tool_call></think>'
        )
    assert (
        parse_action(
            '<think>I should inspect first.</think><tool_call>{"name":"take_pic","arguments":{"note":"Inspect."}}</tool_call>'
        )["name"]
        == "take_pic"
    )


def test_robodojo_idle_lease_expires_and_releases_slot(robot_env):
    async def run():
        sessions = RoboDojoSessions(robot_env, image_size=32, session_timeout=0.04)
        server = TestServer(create_app(sessions))
        await server.start_server()
        try:
            async with RoboDojoClient(str(server.make_url(""))) as client:
                await client.start(1)

                async def wait_for_release():
                    while sessions.sessions:
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_for_release(), timeout=2)
                assert sessions.free.qsize() == 1
        finally:
            await server.close()

    asyncio.run(run())


def test_fsdp_agent_entry_imports_without_megatron():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['megatron'] = None; "
            "from rlinf.runners.agent_runner import AgentRunner; "
            "from rlinf.agents.robodojo.actor import RoboDojoFSDPActor; "
            "from rlinf.agents.robodojo.eval_runner import RoboDojoEvalRunner",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
