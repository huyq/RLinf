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
"""Camera-conditioned LLM trajectories for the multi-turn GRPO actor."""

import base64
import io
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from rlinf.agents.robodojo.service import RoboDojoClient
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.workers.agent.agent_loop import AgentLoopOutput, MultiAgentLoopOutput


@dataclass(frozen=True)
class VisionPrompt:
    """Exact rollout token IDs and the matching actor-side processed camera inputs."""

    input_ids: list[int]
    image_data: list[dict[str, Any]]
    model_inputs: dict[str, torch.Tensor]


def prepare_prompt(
    processor: Any, messages: list[dict], tools: list[dict]
) -> VisionPrompt:
    """Render one chat using the checkpoint processor for both engines."""
    image_urls = [
        part["image"]
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image"
    ]
    images = [
        Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        for url in image_urls
    ]
    text = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(text=[text], images=images, return_tensors="pt")
    model_inputs = {
        key: encoded[key].detach().cpu()
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
        if key in encoded
    }
    if not {"pixel_values", "image_grid_thw", "mm_token_type_ids"}.issubset(
        model_inputs
    ):
        raise ValueError(
            "Expected a Qwen image processor with pixel_values, image_grid_thw, and mm_token_type_ids."
        )
    return VisionPrompt(
        encoded["input_ids"][0].tolist(),
        [
            {
                "format": "processor_output",
                **{
                    key: value.clone()
                    for key, value in model_inputs.items()
                    if key != "mm_token_type_ids"
                },
            }
        ],
        model_inputs,
    )


def _parse_xml_action(body: str, tools: list[dict]) -> dict:
    """Decode Qwen XML parameters using the advertised function schemas."""
    function = re.fullmatch(
        r"<function=([^\s<>]+)>\s*(.*?)\s*</function>", body, flags=re.DOTALL
    )
    if function is None:
        raise ValueError("Expected one complete Qwen function block.")
    name, parameters = function.groups()
    properties = next(
        (
            tool["function"]["parameters"]["properties"]
            for tool in tools
            if tool["function"]["name"] == name
        ),
        {},
    )
    arguments = {}
    end = 0
    for parameter in re.finditer(
        r"<parameter=([^\s<>]+)>\s*(.*?)\s*</parameter>",
        parameters,
        flags=re.DOTALL,
    ):
        key, raw = parameter.groups()
        if parameters[end : parameter.start()].strip() or key in arguments:
            raise ValueError("Expected distinct Qwen parameters without extra content.")
        if properties.get(key, {}).get("type") == "string":
            arguments[key] = raw
        else:
            try:
                arguments[key] = json.loads(raw)
            except json.JSONDecodeError:
                if key in properties:
                    raise ValueError(f"Expected JSON for Qwen parameter {key!r}.")
                arguments[key] = raw
        end = parameter.end()
    if parameters[end:].strip():
        raise ValueError("Expected complete Qwen parameter blocks.")
    return {"name": name, "arguments": arguments}


def parse_action(text: str, tools: list[dict] | None = None) -> dict:
    """Parse one JSON or Qwen XML call; schemas preserve XML string parameters."""
    if "<think>" in text:
        if "</think>" not in text:
            raise ValueError(
                "Expected a completed response outside the reasoning block."
            )
        text = text.split("</think>", 1)[1]
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    if len(matches) > 1:
        raise ValueError("Submit exactly one robot tool call per turn.")
    body = (matches[0] if matches else text).strip()
    value = (
        _parse_xml_action(body, tools or [])
        if body.startswith("<function=")
        else json.loads(body)
    )
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise ValueError("Expected a tool call object with name and arguments.")
    if set(value) - {"name", "arguments"} or "arguments" not in value:
        raise ValueError("Expected only name and arguments in a robot tool call.")
    return value


async def run_episode(
    client: RoboDojoClient,
    processor: Any,
    generate: Callable[..., Awaitable[dict]],
    *,
    seed: int,
    max_turns: int,
    max_prompt_length: int,
    seq_length: int,
    max_new_tokens: int,
    return_logprobs: bool = True,
) -> MultiAgentLoopOutput:
    """Collect one image-conditioned trajectory and broadcast its return to its turns.

    Prior turns retain text history; only the current camera frames are supplied.
    Every actor sample stores the exact prompt, generated tokens, and image grids
    used by rollout. Environment feedback is context and never a loss target.
    The caller manages the client context, which releases its session on failure.
    """
    state = await client.start(seed)
    tools = state["tools"]
    history = []
    outputs = []
    total_reward = 0.0
    system = {
        "role": "system",
        "content": "Control the robot to complete the task using its camera observations and robot state. Submit exactly one tool call per turn. Use move_eef for Cartesian targets in each arm's base frame, or move_joints for joint targets. Include the required note. Only simulator success determines reward. Use done when finished or give_up when unable to proceed.",
    }
    for turn in range(max_turns):
        content = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "observation": state["observation"],
                        "previous_tool_result": state.get("feedback"),
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        for image in state["images"]:
            content.extend(
                [
                    {"type": "text", "text": "Camera: " + image["camera"]},
                    {"type": "image", "image": image["data_url"]},
                ]
            )
        current = {"role": "user", "content": content}
        while True:
            prompt = prepare_prompt(processor, [system, *history, current], tools)
            if len(prompt.input_ids) <= max_prompt_length:
                break
            if not history:
                raise ValueError(
                    f"Current camera prompt has {len(prompt.input_ids)} tokens, exceeding max_prompt_length={max_prompt_length}; reduce image_size or increase the context budget."
                )
            del history[:2]
        remaining = min(max_new_tokens, seq_length - len(prompt.input_ids))
        if remaining < 1:
            raise ValueError(
                "Expected room for generated response tokens after the camera prompt."
            )
        result = await generate(
            prompt.input_ids,
            image_data=prompt.image_data,
            sampling_params={"max_new_tokens": remaining},
        )
        response_ids = result["output_ids"]
        if not response_ids or len(response_ids) > remaining:
            raise RuntimeError(
                f"Expected 1..{remaining} generated tokens, got {len(response_ids)}."
            )
        logprobs = result.get("logprobs")
        if return_logprobs and (logprobs is None or len(logprobs) != len(response_ids)):
            raise RuntimeError(
                "Expected one rollout log probability per generated token."
            )
        text = processor.tokenizer.decode(response_ids, skip_special_tokens=False)
        # EOS is retained in training IDs but does not belong to tool-call JSON.
        parse_text = text
        for special in (processor.tokenizer.eos_token, processor.tokenizer.pad_token):
            if special:
                parse_text = parse_text.removesuffix(special).strip()
        try:
            call = parse_action(parse_text, tools)
        except (ValueError, TypeError):
            call = {"name": "", "arguments": {}}
        state = await client.step(call)
        total_reward += state["reward"]
        ended = state["terminated"] or state["truncated"] or turn + 1 == max_turns
        outputs.append(
            AgentLoopOutput(
                prompt_ids=prompt.input_ids,
                response_ids=response_ids,
                response_logprobs=logprobs if return_logprobs else None,
                is_end=ended,
                extra_fields={"multi_modal_inputs": prompt.model_inputs},
            )
        )
        history.extend(
            [
                {"role": "user", "content": content[0]["text"]},
                {"role": "assistant", "content": parse_text},
            ]
        )
        if ended:
            break
    for output in outputs:
        output.reward_score = total_reward
    return MultiAgentLoopOutput(
        single_turn_outputs=outputs,
        extra_fields={
            "task_return": total_reward,
            "success": bool(state["observation"]["success"]),
        },
    )


def pack_episodes(episodes: list[MultiAgentLoopOutput]) -> DynamicRolloutResult:
    """Build actor-ready grouped turns without retokenizing or losing camera tensors."""
    turns = [
        (index, turn)
        for index, episode in enumerate(episodes)
        for turn in episode.single_turn_outputs
    ]
    if not episodes or any(not episode.single_turn_outputs for episode in episodes):
        raise ValueError(
            "Expected at least one generated turn in every GRPO trajectory."
        )
    return DynamicRolloutResult(
        num_sequence=len(turns),
        group_size=len(episodes),
        idx_to_traj=[index for index, _ in turns],
        input_ids=[turn.prompt_ids + turn.response_ids for _, turn in turns],
        prompt_lengths=[len(turn.prompt_ids) for _, turn in turns],
        response_lengths=[len(turn.response_ids) for _, turn in turns],
        rollout_logprobs=[turn.response_logprobs for _, turn in turns]
        if all(turn.response_logprobs is not None for _, turn in turns)
        else None,
        is_end=[turn.is_end for _, turn in turns],
        rewards=[turn.reward_score for _, turn in turns],
        multi_modal_inputs=[
            turn.extra_fields["multi_modal_inputs"] for _, turn in turns
        ],
        extra_fields_traj={
            key: [episode.extra_fields[key] for episode in episodes]
            for key in ("task_return", "success")
        },
    )
