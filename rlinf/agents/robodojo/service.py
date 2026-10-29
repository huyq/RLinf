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
"""HTTP sessions that keep simulator ownership separate from LLM workers."""

import asyncio
import base64
import io
import math
from types import TracebackType
from uuid import uuid4

import aiohttp
from aiohttp import web
from PIL import Image

from rlinf.agents.robodojo.tools import RoboDojoAgentTools
from rlinf.envs.sim.robodojo.robodojo_env import RoboDojoEnv


class RoboDojoSessions:
    """Lease isolated simulator slots to trajectories and reset each saved layout.

    The caller owns env and must invoke service operations on its owning thread.
    One session serializes its actions; sessions never share a vector slot.
    """

    def __init__(
        self,
        env: RoboDojoEnv,
        max_turns: int = 16,
        image_size: int = 256,
        session_timeout: float = 3600,
    ) -> None:
        if (
            max_turns < 1
            or image_size < 32
            or not math.isfinite(session_timeout)
            or session_timeout <= 0
        ):
            raise ValueError("Expected positive max_turns and image_size >= 32.")
        self.env = env
        self.agent = RoboDojoAgentTools(env, {"max_turns": max_turns})
        self.image_size = image_size
        self.free = asyncio.Queue()
        for slot in range(env.num_envs):
            self.free.put_nowait(slot)
        self.sessions = {}
        self.pending = {}
        self.session_timeout = session_timeout
        self.deadlines = {}

    def _observation(self, slot: int) -> dict:
        raw = self.env.observe(slot)
        images = []
        for key, camera in (
            ("full_image", "head"),
            ("left_wrist_image", "left_wrist"),
            ("right_wrist_image", "right_wrist"),
        ):
            if raw.get(key) is None:
                continue
            image = Image.fromarray(raw[key]).convert("RGB")
            image.thumbnail((self.image_size, self.image_size))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images.append(
                {
                    "camera": camera,
                    "data_url": "data:image/png;base64,"
                    + base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
            )
        if not images:
            raise RuntimeError(
                "RoboDojo requires at least one camera image for VLM training."
            )
        return {"observation": self.agent.observe(slot), "images": images}

    async def start(self, seed: int, session_id: str | None = None) -> dict:
        """Acquire a slot and reset it to the requested layout; cancellation releases it."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f"Expected a nonnegative integer layout seed, got {seed!r}."
            )
        session_id = session_id or uuid4().hex
        if session_id in self.sessions or session_id in self.pending:
            raise ValueError("Expected a new simulator session ID.")
        self.pending[session_id] = asyncio.current_task()
        slot = None
        try:
            slot = await self.free.get()
            seeds = self.env.reset_state_ids.tolist()
            seeds[slot] = seed
            self.env.reset(env_idx=slot, env_seeds=seeds)
            result = self._observation(slot)
            self.sessions[session_id] = (slot, asyncio.Lock())
            self.deadlines[session_id] = (
                asyncio.get_running_loop().time() + self.session_timeout
            )
            return {**result, "session_id": session_id, "tools": self.agent.schemas()}
        except BaseException:
            if slot is not None:
                self.free.put_nowait(slot)
            raise
        finally:
            self.pending.pop(session_id, None)

    async def step(self, session_id: str, call: dict) -> dict:
        """Execute one tool call and return the next images, native reward, and feedback."""
        slot, lock = self.sessions[session_id]
        async with lock:
            if session_id not in self.sessions:
                raise KeyError(session_id)
            self.deadlines[session_id] = (
                asyncio.get_running_loop().time() + self.session_timeout
            )
            _, reward, terminated, truncated, info = self.agent.step(call, env_idx=slot)
            return {
                **self._observation(slot),
                "reward": float(reward[slot]),
                "terminated": bool(terminated[slot]),
                "truncated": bool(truncated[slot]),
                "feedback": info["agent"],
            }

    async def end(self, session_id: str) -> None:
        """Release a session once; ending an already released session is harmless."""
        pending = self.pending.get(session_id)
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        entry = self.sessions.get(session_id)
        if entry is not None:
            slot, lock = entry
            async with lock:
                if self.sessions.pop(session_id, None) is not None:
                    self.deadlines.pop(session_id, None)
                    self.free.put_nowait(slot)


def create_app(sessions: RoboDojoSessions) -> web.Application:
    """Build the simulator HTTP application; the caller owns the environment."""
    app = web.Application(client_max_size=8 * 1024 * 1024)

    async def start(request):
        body = await request.json()
        return web.json_response(
            await sessions.start(body["seed"], body.get("session_id"))
        )

    async def step(request):
        try:
            result = await sessions.step(
                request.match_info["session_id"], await request.json()
            )
        except KeyError as error:
            raise web.HTTPNotFound(text="Unknown simulator session.") from error
        return web.json_response(result)

    async def end(request):
        await sessions.end(request.match_info["session_id"])
        return web.json_response({"released": True})

    async def health(request):
        return web.json_response(
            {"slots": sessions.env.num_envs, "free_slots": sessions.free.qsize()}
        )

    async def expire_leases(app):
        async def sweep():
            while True:
                await asyncio.sleep(min(30, sessions.session_timeout / 2))
                now = asyncio.get_running_loop().time()
                for session_id, deadline in list(sessions.deadlines.items()):
                    if deadline <= now:
                        await sessions.end(session_id)

        task = asyncio.create_task(sweep())
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for session_id in list(sessions.pending):
            await sessions.end(session_id)
        for session_id in list(sessions.sessions):
            await sessions.end(session_id)

    app.cleanup_ctx.append(expire_leases)
    app.router.add_post("/sessions", start)
    app.router.add_post("/sessions/{session_id}/step", step)
    app.router.add_delete("/sessions/{session_id}", end)
    app.router.add_get("/health", health)
    return app


class RoboDojoClient:
    """Async simulator client; physical actions are never retried after a timeout."""

    def __init__(self, base_url: str, timeout: float = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.http = None
        self.session_id = None

    async def __aenter__(self) -> "RoboDojoClient":
        self.http = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self.session_id is not None:
                try:
                    await asyncio.shield(self.close())
                except Exception:
                    if exc is None:
                        raise
                    from rlinf.utils.logging import get_logger

                    get_logger().warning(
                        "Failed to release the simulator session after a rollout error.",
                        exc_info=True,
                    )
        finally:
            await self.http.close()

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        async with self.http.request(
            method, self.base_url + path, json=body
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def start(self, seed: int) -> dict:
        """Start one saved-layout episode."""
        if self.session_id is not None:
            raise RuntimeError(
                "Close the current simulator session before starting another."
            )
        self.session_id = uuid4().hex
        result = await self._request(
            "POST", "/sessions", {"seed": seed, "session_id": self.session_id}
        )
        return result

    async def step(self, call: dict) -> dict:
        """Execute a tool call on the owned episode."""
        if self.session_id is None:
            raise RuntimeError("Start a simulator session before executing a tool.")
        return await self._request("POST", f"/sessions/{self.session_id}/step", call)

    async def close(self) -> None:
        """Release this episode's simulator slot."""
        if self.session_id is not None:
            await self._request("DELETE", f"/sessions/{self.session_id}")
            self.session_id = None
