Use RoboDojo Environments
=========================

Drive RoboDojo tasks with LLM function calls, read structured robot feedback, and check the interaction with a model-free smoke run. Bind the environment to an agent tool adapter; your agent loop supplies model generation and formats observations for the LLM.

For camera-conditioned LLM RL, continue with :doc:`../agentic/robodojo`.

Overview
--------

The adapter owns the simulator through a pinned ``rlinf-robodojo-runtime`` bridge. The bridge starts one isolated Isaac Sim scene per slot and uses RoboDojo's task predicates to report success.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      LLM tool agent

   .. grid-item-card:: Algorithms
      :text-align: center

      Caller-defined

   .. grid-item-card:: Tasks
      :text-align: center

      Bottle disposal example

   .. grid-item-card:: Hardware
      :text-align: center

      Linux x86_64 · NVIDIA GPU

The simulator uses Python 3.11, Torch 2.7, and a CUDA 12.x toolkit (12.6 or newer) in a separate virtual environment. Keep a PPU LLM trainer in its existing environment. A future remote tool service can connect the two; the adapter itself is a local environment interface.

Tasks
~~~~~

Start with the supplied task configuration, then select other tasks supported by the installed runtime.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Configuration
   * - ``put_bottles_into_dustbin``
     - ``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml``

Set ``task_config.task_name`` to change tasks. Use fixed layouts with ``task_config.random: False`` when comparing GRPO trajectories: the runtime's random-layout mode draws fresh scene seeds internally.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The adapter returns CPU tensors, allowing an external controller to consume observations without sharing simulator device memory.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Specification
   * - Observation
     - ``main_images``: uint8 ``[N, H, W, 3]``; ``wrist_images``: optional uint8 ``[N, cameras, H, W, 3]``; ``states``: float32 ``[N, 14]`` for ARX-X5.
   * - Action
     - ``execute`` accepts typed ``JointTarget`` joint targets, ``JointControls`` trajectories, or ``EefTarget`` poses for one slot. The agent adapter exposes ``move_joints``, ``move_eef``, ``wait``, ``take_pic``, ``done``, and ``give_up``.
   * - Reward
     - Native aggregate chunk reward, or configured task-success reward. The example uses ``reward_coef: 1.0`` and ``use_rel_reward: True`` to reward the first successful transition.
   * - Prompt
     - ``task_descriptions`` contains one scene instruction per slot.

The agent interface follows `inspect-robots' named partial-target tools <https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent>`_. ``move_joints`` takes a ``targets`` object and a short ``note``. Name only the dimensions you want to change: ``left_j0`` through ``left_j5``, ``right_j0`` through ``right_j5``, and ``left_gripper`` / ``right_gripper``. Arm targets are absolute radians; gripper targets are opening fractions in ``[0, 1]``. Unnamed dimensions hold their observed values.

The controller interpolates each move using ``controller.max_joint_step`` and ``controller.max_gripper_step`` per control, and rejects motions exceeding ``controller.max_motion_steps``. Pass ``joint_limits`` to the agent adapter for named ``[lower, upper]`` arm bounds when you have the robot's operating limits. The feedback reports requested and executed controls. Joint moves include measured target residuals; EE moves return the measured pose in the agent observation so you can check whether the target was reached.

For Cartesian control, ``move_eef`` accepts a ``targets`` object with optional ``left`` and ``right`` poses, each ``[x, y, z, yaw, pitch, roll]`` in that arm's own base frame. Positions use metres; angles use radians with rotation ``Rz(yaw) Ry(pitch) Rx(roll)``. Optional ``left_gripper`` and ``right_gripper`` values use opening fractions. The controller converts poses to world coordinates, solves all requested arms before executing, and interpolates the resulting joint targets. An IK failure returns tool feedback without advancing physics. This is joint-space interpolation toward an EE target; it does not prescribe a straight Cartesian path.

The tool adapter lives in ``rlinf/agents/robodojo/tools.py`` and owns schemas, JSON parsing, notes, hindsight, turns, and agent stop decisions. The environment owns simulator episodes, observations, rewards, and physical budgets. Its public ``execute`` method accepts ``JointTarget``, ``JointControls``, or ``EefTarget``; ``observe`` reads one native slot and ``snapshot`` returns cached batched observations. Native EEF poses use ``[x, y, z, qw, qx, qy, qz]``; the agent adapter formats them as six Cartesian/Euler values for reuse in ``move_eef``.

For existing controllers, arrays still accept ``[N, 14]`` or ``[N, C, 14]`` joint targets in native left-arm, right-arm, left-gripper, right-gripper order. ``state_action_order: aloha`` changes array/state ordering to left arm, left gripper, right arm, right gripper. Native dictionary actions use ``left_arm_joint_state``, ``right_arm_joint_state``, ``left_ee_joint_state``, and ``right_ee_joint_state`` with values shaped ``[N, width]`` or ``[N, C, width]``. Agent dimension names always refer to native joints, regardless of array ordering.

Installation
------------

Install the simulator into its own environment on an NVIDIA host:

.. code-block:: bash

   bash requirements/install.sh embodied --env robodojo --venv .robodojo-venv
   source .robodojo-venv/bin/activate

This installs the pinned runtime, Isaac Sim 5.1, cuRobo, and editable IsaacLab sources. Keep the virtual environment and its IsaacLab checkout together. The installer rejects incompatible platform, Python, and Torch selections before changing the environment.

You can also build the environment image:

.. code-block:: bash

   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-robodojo \
      --build-arg PLATFORM=nvidia -t rlinf:robodojo .

The image installs the simulator dependencies; mount scene assets when starting a container with NVIDIA GPU access.

Prepare Scene Assets
--------------------

Download the robot, object, material, and layout assets from the `RoboDojo dataset <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_. Choose a writable directory with room for downloads and generated robot configurations:

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
      --include 'Assets/**' --local-dir /data/robodojo
   export ROBODOJO_ASSETS_ROOT=/data/robodojo
   python - <<'PY'
   import os
   import runpy
   from robodojo_runtime import source_root

   script = source_root() / "utils/update_embodiment_config_path.py"
   os.chdir(os.environ["ROBODOJO_ASSETS_ROOT"])
   runpy.run_path(str(script), run_name="__main__")
   PY

``ROBODOJO_ASSETS_ROOT`` names the directory containing ``Assets/``. The script expands robot path templates; rerun it after moving the assets. You may instead set ``assets_path`` in the environment config.

The installed runtime includes its compatible RoboDojo source tree. A separate eval-only checkout, such as ``../RoboDojo``, does not provide the runtime's ``src/collect_client`` interface. Use the packaged source for this adapter; use the upstream checkout for assets and task references. NVIDIA scene resources may also be downloaded when a scene first loads.

Run It
------

Accept the Isaac Sim license as required by your installation, then run the smoke check:

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES
   PYTHONPATH="$PWD" python tests/e2e_tests/embodied/robodojo_smoke.py \
      --assets-root "$ROBODOJO_ASSETS_ROOT" --steps 3

The check loads layout 0, reads cameras and named state, alternates ``move_joints`` and ``move_eef`` while holding the current pose, ends the episode with ``done``, and closes the simulator. Exit code 0 verifies those operations; it does not measure task success. Add ``--num-envs 2`` to check slot isolation and partial reset. ``--interface joint`` exercises arrays; ``--interface eef`` exercises typed Cartesian actions directly.

For an agent loop, create the registered environment, bind its agent tools, and execute a function call:

.. code-block:: python

   from omegaconf import OmegaConf
   from rlinf.envs import get_env_cls
   from rlinf.agents.robodojo.tools import RoboDojoAgentTools

   cfg = OmegaConf.load(
       "examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml"
   )
   env = get_env_cls(cfg.env_type)(cfg, 1, 0, 1, None)
   try:
       obs, info = env.reset(env_seeds=[0])
       agent = RoboDojoAgentTools(env)
       tools = agent.schemas()
       agent_observation = agent.observe(env_idx=0)
       obs, reward, terminated, truncated, info = agent.step({
           "name": "move_joints",
           "arguments": {
               "targets": {"left_j0": 0.1, "left_gripper": 1.0},
               "note": "Approach with the left arm and open its gripper.",
           },
       }, env_idx=0)
       tool_feedback = info["agent"]
   finally:
       env.close()

Pass ``tools`` to your LLM rollout and ``agent_observation`` as structured context. ``agent.observe`` exposes the instruction, named joints, available camera names, success flag, and remaining control/turn budgets. The returned ``obs`` contains image tensors for a vision-capable model or a separate perception formatter. Serialize ``tool_feedback`` as the tool response and use the selected slot's reward for training.

``agent.step`` accepts both ``{name, arguments}`` and an OpenAI ``{type: function, function: {name, arguments}}`` envelope; ``arguments`` can be a dictionary or JSON string. An existing RLinf tool worker can call ``agent.call(request.name, request.arguments, env_idx=0)`` directly.

``take_pic`` refreshes observations without advancing physics. ``wait`` holds the current pose for its requested ``steps``. ``done`` requires a ``summary`` and ``give_up`` requires a ``reason``; both accept optional ``hindsight`` and end the episode without manufacturing a success reward. Read success from RoboDojo's task predicate in the observation. Argument errors return ``ok: False`` and an error in ``info["agent"]``; simulator faults raise to the caller.

Pass ``{"max_turns": 64, "joint_limits": {}}`` as the optional second argument to ``RoboDojoAgentTools`` to configure the agent. Each tool call, including an invalid one, consumes a turn. Only executed controls consume the agent's physical step budget. The adapter requires an explicit environment reset after an agent stop or turn limit; the environment tracks a reset generation per slot so the adapter clears only that slot’s state. ``done`` and ``give_up`` do not change the simulator termination flags. Agent execution uses explicit resets even when the array interface uses ``auto_reset`` or ``ignore_terminations``. One slot's call never advances the other scenes.

``reset`` returns observations for all slots. For a partial reset, pass ``env_idx=[0]`` and a full, globally indexed ``env_seeds`` list; other slots retain their state. ``group_size`` repeats each sampled reset seed across a group. Keep those seeds fixed while collecting the trajectories you want to compare.

For array controllers, ``step`` returns aggregate chunk results. ``chunk_step`` returns one final observation and info, with rewards and terminal flags in the last column of ``[N, C]`` tensors. ``info["valid_action_mask"]`` marks physically executed controls. Array calls count submitted rollout positions in ``elapsed_steps``, including padding after a slot ends; agent calls count actual execution. Keep ``max_episode_steps`` and ``task_config.max_episode_steps`` equal.

For array calls with ``auto_reset: True``, completed slots reset and their previous observations, metrics, and done masks remain in ``final_observation``, ``final_info``, ``_final_observation``, and ``_final_info``. Keep ``enable_offload: False``: ``close`` and ``offload`` permanently release the simulator and are idempotent.

Validation
----------

Run CPU tests without Isaac Sim:

.. code-block:: bash

   python -m pytest tests/unit_tests/test_sim_env.py -q

These exercise the adapter against a fake external runtime. The real simulator smoke job is enabled when the repository variable ``ROBODOJO_ASSETS_ROOT`` points to prepared assets on the NVIDIA ``embodied`` runner. Treat simulator startup, task completion, and LLM learning as separate validation stages.
