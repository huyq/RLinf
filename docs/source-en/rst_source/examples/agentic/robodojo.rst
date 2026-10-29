Train a Camera-Based RoboDojo Agent
===================================

Train Qwen3.8-27B to control RoboDojo through tool calls using camera images and simulator task rewards. This example connects a simulator service, multi-turn SGLang rollout, FSDP2 updates, checkpoint export, and saved-layout evaluation.

Overview
--------

Run the LLM workers in your training environment and the simulator in its own NVIDIA environment. HTTP sessions connect them, so the trainer can retain its vendor PyTorch installation.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      Qwen3.8-27B / qwen3_5

   .. grid-item-card:: Algorithms
      :text-align: center

      Trajectory GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      RoboDojo saved layouts

   .. grid-item-card:: Hardware
      :text-align: center

      4 training devices + NVIDIA simulator

The supplied checkpoint at ``/mnt/nas/huyuanquan/models/Qwen3.8-27B`` declares ``Qwen3_5ForConditionalGeneration`` in its config. The recipe selects ``model_type: qwen3_5`` and uses its pretrained vision encoder and language decoder. Robot actions remain function calls; the model generates their names and arguments as text.

.. code-block:: text

   saved layout seeds -> agent loop -> SGLang generation
                            |                 ^
                       HTTP session           | updated weights
                            |                 |
                      RoboDojo service -> image-bearing turns -> FSDP2 GRPO

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Each generation sees the latest head and wrist images, task instruction, measured joints, EE poses, and the previous tool result. Earlier turns retain text history; earlier camera frames are omitted. The prompt builder drops the oldest text turns when needed to fit the context budget, while preserving the current images.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Contract
   * - Images
     - Head, left wrist, and right wrist when available; the service bounds their side length, then the Qwen processor prepares image patches and grids.
   * - Actions
     - One ``move_eef``, ``move_joints``, ``wait``, ``take_pic``, ``done``, or ``give_up`` call per generation. The parser accepts JSON and the checkpoint template's Qwen XML function/parameter format. See :doc:`../embodied/robodojo` for coordinates and tool arguments.
   * - Reward
     - Sum of simulator rewards within the trajectory. The default environment rewards the first task-success transition. Declaring ``done`` does not award success.
   * - Training samples
     - One generated assistant turn per sample. Prompt tokens, image placeholders, and tool feedback are masked out of the policy loss.

A single processor invocation supplies the exact token IDs, ``pixel_values``, ``image_grid_thw``, and ``mm_token_type_ids`` needed for training. Rollout receives the same pixel tensors and grids through SGLang's ``processor_output`` interface. The actor rebuilds multimodal positions for each turn and scores the generated tokens with their original images. Visual and language parameters both receive gradients.

Installation
------------

Prepare the simulator environment and assets using :doc:`../embodied/robodojo`. On a standard NVIDIA training host, install the existing agentic runtime:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

The `PAI-PPU V2.1.0 image release notes <https://help.aliyun.com/en/pai/pai-ppu-v2-1-0-official-image-release-note>`_ list the inference image ``sglang:0.5.10-xpu2.1.0`` with PPU SDK 2.1.0, EIC SDK 1.3.9, and CUDA 13.0. This image release requires driver 2.1.0 (ra1f23041713). Preserve the official image's ``aiext-pypi`` package source and matching dependency versions. The NVIDIA installation command above selects CUDA dependencies; follow the PPU vendor instructions for the training host.

The release notes scope training images to PAI-DSW/DLC and inference images to PAI-EAS/ModelGallery. This recipe launches the FSDP2 actor and SGLang engine on the same training devices and requires a validated combined PPU environment. Using separate training and inference deployments would require additional worker-placement and weight-synchronization support; the current multi-turn FSDP actor supports colocation only. The training image matrix lists PyTorch 2.9; inspect the versions in the Python environment selected for the run before resolving dependencies.

Create a venv that reuses the working vendor PyTorch stack. Select the unified vendor index explicitly: the older ``ubuntu_cu130`` index returned a standard SGLang wheel in this environment, while the unified index provides builds marked ``ppu2.1.0``. Inspect the dependency plan before installing:

.. code-block:: bash

   python -m venv --system-site-packages .venv-sglang-ppu
   source .venv-sglang-ppu/bin/activate
   python -m pip show torch transformers
   python -m pip install --dry-run --no-build-isolation \
     --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/ \
     "sglang==0.5.10+v0.3.0.ppu2.1.0"

This build accepts the installed PyTorch 2.9.0 and requires Transformers 5.3.0. Its metadata pins ``cuda-python==12.9``, whose vendor binary does not match the tested CUDA 13.0 host. The unified index provides matching Python 3.12 / PPU SDK 2.1.0 builds of ``cuda-python`` and ``cuda-bindings`` at ``13.0.0+v0.1.0.ppu2.1.0.ce``. Installing these in the venv repairs Runtime API device enumeration, which failed with the inherited 13.2 bindings. SGLang's strict 12.9 dependency declaration still conflicts with this combination. The default FlashInfer CuTe RMSNorm also fails on this host; the CUDA fallback passes with ``FLASHINFER_USE_CUDA_NORM=1``. Validate full generation before training. ``--system-site-packages`` exposes host packages, while packages installed through the venv pip stay in the venv.

The tested PPU wheel also requires ``fla==1.0.0+v0.1.0.ppu2.1.0.oe``. Its module is named ``fused_sigmoid_gating``. Apply the :download:`compatibility patch <../../../../../examples/agent/robodojo/sglang_0_5_10_ppu.patch>` to this exact SGLang build to correct the FLA imports and logging calls, and forward preprocessed image grids to M-RoPE. After installing the engine and its dependencies in the venv, run:

.. code-block:: bash

   python -m pip install --no-build-isolation \
     --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/ \
     "cuda-python==13.0.0+v0.1.0.ppu2.1.0.ce" \
     "cuda-bindings==13.0.0+v0.1.0.ppu2.1.0.ce" \
     "fla==1.0.0+v0.1.0.ppu2.1.0.oe"
   ppu_site=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
   patch -p1 -d "$ppu_site" < examples/agent/robodojo/sglang_0_5_10_ppu.patch
   export FLASHINFER_USE_CUDA_NORM=1

The patch targets the vendor wheel only. Keep ``FLASHINFER_USE_CUDA_NORM=1`` when launching rollout and training in this environment.

Upstream SGLang 0.5.10 includes the `Qwen3.5 model implementation <https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/models/qwen3_5.py>`_ and accepts preprocessed images through `Engine.async_generate <https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/entrypoints/engine.py>`_. Its version therefore provides the upstream interfaces used by this recipe. Verify the PPU build's Qwen3.5 multimodal kernels and RLinf's tokenizer/scheduler extensions for weight synchronization and memory release/resume before the first RL update.

A successful torchtitan run validates that backend; RLinf FSDP2, SGLang generation, and weight synchronization still need verification together on PPU. The CPU agent tests pass with Transformers 5.3.0 and 5.8 using a tiny real Qwen3.5 model; these test versions do not prescribe the PPU vendor build's dependencies. A TP=1, BF16 smoke run on PPU-ZW810E loaded the supplied 27B checkpoint and generated two camera-conditioned turns against the mock simulator. The formal Ray eval entry also completed a two-turn episode using real rollout and agent workers with Channel image transfer. FSDP2 updates, TP=4 placement, weight synchronization, and native Isaac Sim execution still need hardware validation.

Your training environment must expose ``Qwen3_5ForConditionalGeneration``, ``AutoProcessor``, and a SGLang engine supporting Qwen3.5 multimodal generation and preprocessed image inputs. The simulator process needs RoboDojo and its runtime, while the trainer owns model loading and optimization.

Run It
------

Start the simulator service on the NVIDIA simulator host after activating its virtual environment:

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES
   PYTHONPATH="$PWD" python examples/agent/robodojo/serve_env.py \
      --assets-root /data/robodojo --num-envs 2 --host 0.0.0.0 --port 8090

The service leases one slot per trajectory and resets it to the requested saved-layout seed. Calls on that session advance only its slot. It releases slots on normal completion and client cleanup; idle sessions expire after ``--session-timeout`` seconds, default 3600. Check ``http://simulator-host:8090/health`` to inspect available slots.

Start training from the repository root on the training host:

.. code-block:: bash

   bash examples/agent/robodojo/run_train.sh \
      model_path=/mnt/nas/huyuanquan/models/Qwen3.8-27B \
      robodojo.server_url=http://simulator-host:8090

The preset colocates a four-device FSDP2 actor with TP=4 SGLang rollout. RLinf alternates their memory use and synchronizes updated weights before each rollout batch. Each step samples four layout queries with four trajectories per query. Every trajectory within a GRPO group resets to the same layout; its final return is assigned to all its generated turns.

Use ``examples/agent/robodojo/config/qwen3_8_27b_grpo.yaml`` to select layout seeds, batch size, turn budget, and image size. The initial recipe uses BF16 model parameters, FP32 gradient reduction, one turn per micro-batch, one optimizer minibatch, temperature 1, and KL=0. Old-policy probabilities come from SGLang; ``algorithm.recompute_logprobs`` is false. Vision batches retain their image dictionaries during DP balancing. Padding reuses a real image-bearing sample with a zero loss mask so every FSDP rank traverses the visual branch.

Keep ``actor.pack_traj`` and ``runner.enable_dynamic_batch_size`` false: the text packing path does not implement the image and multimodal-position contract. Configuration validation rejects incompatible settings. Set ``robodojo.request_timeout`` to cover generation and waiting for simulator slots.

Checkpoint and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

The runner saves actor, optimizer, scheduler, and dataloader state every ten updates. Resume from a complete step directory:

.. code-block:: bash

   bash examples/agent/robodojo/run_train.sh \
      robodojo.server_url=http://simulator-host:8090 \
      runner.resume_dir=./logs/robodojo-qwen3_8-27b-grpo/checkpoints/global_step_10

Resume restores the model and layout sampler. New simulator sessions reset their layouts; in-flight episodes are not resumed.

Export the full actor weights together with the original processor and tokenizer, then evaluate the resulting HF checkpoint:

.. code-block:: bash

   PYTHONPATH="$PWD" python examples/agent/robodojo/export_checkpoint.py \
      --weights ./logs/robodojo-qwen3_8-27b-grpo/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt \
      --model-path /mnt/nas/huyuanquan/models/Qwen3.8-27B \
      --output-dir ./exports/robodojo-qwen3_8-27b
   PYTHONPATH="$PWD" python examples/agent/robodojo/eval.py \
      model_path=./exports/robodojo-qwen3_8-27b \
      robodojo.server_url=http://simulator-host:8090

Evaluation uses one greedy trajectory per saved layout. It writes per-layout success, return, turn count, and aggregate success rate to ``./logs/robodojo-qwen3_8-27b-eval/results.json``. Export requires an empty output directory and preserves the model configuration, processor, tokenizer, and generation settings.

Validation
----------

Run the CPU test to check camera rollout, task rewards, GRPO advantages, a real Qwen3.5 visual backward pass, and HF checkpoint export/reload:

.. code-block:: bash

   python -m pytest tests/unit_tests/test_agent.py tests/unit_tests/test_sim_env.py -q

The test replaces the external simulator runtime and scripts rollout actions. Tokenization, image processing, the small Qwen3.5 model, policy log-probabilities, loss, gradients, and parameter updates run normally. It checks that both visual and language weights change. The visual test skips when the installed Transformers does not provide Qwen3.5; use a compatible training environment to run it.

Before testing actor updates, use the single-device PPU preset to exercise the real Ray rollout and agent workers through their channels. Activate the PPU venv and export ``FLASHINFER_USE_CUDA_NORM=1`` before starting Ray; see :doc:`../../guides/multi_node` for environment propagation. Keep workers on the same Python environment. Start the mock simulator in one terminal:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/tests/robot_mocks:$PWD" \
      python examples/agent/robodojo/serve_env.py \
      --num-envs 1 --max-turns 2 --image-size 256 --port 18090 \
      task_config.success_target=0.1

Run the real checkpoint evaluation in another terminal using the same venv:

.. code-block:: bash

   export FLASHINFER_USE_CUDA_NORM=1
   export RLINF_NODE_RANK=0
   PYTHONPATH="$PWD" python examples/agent/robodojo/eval.py \
      --config-name qwen3_8_27b_ppu_smoke

This preset uses TP=1, BF16, a 4096-token context, one saved layout, and at most two tool turns. It forwards context and bounded weight-loading options to the SGLang worker and writes results to ``./logs/robodojo-qwen3_8-27b-ppu-smoke/results.json``. On a host with a small local filesystem, bound Ray's object store when starting the head; this host uses 256 MiB. Synthetic cameras and rewards verify the software path; task success is not a benchmark score.

To connect the actual 27B trainer to a synthetic simulator, start the service with the mock runtime on its Python path:

.. code-block:: bash

   PYTHONPATH="$PWD/tests/robot_mocks:$PWD" python examples/agent/robodojo/serve_env.py \
      --num-envs 2 task_config.success_target=0.1

This runtime supplies synthetic camera images and rewards a left-arm target reaching its test threshold. Use the normal training command with ``robodojo.server_url=http://127.0.0.1:8090``, ``runner.max_steps=1``, and ``runner.save_interval=-1`` for a short update without checkpoint output. The trainer still needs accelerator memory and its SGLang backend. This validates software integration; task learning and real RoboDojo success require the actual simulator and assets.
