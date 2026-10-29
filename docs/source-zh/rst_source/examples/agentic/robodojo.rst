使用相机图像训练 RoboDojo Agent
===============================

使用相机图像和仿真任务奖励，训练 Qwen3.8-27B 通过工具调用控制 RoboDojo。本例串联仿真服务、多轮 SGLang rollout、FSDP2 更新、checkpoint 导出，以及基于固定布局的评估。

概览
----

在训练环境中运行 LLM worker，在独立的 NVIDIA 环境中运行仿真器。两端通过 HTTP 会话连接，训练端可以保留厂商提供的 PyTorch 环境。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3.8-27B / qwen3_5

   .. grid-item-card:: 算法
      :text-align: center

      Trajectory GRPO

   .. grid-item-card:: 任务
      :text-align: center

      RoboDojo 固定布局

   .. grid-item-card:: 硬件
      :text-align: center

      4 张训练卡 + NVIDIA 仿真卡

``/mnt/nas/huyuanquan/models/Qwen3.8-27B`` 中的 checkpoint 声明其架构为 ``Qwen3_5ForConditionalGeneration``。配置因此选择 ``model_type: qwen3_5``，使用已有的视觉编码器和语言解码器。模型以文本形式生成工具名和参数，再由工具执行机器人动作。

.. code-block:: text

   saved layout seeds -> agent loop -> SGLang generation
                            |                 ^
                       HTTP session           | updated weights
                            |                 |
                      RoboDojo service -> image-bearing turns -> FSDP2 GRPO

观测与动作
~~~~~~~~~~

每轮生成都使用最新的头部和腕部图像、任务指令、实测关节、EE 位姿以及上一次工具反馈。历史轮次保留文本，不保留旧相机帧。上下文超出预算时，prompt 构造器删除最早的文本轮次，并保留当前图像。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 契约
   * - 图像
     - 使用可用的头部、左腕和右腕相机；服务先限制图像边长，再由 Qwen processor 生成图像 patch 和网格。
   * - 动作
     - 每轮生成一个 ``move_eef``、``move_joints``、``wait``、``take_pic``、``done`` 或 ``give_up`` 调用。解析器支持 JSON，以及 checkpoint 模板使用的 Qwen XML function/parameter 格式。坐标和参数定义见 :doc:`../embodied/robodojo`。
   * - 奖励
     - 累加 trajectory 内的仿真奖励。默认环境对首次任务成功给奖，声明 ``done`` 不会产生成功奖励。
   * - 训练样本
     - 每条样本对应一个 assistant 生成轮次。prompt token、图像占位符和工具反馈均不计入 policy loss。

一次 processor 调用同时生成精确的 token ID、``pixel_values``、``image_grid_thw`` 和 ``mm_token_type_ids``。rollout 通过 SGLang 的 ``processor_output`` 接口接收相同的像素张量和网格。actor 按轮次重新计算多模态位置编码，并使用采样时的图像计算生成 token 的概率。视觉和语言参数都会获得梯度。

安装依赖
--------

按 :doc:`../embodied/robodojo` 准备仿真环境和资产。标准 NVIDIA 训练机器可以安装现有的 agentic 环境：

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang

`PAI-PPU V2.1.0 镜像发布说明 <https://help.aliyun.com/en/pai/pai-ppu-v2-1-0-official-image-release-note>`_ 明确列出推理镜像 ``sglang:0.5.10-xpu2.1.0``，配套 PPU SDK 2.1.0、EIC SDK 1.3.9 和 CUDA 13.0。这版镜像要求驱动 2.1.0（ra1f23041713）。请保留官方镜像的 ``aiext-pypi`` 源和配套依赖版本。上面的 NVIDIA 安装命令会选择 CUDA 依赖；PPU 训练机器应按厂商说明安装。

发布说明将训练镜像限定于 PAI-DSW/DLC，将推理镜像限定于 PAI-EAS/ModelGallery。本例在同一组训练设备上运行 FSDP2 actor 和 SGLang engine，需要验证二者共存的 PPU 环境。若将训练和推理分别部署，还需适配 worker placement 与权重同步；当前多轮 FSDP actor 仅支持共置。训练镜像矩阵列出 PyTorch 2.9，解析依赖前应检查选定 Python 环境中的实际版本。

创建 venv 并复用已验证的厂商 PyTorch 栈。显式选择厂商统一源：本机原有的 ``ubuntu_cu130`` 源返回普通 SGLang wheel，而统一源提供带 ``ppu2.1.0`` 标记的适配版。先检查依赖安装计划：

.. code-block:: bash

   python -m venv --system-site-packages .venv-sglang-ppu
   source .venv-sglang-ppu/bin/activate
   python -m pip show torch transformers
   python -m pip install --dry-run --no-build-isolation \
     --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/ \
     "sglang==0.5.10+v0.3.0.ppu2.1.0"

该构建允许保留本机 PyTorch 2.9.0，并要求 Transformers 5.3.0。其元数据固定 ``cuda-python==12.9``，对应厂商二进制包与本次 CUDA 13.0 主机不匹配。统一源提供了适配 Python 3.12 / PPU SDK 2.1.0 的 ``cuda-python`` 和 ``cuda-bindings``，版本均为 ``13.0.0+v0.1.0.ppu2.1.0.ce``。在 venv 中安装这两个包后，Runtime API 设备枚举检查通过；继承的 13.2 绑定在该检查中失败。SGLang 固定 12.9 的依赖声明仍与此组合冲突。本机默认 FlashInfer CuTe RMSNorm 也会失败，设置 ``FLASHINFER_USE_CUDA_NORM=1`` 后 CUDA fallback 检查通过。训练前还需验证完整生成。``--system-site-packages`` 使 venv 可以读取系统包，通过 venv pip 安装的包仍写入 venv。

本次 PPU wheel 还需要 ``fla==1.0.0+v0.1.0.ppu2.1.0.oe``，其模块名为 ``fused_sigmoid_gating``。对此精确版本的 SGLang 应用 :download:`兼容补丁 <../../../../../examples/agent/robodojo/sglang_0_5_10_ppu.patch>`，修正 FLA 导入、日志调用，以及预处理图像网格到 M-RoPE 的传递。在 venv 中安装 engine 及其依赖后运行：

.. code-block:: bash

   python -m pip install --no-build-isolation \
     --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/ \
     "cuda-python==13.0.0+v0.1.0.ppu2.1.0.ce" \
     "cuda-bindings==13.0.0+v0.1.0.ppu2.1.0.ce" \
     "fla==1.0.0+v0.1.0.ppu2.1.0.oe"
   ppu_site=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
   patch -p1 -d "$ppu_site" < examples/agent/robodojo/sglang_0_5_10_ppu.patch
   export FLASHINFER_USE_CUDA_NORM=1

该补丁仅针对上述厂商 wheel。在此环境中启动 rollout 和训练时，保持 ``FLASHINFER_USE_CUDA_NORM=1``。

上游 SGLang 0.5.10 已包含 `Qwen3.5 模型实现 <https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/models/qwen3_5.py>`_，其 `Engine.async_generate <https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/entrypoints/engine.py>`_ 支持预处理图像输入，因此具备本例使用的上游接口。首次 RL 更新前，还需验证 PPU 构建中的 Qwen3.5 多模态算子，以及 RLinf 注入的 tokenizer/scheduler 扩展，包括权重同步、显存释放和恢复。

torchtitan 上的成功运行验证的是该后端；RLinf FSDP2、SGLang 生成和权重同步仍需在 PPU 上联跑验证。CPU agent 测试已分别在 Transformers 5.3.0 和 5.8 上使用真实的小型 Qwen3.5 模型通过，这些测试版本不作为覆盖 PPU 厂商依赖的依据。已在 PPU-ZW810E 上以 TP=1、BF16 加载该 27B checkpoint，并针对 mock 仿真环境生成两轮相机条件响应。正式 Ray eval 入口也已通过真实 rollout、agent Worker 及 Channel 图像传输完成两轮 episode。FSDP2 更新、TP=4 placement、权重同步和真实 Isaac Sim 执行仍需硬件验证。

训练环境需要提供 ``Qwen3_5ForConditionalGeneration``、``AutoProcessor``，以及支持 Qwen3.5 多模态生成和预处理图像输入的 SGLang engine。仿真进程负责加载 RoboDojo runtime，训练进程负责模型和优化器。

开始运行
--------

在 NVIDIA 仿真机器中激活对应虚拟环境，再启动服务：

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES
   PYTHONPATH="$PWD" python examples/agent/robodojo/serve_env.py \
      --assets-root /data/robodojo --num-envs 2 --host 0.0.0.0 --port 8090

服务为每条 trajectory 分配一个 slot，并按请求的固定布局 seed reset。会话中的动作只推进该 slot。正常结束或客户端清理时会释放 slot；空闲会话在 ``--session-timeout`` 指定的秒数后到期，默认 3600 秒。可以通过 ``http://simulator-host:8090/health`` 查看可用 slot。

在训练机器的仓库根目录启动训练：

.. code-block:: bash

   bash examples/agent/robodojo/run_train.sh \
      model_path=/mnt/nas/huyuanquan/models/Qwen3.8-27B \
      robodojo.server_url=http://simulator-host:8090

默认配置将四卡 FSDP2 actor 与 TP=4 的 SGLang rollout 放在相同训练设备上。RLinf 交替使用二者的显存，并在每次 rollout 前同步更新后的权重。每个 step 采样四个布局 query，每个 query 生成四条 trajectory。同一个 GRPO group 内的 trajectory 使用相同布局；trajectory 的最终 return 会赋给其每个生成轮次。

通过 ``examples/agent/robodojo/config/qwen3_8_27b_grpo.yaml`` 配置布局 seed、batch 大小、轮数和图像大小。初始配置使用 BF16 模型参数、FP32 梯度归约、每个 micro-batch 一个轮次、一个优化器 minibatch、temperature 1 和 KL=0。旧 policy 的概率由 SGLang 返回，``algorithm.recompute_logprobs`` 为 false。DP 均衡时，图像字典始终与对应轮次一起移动。补齐样本复用真实图像输入并将 loss mask 置零，确保每个 FSDP rank 都经过视觉分支。

保持 ``actor.pack_traj`` 和 ``runner.enable_dynamic_batch_size`` 为 false；文本打包路径没有实现图像和多模态位置编码的契约。配置检查会拒绝不兼容的设置。``robodojo.request_timeout`` 应覆盖生成时间和等待仿真 slot 的时间。

保存与评估
~~~~~~~~~~

runner 每十次更新保存 actor、优化器、scheduler 和 dataloader 状态。可以从完整的 step 目录恢复：

.. code-block:: bash

   bash examples/agent/robodojo/run_train.sh \
      robodojo.server_url=http://simulator-host:8090 \
      runner.resume_dir=./logs/robodojo-qwen3_8-27b-grpo/checkpoints/global_step_10

恢复时会载入模型和布局采样进度。新的仿真会话重新 reset 布局，不恢复中途的 episode。

将完整 actor 权重与原始 processor、tokenizer 一起导出为 HF checkpoint，再进行评估：

.. code-block:: bash

   PYTHONPATH="$PWD" python examples/agent/robodojo/export_checkpoint.py \
      --weights ./logs/robodojo-qwen3_8-27b-grpo/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt \
      --model-path /mnt/nas/huyuanquan/models/Qwen3.8-27B \
      --output-dir ./exports/robodojo-qwen3_8-27b
   PYTHONPATH="$PWD" python examples/agent/robodojo/eval.py \
      model_path=./exports/robodojo-qwen3_8-27b \
      robodojo.server_url=http://simulator-host:8090

评估对每个固定布局执行一条 greedy trajectory，将各布局的成功标记、return、轮数和整体成功率写入 ``./logs/robodojo-qwen3_8-27b-eval/results.json``。导出要求输出目录为空，并保留模型配置、processor、tokenizer 和 generation 设置。

验证
----

运行 CPU 测试，检查相机 rollout、任务奖励、GRPO advantage、真实 Qwen3.5 视觉反向传播，以及 HF checkpoint 导出和重新加载：

.. code-block:: bash

   python -m pytest tests/unit_tests/test_agent.py tests/unit_tests/test_sim_env.py -q

测试替换外部仿真 runtime，并预设 rollout 动作。tokenizer、图像 processor、小型 Qwen3.5 模型、policy log-prob、loss、梯度和参数更新均实际执行，同时检查视觉与语言权重发生变化。若 Transformers 未提供 Qwen3.5，视觉测试会跳过；请在兼容的训练环境中运行该测试。

测试 actor 更新前，使用单卡 PPU 配置检查真实 Ray rollout、agent Worker 及其 Channel 通信。激活 PPU venv，在启动 Ray 前导出 ``FLASHINFER_USE_CUDA_NORM=1``，环境传递规则见 :doc:`../../guides/multi_node`。Worker 应使用相同的 Python 环境。先在一个终端中启动 mock 仿真服务：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/tests/robot_mocks:$PWD" \
      python examples/agent/robodojo/serve_env.py \
      --num-envs 1 --max-turns 2 --image-size 256 --port 18090 \
      task_config.success_target=0.1

在另一个使用相同 venv 的终端中执行真实 checkpoint 评估：

.. code-block:: bash

   export FLASHINFER_USE_CUDA_NORM=1
   export RLINF_NODE_RANK=0
   PYTHONPATH="$PWD" python examples/agent/robodojo/eval.py \
      --config-name qwen3_8_27b_ppu_smoke

该配置使用 TP=1、BF16、4096-token 上下文、一个固定布局和最多两轮工具调用，并将上下文及有界权重读取参数传给 SGLang Worker。结果写入 ``./logs/robodojo-qwen3_8-27b-ppu-smoke/results.json``。若主机本地文件系统空间较小，应在启动 Ray head 时限制 object store；本次主机设为 256 MiB。合成相机与奖励用于验证软件链路，任务成功率不作为 benchmark 分数。

若要让实际的 27B trainer 连接合成仿真环境，将 mock runtime 放到服务端的 Python 搜索路径：

.. code-block:: bash

   PYTHONPATH="$PWD/tests/robot_mocks:$PWD" python examples/agent/robodojo/serve_env.py \
      --num-envs 2 task_config.success_target=0.1

该 runtime 返回合成相机图像，并在左臂目标达到测试阈值时给奖。沿用训练命令，设置 ``robodojo.server_url=http://127.0.0.1:8090``、``runner.max_steps=1`` 和 ``runner.save_interval=-1``，可执行一次不保存 checkpoint 的短更新。trainer 仍需要相应的加速卡显存和 SGLang 后端。这一步验证软件链路；任务学习效果和真实 RoboDojo 成功率需要实际仿真器与资产验证。
