使用 RoboDojo 环境
==================

通过 LLM 函数调用控制 RoboDojo 任务，读取结构化机器人反馈，并用不加载模型的 smoke 测试检查交互链路。将环境绑定到 agent 工具适配器，由 agent loop 负责模型生成，以及面向 LLM 的观测转换。

概览
----

适配层通过固定版本的 ``rlinf-robodojo-runtime`` 管理仿真器。runtime 为每个 slot 启动独立的 Isaac Sim 场景，并用 RoboDojo 原生任务条件判断成功。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      LLM 工具 agent

   .. grid-item-card:: 算法
      :text-align: center

      调用方选择

   .. grid-item-card:: 任务
      :text-align: center

      将瓶子放入垃圾桶

   .. grid-item-card:: 硬件
      :text-align: center

      Linux x86_64 · NVIDIA GPU

仿真器使用独立虚拟环境中的 Python 3.11、Torch 2.7 和 CUDA 12.x toolkit（至少 12.6 版）。PPU 上的 LLM trainer 保留现有环境。后续可以通过远程工具服务连接二者；当前适配层提供本地环境接口。

任务
~~~~

先使用提供的任务配置，再选择已安装 runtime 支持的其他任务。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 任务
     - 配置
   * - ``put_bottles_into_dustbin``
     - ``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml``

通过 ``task_config.task_name`` 切换任务。比较 GRPO 轨迹时，使用 ``task_config.random: False`` 固定布局；runtime 的随机布局模式会在内部重新抽取场景 seed。

观测与动作
~~~~~~~~~~

适配层返回 CPU tensor，外部控制器可以读取观测，无需共享仿真器设备内存。

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 说明
   * - 观测
     - ``main_images`` 为 uint8 ``[N, H, W, 3]``；可选的 ``wrist_images`` 为 uint8 ``[N, cameras, H, W, 3]``；ARX-X5 的 ``states`` 为 float32 ``[N, 14]``。
   * - 动作
     - ``execute`` 接收单个 slot 的类型化 ``JointTarget`` 关节目标、``JointControls`` 轨迹或 ``EefTarget`` 位姿；agent 适配器提供 ``move_joints``、``move_eef``、``wait``、``take_pic``、``done`` 和 ``give_up`` 工具。
   * - 奖励
     - 原生 chunk 总奖励，或配置的任务成功奖励。示例使用 ``reward_coef: 1.0`` 和 ``use_rel_reward: True``，奖励首次成功的状态转移。
   * - 指令
     - ``task_descriptions`` 为每个 slot 提供一条场景指令。

agent 接口参考 `inspect-robots 的具名部分目标工具 <https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent>`_。``move_joints`` 接收 ``targets`` 对象和简短的 ``note``。只需指定要修改的维度：``left_j0`` 至 ``left_j5``、``right_j0`` 至 ``right_j5``，以及 ``left_gripper`` / ``right_gripper``。手臂目标为绝对关节角，单位为弧度；夹爪目标为 ``[0, 1]`` 范围的张开比例。未指定的维度保持当前观测值。

控制器按 ``controller.max_joint_step`` 和 ``controller.max_gripper_step`` 插值生成控制目标，超过 ``controller.max_motion_steps`` 的动作会被拒绝。获得机器人工作范围后，可以向 agent 适配器传入 ``joint_limits``，配置具名的 ``[lower, upper]`` 手臂关节边界。反馈包含请求步数和实际执行步数。关节动作还返回测量值相对目标的偏差；EE 动作通过 agent 观测返回测量位姿，可用于判断目标是否到达。

笛卡尔控制工具 ``move_eef`` 的 ``targets`` 可指定 ``left``、``right`` 位姿，每个位姿为对应手臂基座坐标系下的 ``[x, y, z, yaw, pitch, roll]``。位置单位为米，角度单位为弧度，旋转约定为 ``Rz(yaw) Ry(pitch) Rx(roll)``。可选的 ``left_gripper``、``right_gripper`` 使用张开比例。控制器先将位姿转换到世界坐标，再求解所有指定手臂的 IK，全部成功后才对关节目标插值并执行。IK 失败会返回工具反馈，不推进物理仿真。这条执行路径在关节空间插值到 EE 目标，不保证末端沿笛卡尔直线运动。

工具适配器位于 ``rlinf/agents/robodojo/tools.py``，负责 schema、JSON 解析、note、hindsight、轮数和 agent 停止决策。环境负责仿真 episode、观测、奖励和物理步数预算。其公开方法 ``execute`` 接收 ``JointTarget``、``JointControls`` 或 ``EefTarget``；``observe`` 读取一个 slot 的原生观测，``snapshot`` 返回缓存的批量观测。原生 EE 位姿为 ``[x, y, z, qw, qx, qy, qz]``；agent 适配器将其转换为六维笛卡尔位置和欧拉角，可直接用于 ``move_eef``。

现有控制器仍可传入 ``[N, 14]`` 或 ``[N, C, 14]`` 数组，原生顺序为左臂、右臂、左夹爪、右夹爪。``state_action_order: aloha`` 将数组和状态顺序改为左臂、左夹爪、右臂、右夹爪。原生字典动作使用 ``left_arm_joint_state``、``right_arm_joint_state``、``left_ee_joint_state`` 和 ``right_ee_joint_state``，各字段形状为 ``[N, width]`` 或 ``[N, C, width]``。agent 维度名称始终指向原生关节，不受数组顺序影响。

安装
----

在 NVIDIA 主机上为仿真器创建独立环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env robodojo --venv .robodojo-venv
   source .robodojo-venv/bin/activate

该命令安装固定版本的 runtime、Isaac Sim 5.1、cuRobo 和 editable IsaacLab 源码。请一起保留虚拟环境及其中的 IsaacLab checkout。安装器会在修改环境前拒绝不兼容的平台、Python 和 Torch 选择。

也可以构建环境镜像：

.. code-block:: bash

   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-robodojo \
      --build-arg PLATFORM=nvidia -t rlinf:robodojo .

镜像安装仿真依赖；启动容器时需要开放 NVIDIA GPU，并挂载场景资产。

准备场景资产
------------

从 `RoboDojo 数据集 <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_ 下载机器人、物体、材质和布局资产。选择可写目录，并为下载文件和生成的机器人配置预留空间：

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

``ROBODOJO_ASSETS_ROOT`` 指向包含 ``Assets/`` 的目录。脚本展开机器人路径模板；移动资产后需要重新运行。也可以在环境配置中设置 ``assets_path``。

安装的 runtime 已包含兼容的 RoboDojo 源码。独立的 eval-only checkout（例如 ``../RoboDojo``）没有 runtime 所需的 ``src/collect_client`` 接口，因此本适配层使用打包源码；上游 checkout 可用于查阅资产和任务定义。首次加载场景时，还可能下载 NVIDIA 场景资源。

运行
----

按安装要求接受 Isaac Sim 许可，然后执行 smoke 检查：

.. code-block:: bash

   export OMNI_KIT_ACCEPT_EULA=YES
   PYTHONPATH="$PWD" python tests/e2e_tests/embodied/robodojo_smoke.py \
      --assets-root "$ROBODOJO_ASSETS_ROOT" --steps 3

该检查加载布局 0，读取相机和具名状态，交替调用 ``move_joints`` 和 ``move_eef`` 维持当前姿态，调用 ``done`` 结束 episode，最后关闭仿真器。退出码 0 表示这些操作成功，不代表任务完成。添加 ``--num-envs 2`` 可检查 slot 隔离和部分 reset。``--interface joint`` 检查数组接口；``--interface eef`` 直接检查类型化笛卡尔动作。

agent loop 可以创建已注册的环境，绑定 agent 工具适配器，并执行函数调用：

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

将 ``tools`` 交给 LLM rollout，并把 ``agent_observation`` 作为结构化上下文。``agent.observe`` 返回指令、具名关节、可用相机名称、成功标记和剩余控制步数、轮数。返回的 ``obs`` 保留图像 tensor，可交给支持视觉的模型或单独的感知转换模块。将 ``tool_feedback`` 序列化为工具响应，并使用所选 slot 的奖励训练模型。

``agent.step`` 同时接收 ``{name, arguments}`` 和 OpenAI 格式的 ``{type: function, function: {name, arguments}}``；``arguments`` 可以是字典或 JSON 字符串。已有的 RLinf tool worker 可以直接调用 ``agent.call(request.name, request.arguments, env_idx=0)``。

``take_pic`` 更新观测，不推进物理仿真。``wait`` 按指定的 ``steps`` 保持当前姿态。``done`` 要求提供 ``summary``，``give_up`` 要求提供 ``reason``；二者都可附带 ``hindsight``，并结束 episode，不会凭 agent 声明生成成功奖励。请读取观测中由 RoboDojo 任务条件决定的成功标记。参数错误通过 ``info["agent"]`` 返回 ``ok: False`` 和错误说明；仿真器故障则向调用方抛出异常。

可向 ``RoboDojoAgentTools`` 的第二个参数传入 ``{"max_turns": 64, "joint_limits": {}}``，配置 agent。每次工具调用都会消耗一轮，非法调用也计入；只有实际执行的控制才消耗 agent 的物理步数预算。agent 停止或耗尽轮数后需要显式 reset 环境。环境为各 slot 记录 reset 代次，适配器据此只清除对应 slot 的 agent 状态；``done`` 和 ``give_up`` 不改写仿真器的终止标记。agent 执行使用显式 reset，即使数组接口设置了 ``auto_reset`` 或 ``ignore_terminations``。一个 slot 的调用不会推进其他场景。

``reset`` 返回所有 slot 的观测。部分 reset 使用 ``env_idx=[0]``，同时提供按全局 slot 索引的完整 ``env_seeds`` 列表；其他 slot 保留原状态。``group_size`` 会将抽取的 reset seed 在组内重复。采集需要相互比较的轨迹时，保持这些 seed 固定。

对于数组控制器，``step`` 返回 chunk 汇总结果。``chunk_step`` 返回一份末尾观测和 info，并将奖励和终止标记放在 ``[N, C]`` tensor 的最后一列。``info["valid_action_mask"]`` 标记实际执行的控制。数组调用的 ``elapsed_steps`` 统计提交的 rollout 位置，包括 slot 结束后的 padding；agent 调用统计实际执行步数。``max_episode_steps`` 与 ``task_config.max_episode_steps`` 需要保持一致。

数组调用设置 ``auto_reset: True`` 后，完成的 slot 自动 reset，原观测、指标和 done mask 保留在 ``final_observation``、``final_info``、``_final_observation`` 和 ``_final_info`` 中。保持 ``enable_offload: False``：``close`` 和 ``offload`` 会永久释放仿真器，可重复调用。

验证
----

无需 Isaac Sim 即可运行 CPU 测试：

.. code-block:: bash

   python -m pytest tests/unit_tests/test_sim_env.py -q

这些测试使用假的外部 runtime 检查适配层。仓库变量 ``ROBODOJO_ASSETS_ROOT`` 指向 NVIDIA ``embodied`` runner 上准备好的资产后，真实仿真 smoke CI job 才会启用。请分别验证仿真启动、任务完成和 LLM 学习效果。
