# Skill RL Train · 在 CL-Bench 上训练记忆策略

打第二手牌时，第一手里对手的哪次跟注值得再看？这个项目在 Continual Learning Bench（CL-Bench）上训练一个 **Qwen3-8B 记忆助手**，让它学会何时记、记哪条、何时取回。以 Poker 为例，DeepSeek Harness（DSH）、Codex 或 Claude Code 负责出牌，CL-Bench 负责发牌和计分。训练只更新记忆助手的 **LoRA**，即加载到基座模型上的一组小规模可训练权重。

**贡献新算法，从自己的目录开始。** 每个算法独占 `skill_rsi_rl/algorithms/<算法名>/`，训练入口、损失函数和测试都放在里面。运行 `python -m skill_rsi_rl.algorithms --list` 即可查看可用模式。GRPO 支持离线和在线，`best_action_sft` 提供可运行的离线监督训练示例。目录边界、接入步骤和验收命令见[新算法贡献指南](doc/05_贡献新的后训练算法.md)。

![从左到右，记忆助手取出笔记、原生宿主据此出牌、CL-Bench 记录得分](doc/assets/memory-poker.jpg)

[单卡快速开始](#快速开始) · [按场景选教程](#典型试用场景) · [离线与在线训练](#离线与在线训练) · [宿主支持与实测](#宿主支持与实测) · [四臂评测](#启动学生服务并做四臂配对评测)

## 典型试用场景

| 你想试什么 | 从什么开始 | 能得到什么 |
|---|---|---|
| 我有一张 H20，想跑出第一份权重 | 用本地 Qwen3-8B 运行内置记忆场景，无需模型服务账号。[单卡首训](doc/01_单卡首次训练.md) | 一份 LoRA，以及训练前后在开发集上的对照报告 |
| 我有上一批实验数据，想换参数再训一次 | 有分支评分 `scored.json` 就先采样，有训练记录 `train.jsonl` 就直接训练。[已有数据做离线训练](doc/02_用已有数据做离线训练.md) | 同一批记录可反复使用，无需重新付费跑牌局 |
| 我想知道学生更新后会怎样记下一手牌 | 让当前学生在真实 Poker 中采样，更新后换新权重再跑一轮。[在线采样与更新](doc/04_CL-Bench采集与在线训练.md#在线采样并更新两轮) | 每轮的真实轨迹、官方奖励和新权重 |
| 我有两张 H20，或上次训练只跑了一轮 | 用同一份 replay 启动 DDP，或从完整 checkpoint 恢复优化器。[多卡与断点续训](doc/03_多卡训练与断点续训.md) | 多卡分担样本计算，或接着上次的步数训练 |
| 我想知道加上记忆助手能否多赢筹码 | 固定出牌模型与发牌顺序，对比原生、基座、离线和在线四组。[四臂配对评测](#启动学生服务并做四臂配对评测) | 同一批牌局上的奖励差、成功率差与完整配对数 |

## 快速开始

准备 Linux、支持 BF16 的 NVIDIA H20 和本地 Hugging Face 格式的 Qwen3-8B 权重。从项目根目录（能看到 `EasyR1/` 和 `scripts/`）执行以下命令。首次实验使用内置场景，训练和评测都在本机进行。

**1. 安装训练环境。** 下面是已验证的 Python 3.11、PyTorch 2.6.0 和 CUDA 12.4 组合。已有配好的 `skill-rl` 环境可直接激活，进入第 2 步。[完整环境教程](doc/00_环境配置.md)

```bash
conda create -n skill-rl python=3.11 pip -y
conda activate skill-rl
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r env/train.txt -c env/easyr1-h20-constraints.txt
python -m pip install --no-deps -e ./EasyR1 -e .
```

**2. 指向权重，检查 GPU。** 将 `SKILL_RL_MODEL` 改为模型实际目录，里面应有配置、tokenizer、对话模板与全部权重分片。下面用物理卡 0 举例，预检输出 `"ready": true` 即可继续。

```bash
export SKILL_RL_MODEL="$PWD/models/Qwen3-8B"
export SKILL_RSI_PYTHON="$CONDA_PREFIX/bin/python"
T="$SKILL_RSI_PYTHON"
CUDA_VISIBLE_DEVICES=0 "$T" -m skill_rsi_rl.preflight \
  --expected-gpus 1 --backend easyr1 --model "$SKILL_RL_MODEL"
```

**3. 预览并启动训练。** `--plan` 打印命令，`--run` 真正执行。每次独立实验使用一个新目录。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_1h20.sh --plan runs/v2-smoke
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_1h20.sh --run  runs/v2-smoke
```

默认 v2 场景用 128 个训练状态和另外 64 个开发状态，每个状态抽样 16 次。完成后可在输出目录找到数据、权重和成绩。

| `runs/v2-smoke/` 下的产物 | 怎么用 |
|---|---|
| `train.jsonl`、`dev.jsonl` | 保存输入、候选动作、采样概率和奖励，可复用作离线训练数据（replay） |
| `train/checkpoint-0001/` | `adapter_model.safetensors` 是 LoRA 权重，`training_state.pt` 保存续训所需的优化器和进度 |
| `dev-score.json` | `reward_delta` 是相对基座的平均奖励差，`success_delta_pp` 是成功率变化的百分点。两者都可能为负 |

## 学生究竟学什么

“学生”就是上面的记忆助手。每次宿主需要回答问题时，学生能看到当前问题、已公开的反馈和这条序列自己的记忆。代码为它列出当前可用的候选，它从中选一个。

| 动作 | 本次发生什么 | 何时有用 |
|---|---|---|
| `noop` 跳过 | 保留记忆，本次不附笔记 | 当前牌面和反馈已经足够判断 |
| `recall` 回读 | 从指定条目取原文，附给出牌模型 | 想看清对手上次面对加注时到底做了什么 |
| `compact` 摘录 | 压缩空白，截取几条笔记的开头拼给出牌模型，同时保留各条原文 | 想在有限篇幅内同时看到几手牌的记录 |
| `write` 写入 | 把候选中的公开反馈存入指定键，供之后读取 | 刚观察到对手跟注，想让下一手仍能看到这条信息 |

单卡首训用库存交接来练同一件事。比如交接单刚把 B 的编号从 R09 更正为 R42，学生可以用 `write` 把更正存好。等下一班只问“B 的编号是什么？”，再用 `recall` 把新编号交给答题者。计分同时看眼前的核对结果和下一次查询能否答对。若起初既没有反馈也没有笔记，学生只有 `noop` 可选。[内置交接场景](skill_rsi_rl/scenarios_v2.py) · [动作实现](skill_rsi_rl/contracts.py) · [四个动作的完整例子](plan/01_总体方案与架构.md)

## 离线与在线训练

**组相对策略优化（GRPO）** 可以理解为“在同一起点比较几次尝试”。如果几条轨迹从同一手牌出发，拿到的奖励却不同，学生就能根据组内差异调整选动作的概率。项目复用 EasyR1 的策略损失与 KL 计算。

| 方法 | 一组数据怎样产生 | 下一次更新用什么 |
|---|---|---|
| 离线（off-policy） | 从同一初态分别固定首个候选，后续沿用相同规则。用相同计分窗口的官方回报给候选评分，再由冻结学生采样并记录当时的概率。 | 重读固定 replay，可训练多轮。内置首训则由场景规则直接提供评分。 |
| 在线（on-policy） | 固定当前学生，从同一初态独立跑多条完整序列。学生每次按概率选动作，记忆也随之改变。 | 用这组序列的回报更新一次，下一轮由新权重重新采集。 |

**怎样读真实结果？** 一次 DSH Poker 离线验收从同一初态各跑两手，首步选 `noop`、`recall`、`compact` 的实得均分分别为 **4.3、9.9、3.4**。冻结学生在这一个状态上抽样 96 次，组成一个训练组。换到另一个开发状态，训练前后都选了 `recall`，同得 **10.75**。这个小实验验证了权重更新，单次分支的分差和这一个开发状态不足以判断泛化收益。

### 在真实 Poker 上试一次

先给原生任务准备独立的 Python 3.13 环境。保持当前终端在 `skill-rl`，按下表选一个宿主配置凭据。原生采集和评测会调用所选服务并可能计费。[原生环境与凭据配置](doc/04_CL-Bench采集与在线训练.md)

```bash
conda create -n clbench-py313 python=3.13 -y
conda run -n clbench-py313 bash scripts/setup.sh bench
B="$PWD/.venv-bench/bin/python"
```

| 宿主 | `RECIPE` | 宿主侧准备 |
|---|---|---|
| DeepSeek Harness（DSH） | `poker` | 用 `bash scripts/setup.sh dsh` 安装 Node 24.22.3 和 DSH 0.2.2-rc.2，配置对应的 `DEEPSEEK_API_KEY` |
| Codex | `poker-codex` | 准备 Docker、`node:22-slim` 和容器可用的 apt/npm 包源。默认配方使用 `OPENAI_API_KEY` 和 Codex CLI 0.125.0 |
| Claude Code | `poker-claude` | Docker 条件与 Codex 相同。默认配方使用 `ANTHROPIC_API_KEY`、Claude Code 2.1.119 和 `claude-sonnet-4-6` |

仓内 DSH 配方使用官方地址与 `deepseek-flash`。使用其他已获授权的兼容网关时，复制训练与开发配方，把 `benchmark.system_params` 中的 `base_url` 和 `model` 成对改为实际值，并使用该网关的凭据。命令中的 `--config` 应指向这些副本。[DSH 训练配方](configs/poker-train.json) · [开发配方](configs/poker-dev.json)

<details>
<summary><b>离线实验</b> · 从三端中的一个采集分支评分，再训练 LoRA</summary>

**① 采集分支。** 以 DSH 为例，分别采集训练和开发状态的官方评分。已有评分可以跳到第 ② 步。

```bash
RECIPE=poker
BRANCHES="runs/${RECIPE}-branches"
for split in train dev; do
  "$B" -m skill_rsi_rl.fresh_branches \
    --config "configs/${RECIPE}-$split.json" --memory configs/poker-public-memory.json \
    --source-ref "configs/${RECIPE}-$split.json + configs/poker-public-memory.json" \
    --output "$BRANCHES/$split"
done
```

**② 把评分变成训练数据。** 将 `BRANCHES` 设为含有 `train/scored.json` 和 `dev/scored.json` 的目录。冻结的学生从候选中采样，把动作、采样概率和对应奖励写入一个新的数据目录。

```bash
BRANCHES=runs/poker-branches
DATA=runs/poker-replay
mkdir -p "$DATA"
for split in train dev; do
  CUDA_VISIBLE_DEVICES=0 "$T" -m skill_rsi_rl.collect sample --split "$split" \
    --model "$SKILL_RL_MODEL" --device cuda:0 --dtype bfloat16 \
    --replay-spec "$BRANCHES/$split/scored.json" --samples-per-state 64 \
    --temperature 1.5 --uniform-mix 0.35 --output "$DATA/$split.jsonl"
done
```

**③ 校验、训练、比较。** 已有 replay 可从这里开始。将 `DATA` 指向已有 `train.jsonl` 和 `dev.jsonl` 的目录，将 `OUT` 设为本次实验的新目录。只有训练集时也可以先执行中间的训练命令。[已有数据的用法](doc/02_用已有数据做离线训练.md)

```bash
DATA=runs/poker-replay
OUT=runs/poker-offline
"$T" -m skill_rsi_rl.verify_replay --train "$DATA/train.jsonl" --dev "$DATA/dev.jsonl"
CUDA_VISIBLE_DEVICES=0 "$T" -m skill_rsi_rl.train --backend easyr1 \
  --model "$SKILL_RL_MODEL" --device cuda --dtype bf16 \
  --replay "$DATA/train.jsonl" --output "$OUT/train" --epochs 1 --per-device-batch-size 1
CUDA_VISIBLE_DEVICES=0 "$T" -m skill_rsi_rl.dev_eval --model "$SKILL_RL_MODEL" \
  --device cuda:0 --adapter "$OUT/train/checkpoint-0001" \
  --train-replay "$DATA/train.jsonl" --replay "$DATA/dev.jsonl" --output "$OUT/dev-score.json"
```

校验时看 `split_overlap` 三项均为 0，训练集的 `eligible_grpo_groups` 大于 0。组内奖励全相同就没有动作优劣信号。对已有 DSH 历史归档，可用 `verify_history` 核对决策边界，再用 `branch_score` 在授权范围内重新评分。[离线教程](doc/02_用已有数据做离线训练.md)

</details>

**在线实验**可以直接从基座开始。下例用 DSH 跑两轮。测试其他宿主时，换用对应的 `RECIPE` 和新的输出目录。若想从已训练的 LoRA 起步，可加 `--adapter runs/v2-smoke/train/checkpoint-0001`。

```bash
RECIPE=poker
CUDA_VISIBLE_DEVICES=0 "$T" -m skill_rsi_rl.online \
  --config "configs/${RECIPE}-train.json" --model "$SKILL_RL_MODEL" \
  --bench-python "$B" --rounds 2 --output runs/dsh-online
```

检查 `round-0001/result.json` 的 `policy_id` 是否等于 `round-0002/result.json` 的 `parent_policy_id`，以及两轮的 `metrics.optimizer_step` 是否依次为 1、2。这样能确认第二轮用到了第一轮发布的权重。若生成 `no-update.json`，查看被拒组的原因。[在线训练与恢复](doc/04_CL-Bench采集与在线训练.md#在线采样并更新两轮)

## 宿主支持与实测

| 宿主 | 已验证的范围 |
|---|---|
| DSH | 服务器上使用 `deepseek-v4.1-flash` 完成真实 Poker 分支采集、离线 LoRA 训练和两轮在线 GRPO。在线两轮共 4 条独立序列、8 手牌，第二次更新的 144 个 LoRA 张量全部发生变化。 |
| Codex | macOS 本机 CLI 收到真实模型回复，DSW 容器完成 CLI 初始化。该 DSW 的模型探针收到地区限制 403，完整 Poker 训练待验。 |
| Claude Code | macOS 的 Claude Code 2.1.330 在用户的 DeepSeek 配置下连续收到两条回复，两条记录的模型标识均为 `deepseek-v4.1-flash`。DSW 容器已验证 CLI 安装，完整 Poker 训练待验。 |

三端都有采集、训练和配对评测入口。Claude Code 的默认容器入口没有转发自定义服务地址，因此本机的 DeepSeek 配置还需在 DSW 容器中单独接入。逐项记录保存在本工作区的 `runs/acceptance-20261122-a/验收记录.md`。[宿主待解决事项](doc/question.md)

## 启动学生服务并做四臂配对评测

比较同一批牌时，让**原生宿主**单独打一次，再分别接入**基座学生、离线学生、在线学生**各打一次。“四臂”就是这四组设置，发牌顺序一致，可以逐序列比较奖励。用自己的运行目录替换下面的示例路径。

```bash
"$T" -m skill_rsi_rl.policy_configs --model "$SKILL_RL_MODEL" \
  --report runs/poker-offline/dev-score.json --adapter runs/poker-offline/train/checkpoint-0001 \
  --online-result runs/dsh-online/round-0002/result.json --online-port 8713 --output runs/policies
```

另开三个终端，分别执行上一步输出的服务指令，按可用 GPU 调整 `--device`，让端口 8711、8712、8713 同时可达。回到已导出宿主凭据的原终端，在项目根目录运行下面的 DSH 示例。`--config` 使用与实验相同的执行器配方。

```bash
bash scripts/eval_dsh.sh --run --tag dsh-poker --profile quick --task exploitable_poker \
  --config configs/poker-train.json --base-policy runs/policies/base.json \
  --offline-policy runs/policies/trained.json --online-policy runs/policies/online.json
```

`quick` 只有一条完整配对，适合检查流程。使用 `--profile full` 做完整评测，报告在 `runs/evaluations/<tag>-<profile>/paired-report.json`。可先用 `--check` 替换 `--run` 核对策略服务身份。Codex 和 Claude Code 使用 `.venv-bench/bin/python -m skill_rsi_rl.evaluate`，分别指定 `--harness codex` 或 `--harness claude` 及相应配方。本次验收覆盖了 DSH 训练和权重重载，完整四臂收益仍需单独评测。

## 续训与项目结构

- **离线续训**需要保留 replay、模型、卡数和每卡 batch，用 `--resume-from` 指向含优化器状态的 checkpoint。`--epochs 2` 表示总共训练两轮。换数据或卡数时，使用新目录和 `--init-adapter` 继承 LoRA。[双卡与续训示例](doc/03_多卡训练与断点续训.md)
- **在线续训**使用相同配置、初始权重和输出目录重跑，程序会核验并复用已封存的有效批次和 checkpoint。若已耗尽尝试名额，先处理 `no-update.json` 中记录的原因，再用新目录开始。
- **迁移到另一台机器**时复制完整项目，保留内置 EasyR1、CL-Bench 和 `sources.lock.json`，在新机器重建环境并指定权重路径。模型权重、replay 和 `runs/` 下的产物按需另行复制。

| 位置 | 主要内容 |
|---|---|
| [skill_rsi_rl/](skill_rsi_rl/) | 候选记忆动作、学生采样、训练、服务和评测 |
| [configs/](configs/) · [scripts/](scripts/) · [env/](env/) | 三端 Poker 配方、启动脚本与环境版本约束 |
| [EasyR1/](EasyR1/) · [CL-Bench](vendor/continual-learning-bench/) · [sources.lock.json](sources.lock.json) | 随项目提供的训练算子、官方任务和源码校验清单 |
| [中文教程](doc/00_环境配置.md) · [总体方案与架构图](plan/01_总体方案与架构.md) | 按场景启动的步骤，以及离线、在线算法流程图和时序图 |
