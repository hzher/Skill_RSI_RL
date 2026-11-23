# 在 DSH、Codex 或 Claude Code 上采样

选一种宿主，再选一种训练方式。**在线**由当前学生边跑任务边收反馈，每轮采完再更新。**离线**先用宿主给候选动作评分，随后可在本地重复训练。两种方式彼此独立，采集阶段都会调用所选宿主的模型服务。

## 准备原生运行环境

先完成[训练环境配置](00_环境配置.md)。只在首次运行 CL-Bench 时，额外准备 Python 3.13。下面用另一个 Conda 环境提供解释器，由项目脚本按官方锁文件安装 Poker 依赖。完成后当前终端仍在 `skill-rl`，原生任务由 `.venv-bench/bin/python` 执行。

```bash
conda create -n clbench-py313 python=3.13 -y
conda run -n clbench-py313 bash scripts/setup.sh bench
.venv-bench/bin/python --version
```

| 选择的宿主 | `RECIPE` 取值 | 运行机还需准备 |
|---|---|---|
| DeepSeek Harness | `poker` | 运行 `./scripts/setup.sh dsh` 安装 Node 24.22.3 与 DSH 0.2.2-rc.2，导出 `DEEPSEEK_API_KEY` |
| Codex | `poker-codex` | 可用的 Docker daemon，导出 `OPENAI_API_KEY`。容器安装 Codex CLI 0.125.0 |
| Claude Code | `poker-claude` | 可用的 Docker daemon，导出 `ANTHROPIC_API_KEY`。容器安装 Claude Code CLI 2.1.119 |

Codex 和 Claude Code 使用 `node:22-slim`。运行机需要能拉取镜像，容器内需要能访问 apt 和 npm 包源。先用 `docker run --rm node:22-slim node --version` 检查基础镜像。DSH 安装后用 `PATH="$PWD/.venv-dsh/bin:$PATH" .venv-dsh/bin/dsh --version` 核对版本。模型凭据通过当前终端的环境变量传给所选宿主。

**本次验收的 DSW** 已将 Docker 装入 `skill-rl`，并准备了带 apt 和 npm 缓存的镜像。使用 Codex 或 Claude Code 时，可在 train、dev 配方副本的 `benchmark.system_params` 中加入 `"docker_image": "skill-rl/native-node22:locked-20261122"`，避免每次创建容器都从外网取包。

自带的 DSH 配方指向 DeepSeek 官方地址。如果使用其他已获授权的网关，复制 train、dev 两份 JSON，将两份的 `benchmark.system_params` 中的 `provider`、`base_url` 和 `model` 改成该网关对应值，再把下文的 `--config` 指向副本。配置文件内显式写出的地址优先于环境变量。密钥保存在环境变量中。

以下以 DSH 为例。选择 Codex 或 Claude Code 时，换成表中对应的 `RECIPE` 并准备该服务的凭据。

```bash
RECIPE=poker
```

若当前 Bash 终端还没有导出密钥，执行下段命令，按提示输入所选宿主的密钥并回车。输入时屏幕不回显。

```bash
case "$RECIPE" in
  poker)        read -rsp 'DeepSeek 密钥：' DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY ;;
  poker-codex)  read -rsp 'OpenAI 密钥：' OPENAI_API_KEY && export OPENAI_API_KEY ;;
  poker-claude) read -rsp 'Anthropic 密钥：' ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY ;;
esac
echo
```

## 在线采样并更新两轮

```bash
CUDA_VISIBLE_DEVICES=0 python -m skill_rsi_rl.online \
  --config "configs/${RECIPE}-train.json" --model "$SKILL_RL_MODEL" \
  --bench-python .venv-bench/bin/python --rounds 2 --output "runs/${RECIPE}-online"
```

默认从基座开始，同一初态每组采 4 条独立序列。要接着一份离线 LoRA 开始在线训练，可加 `--adapter runs/v2-smoke/train/checkpoint-0001`，并换新输出目录。

成功时，`round-0001/` 和 `round-0002/` 各有一份 `batch.json`、`checkpoint/` 和 `result.json`。第二轮的 `parent_policy_id` 应等于第一轮的 `policy_id`，两轮的 `metrics.optimizer_step` 分别为 1、2。

中断后用原配置和原输出目录重跑，程序核验并复用已封存的批次和 checkpoint。采样中断的半条轨迹会被拒绝，还有尝试名额时换一组采样。如果已生成 `no-update.json`，说明该轮尝试已用尽。先查看其中的 `rejected`，按原因修复凭据或调整采样配置，再用**新的输出目录**启动。

## 采集真实分支供离线训练

如果目的是取得固定数据，可以直接从这一步开始。每个候选从相同牌局和相同初始记忆出发，记录后续官方得分。下面分别生成训练集与开发集的评分文件。

```bash
for split in train dev; do
  .venv-bench/bin/python -m skill_rsi_rl.fresh_branches \
    --config "configs/${RECIPE}-$split.json" --memory configs/poker-public-memory.json \
    --source-ref "configs/${RECIPE}-$split.json + configs/poker-public-memory.json" \
    --output "runs/${RECIPE}-branches/$split"
done
```

完成后，`runs/${RECIPE}-branches/` 的 `train/` 和 `dev/` 各有一份 `scored.json`。将这个目录传给[离线教程的入口 B](02_用已有数据做离线训练.md#1-选你的数据入口)，即可完成学生采样和 LoRA 训练。
