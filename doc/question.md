# 原生宿主待解决问题

更新于 2026-11-23。本机 DSH、Codex 和 Claude Code（使用 DeepSeek 配置）均已收到真实模型响应。DSW 已完成 DSH 训练，以下两项涉及服务器端的原生训练环境。

| 状态 | 宿主 | 已确认的问题 | 需要准备什么 | 如何确认解决 |
|---|---|---|---|---|
| 本机已通；DSW 待处理 | Codex | macOS 现有 CLI 已返回 `READY`。DSW 上的 0.125.0 完成容器初始化，但模型服务返回 `403 unsupported_country_region_territory`。 | 本机 CLI 已可用于最小请求。后续若要做原生训练，需选择具备官方服务访问条件的运行环境。 | 在目标环境用原生 Codex 完成 Poker 分支采集和两轮训练。[本机测试](../runs/acceptance-20261122-a/local-cli/codex.json) |
| 本机 DeepSeek 已通；DSW 待验 | Claude Code | 本机 2.1.330 使用 `deepseek-v4.1-flash` 配置，已连续回复两条消息。DSW 的 2.1.119 已通过容器版本检查，现有配方指定 `claude-sonnet-4-6`，容器启动代码没有传入自定义模型服务地址。 | 若在 DSW 沿用 DeepSeek，需要在容器中同时配置已授权的服务地址、与其配套的凭据和模型名。 | 从 DSW 的 benchmark 容器向指定模型发送一次最小请求，取得有效回复后再验收 Poker 采集与训练。[本机 DeepSeek 会话核验](../runs/acceptance-20261122-a/local-cli/claude-deepseek-session.json) |

诊断和训练记录保存在[验收记录](../runs/acceptance-20261122-a/验收记录.md)中。
