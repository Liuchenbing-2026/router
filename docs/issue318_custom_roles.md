# Router #318：自定义 role 的 gRPC 本地集成修复

## 结论与交付边界

2026-10-14：`latest_reminder` 在 Router gRPC 预处理阶段被拒绝的问题已在本地修复。
新增用例先在原代码失败，再在修复后通过。保留原始 role，不改写成 `user`。
本分支是**依赖未合并上游 PR 的集成分支**，不是可直接发布的正式版本，也没有关闭上游 Issue。

- Router 基线：`442b8bb74fba09052428cf3af0513916a4f92b43`。
- 本地分支：`fix/grpc-custom-roles-318-20261014`；交付 SHA 以本文件所属 Git 提交及 HTML 台账为准。
- vLLM Rust 依赖：`8bbb174bc8a3cddffb6be19e7844009f742f76ce`。
- 该依赖来自既有 [vLLM PR #61446](https://github.com/vllm-project/vllm/pull/61446)，本次没有重复编写上游 renderer 补丁。2026-10-14 检查时为 Draft / OPEN。
- 关联：[Router #318](https://github.com/vllm-project/router/issues/318)、[vLLM #61372](https://github.com/vllm-project/vllm/issues/61372)。检查时均 OPEN。
- 本次只使用 CPU；无模型权重、无真实 engine/NPU 推理、无镜像切换或生产部署。

## 触发条件与修复链路

以 `user: Hello` 后追加 `latest_reminder: Be brief 世界` 为例：

```text
HTTP JSON → Router ChatMessage::Other → vllm-chat 消息
          → 模型 renderer → tokenizer → token_ids → gRPC worker
```

HTTP 后端可以直接转发 JSON；gRPC 后端必须先在 Router 上渲染并编码，因此不能仅修 JSON 反序列化。
原 `spec_messages_to_upstream` 对所有 `Other` 报错，且原来固定的 vLLM v0.29.0 类型无法表达 Custom。
即使升级到有 Custom 的版本，DeepSeek native renderer 仍需正确处理 `latest_reminder`。
PR #61446 同时修正角色支持和 user → reminder 的 assistant opener 位置。

原代码的失败用例输出（首次终端实测，未另存原始日志）：

```text
custom_role_reaches_grpc_worker_without_role_rewriting ... FAILED
preprocess: chat message role `latest_reminder` is not supported by the router gRPC frontend
```

本分支修改三个职责块：

1. **消息合同**：`src/backend/preprocess.rs` 将合法的自定义 role 转成上游 Custom；
   `developer` 使用原生 Developer。文本数组保留分段，避免合并后改变模型模板的分隔符。
   非法 role/content 返回错误，不因 `expect` 触发 panic。
2. **依赖与渲染**：`Cargo.toml` / `Cargo.lock` 将三个 vLLM Rust crate 统一固定到上述 PR SHA。
   Hugging Face Jinja renderer 可收到自定义 role；DeepSeek V4/V4.1 native renderer 接受 `latest_reminder`，
   其他不受该模型支持的角色仍明确报错。
3. **升级适配**：`EffortValue`、`ChatTool.defer_loading`、模型文件 revision 参数，以及
   `DecodedText.text` 的增量/flush 文本读取。gRPC wire 仍只传 token IDs，不改 proto。

### 不声称支持的输入

上游 Custom 只有 role/content，不能无损携带任意 JSON 扩展字段。
因此 `name`、`metadata`、`tool_calls` 等自定义消息附加字段，以及 text part 附加字段明确报错，
不能静默丢弃。只接受字符串或标准 text-part 数组；不接受多模态内容、null/missing content。
这解决的是 #318 中 role/content 形态的复现，不是所有任意 raw-message 扩展的完整兼容。

## 本次验证

| 项目 | 结果 | 范围 |
| --- | --- | --- |
| 新增 gRPC 自定义角色集成 | 4 passed | HF 模板、DeepSeek V4/V4.1、参考 token IDs、cache hit、本地 mock worker、错误/continuation 合同 |
| backend 定向单测 | Cargo 报告 34 passed | 其中 2 项既有真实模型对照测试因未设置 `VLLM_ROUTER_MODEL` 提前返回；实际执行断言的为 32 项 |
| 既有 gRPC L0 集成 | 1 passed | 缓存一致性、指标、流式、重试 |
| 既有 HTTP/gRPC 链路 | 6 passed | 本地 HTTP/gRPC mock、token IDs、健康、usage chunk |
| 协议/参数合同 | 104 passed | HTTP 消息保留与输入校验 |
| Python 参考编码器 | 4 cases passed | V4/V4.1 × 有/无 assistant history，chat mode，含 Unicode |
| 构建与格式 | all-targets check / rustfmt / diff check 通过 | 仍有既有 atomic fetch_update 弃用警告；不是全量测试执行 |

Rust 用例报告合计 149 passed，其中 2 个提前返回不计为真实模型验证；147 项实际执行。
Python 4 cases 单独计量，不是额外四个 Rust tests。
测试使用仓库内小型 byte-level BPE tokenizer；没有验证 DeepSeek 正式词表和引擎的端到端一致性。
不包含性能收益、模型输出质量、全部测试或远端 CI 结论。

### 后续入口与缓存回归（2026-10-14）

`tests/grpc_custom_roles.rs` 新增三项回归后，当前该测试目标实际执行 7 项并全部通过：

- 使用实际服务的 HTTP app 处理 `/v1/chat/completions`，覆盖流式和非流式响应；
  独立编码预期 prompt，并检查 mock gRPC worker 收到的 token IDs。
- 五类非法自定义消息返回 HTTP 400，worker 未收到请求；随后合法消息仍能成功。
- 自定义 role 或正文变化时分别建立 L0 缓存项；重复请求命中；缓存已预热时，
  不可表示的 `metadata` 扩展仍被拒绝，缓存计数不变。

此增量只补充测试，不更改运行时依赖或扩展字段合同。上表 147 项是首版实际执行口径，
不把历史批次与本次执行数量相加作为一次完整验收结果。
本次联合重跑 `grpc_custom_roles`（7）、`grpc_frontend_l0`（1）、
`grpc_vs_http_e2e`（6），共 14 项实际执行并通过；格式和 diff 检查通过。

## 复现

Linux aarch64，Rust 1.99.0。先准备 Cargo.lock 指定依赖；缓存齐全后以下命令可离线运行：

```bash
cargo test --locked --offline --test grpc_custom_roles -- --nocapture
cargo test --locked --offline --lib backend:: -- --nocapture
cargo test --locked --offline --test grpc_frontend_l0 --test grpc_vs_http_e2e -- --nocapture
cargo test --locked --offline --lib protocols:: -- --nocapture
cargo check --locked --offline --all-targets
git diff --check
```

参考编码器用独立 vLLM 源码 checkout，切到上述固定 SHA。使用虚拟环境 Python：

```bash
/path/to/venv/bin/python tests/check_deepseek_reminder_reference.py /path/to/vllm-checkout
```

本次参考脚本实际使用已有 `/tmp/venv030/bin/python`（Python 3.12.13，安装元数据 vLLM 0.30.0）；
但执行的是固定 PR SHA 的两个纯 Python encoder 文件，不是安装包的模型推理。
这不等于验证了 `glm5_test` 的镜像 tag/digest，也不等于 v0.30.0 发布版已包含上游修复。
未进入/重启用户 V4.1 会话，未改其安装包。

## 发布与回退

1. 等上游 #61446 合入或与维护者确认替代方案，将临时 PR pin 换成正式接受的 SHA/tag，重新跑上述回归。
2. 如要求任意 raw-message 扩展兼容，需先扩上游类型/renderer 合同；不能绕过校验偷偷丢字段。
3. 真机验收需隔离的 0.30 系列服务环境、实际镜像 ID、模型 tokenizer 与可用的 gRPC worker。
   用同一请求比较 Python/Rust prompt/token IDs，再测流式/非流式生成；这一步才需要 GPU/NPU 推理资源。
4. 本分支独立于认证修复和 KV trust 原型；首版 `e008366` 已推送到个人 fork 的同名分支，未创建 PR/Issue。
   另一修复分支 `fix/issue-318-custom-grpc-roles-20261014` 保留不变；两版依赖与扩展字段支持不同，尚未统一。
   回退时保留此分支，使用原基线构建，
   不对其他工作树执行 reset/checkout，也不把这份临时依赖升级混入生产。
