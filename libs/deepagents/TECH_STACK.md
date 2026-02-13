# deepagents 技术栈文档

## 1. 项目定位

`deepagents` 是一个基于 LangChain/LangGraph 的 Python 深度智能体库，核心能力包括：
- 计划与待办（Todo）
- 文件系统操作（读/写/编辑/搜索）
- 子智能体（SubAgent）分工
- 记忆与技能加载（`AGENTS.md` / `SKILL.md`）
- 可选沙箱命令执行（`execute`）

代码主入口：`deepagents/graph.py` 中的 `create_deep_agent(...)`。

## 2. 语言与运行环境

- 语言：Python 3
- 版本要求：`>=3.11,<4.0`
- 类型化：包含 `py.typed`，对外提供类型提示（PEP 561）

## 3. 核心框架与 SDK

来自 `pyproject.toml` 的核心依赖：

- `langchain-core>=1.2.7,<2.0.0`
- `langchain>=1.2.7,<2.0.0`
- `langchain-anthropic>=1.3.1,<2.0.0`
- `langchain-google-genai>=4.2.0,<5.0.0`
- `wcmatch`（增强 glob 匹配能力）

代码层面的关键框架能力：
- LangChain Agent API：`create_agent`、Middleware 机制、结构化输出
- LangGraph：状态图执行、缓存（`BaseCache`）、检查点（`Checkpointer`）、存储（`BaseStore`）
- 默认模型：Anthropic `claude-sonnet-4-5-20250929`（可切换为任意 `provider:model`）

## 4. 架构分层

### 4.1 Agent 组装层

文件：`deepagents/graph.py`

负责将模型、工具、子智能体、中间件编排为可执行的深度 Agent。默认中间件栈包含：
- `TodoListMiddleware`
- `FilesystemMiddleware`
- `SubAgentMiddleware`
- `SummarizationMiddleware`
- `AnthropicPromptCachingMiddleware`
- `PatchToolCallsMiddleware`

可按参数启用：
- `MemoryMiddleware`
- `SkillsMiddleware`
- `HumanInTheLoopMiddleware`

### 4.2 Middleware 层

目录：`deepagents/middleware/`

主要能力：
- `filesystem.py`：注入 `ls/read_file/write_file/edit_file/glob/grep`，并按后端能力动态支持 `execute`
- `subagents.py`：子智能体注册与调度（`task` 工具）
- `summarization.py`：上下文摘要与裁剪
- `memory.py`：加载多个 `AGENTS.md` 作为长期上下文
- `skills.py`：按 Agent Skills 规范加载 `SKILL.md` 元数据并注入系统提示
- `patch_tool_calls.py`：工具调用修补与兼容性处理

### 4.3 Backend 抽象层

目录：`deepagents/backends/`

通过 `BackendProtocol` 统一文件能力：
- 列表/读取/写入/编辑
- `glob`/`grep`
- 文件上传下载
- 全异步镜像接口（`a*`）

后端实现：
- `StateBackend`：基于运行时状态，适合会话内临时文件
- `FilesystemBackend`：落盘存储（支持 `root_dir`、`virtual_mode` 等）
- `StoreBackend`：对接 LangGraph Store 持久存储
- `CompositeBackend`：路由式组合多后端
- `LocalShellBackend` / `SandboxBackendProtocol`：支持命令执行能力

## 5. 构建、包管理与发布

- 构建系统：`setuptools.build_meta` + `wheel`
- 包管理与任务执行：`uv`（`uv.lock` 已锁定依赖）
- 包元数据：`pyproject.toml`（MIT License，Beta 状态）

## 6. 代码质量与静态检查

- Lint/Format：`ruff`（启用 `ALL` 规则并按项目定制忽略项）
- 类型检查：`ty`（在 Makefile 中预留）
- 导入检查：`scripts/check_imports.py`
- 风格约束：Google 风格 docstring（`pydocstyle` 配置）

常用命令（`Makefile`）：
- `make lint`
- `make format`
- `make test`
- `make integration_test`
- `make check_imports`

## 7. 测试栈

- 测试框架：`pytest`
- 异步测试：`pytest-asyncio`
- 并行测试：`pytest-xdist`
- 覆盖率：`pytest-cov`
- 超时/网络控制：`pytest-timeout`、`pytest-socket`

测试目录结构：
- `tests/unit_tests/`：后端、中间件、图装配与边界行为
- `tests/integration_tests/`：与真实模型/链路相关的集成验证

集成测试环境变量：
- 必需：`ANTHROPIC_API_KEY`
- 可选：`LANGSMITH_API_KEY` 或 `LANGCHAIN_API_KEY`

## 8. 对外 API 与扩展点

对外主入口（`deepagents/__init__.py`）：
- `create_deep_agent`
- `FilesystemMiddleware`
- `MemoryMiddleware`
- `SubAgentMiddleware`
- `SubAgent` / `CompiledSubAgent`

主要扩展方式：
- 传入自定义 `tools`
- 传入自定义 `middleware`
- 定义自定义 `subagents`
- 注入自定义 `backend`（实例或工厂）
- 配置 `memory` 与 `skills` 来源

## 9. 技术栈总结（速览）

- **基础语言**：Python 3.11+
- **Agent 框架**：LangChain + LangGraph
- **模型生态**：Anthropic / Google GenAI（可扩展到其他 provider）
- **架构风格**：中间件驱动 + 后端协议抽象 + 子智能体编排
- **工程化**：uv + setuptools + ruff + pytest
- **特征能力**：文件系统工具链、长上下文摘要、技能系统、记忆系统、可选沙箱执行
