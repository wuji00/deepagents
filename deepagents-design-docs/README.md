# DeepAgents 设计文档

本目录包含 DeepAgents 核心 SDK (`deepagents/libs/deepagents`) 的详细设计文档，基于源码反向工程分析生成。

## 文档清单

| 文档 | 内容概述 |
|------|----------|
| [01-架构设计.md](./01-架构设计.md) | 总体架构、组件关系、数据流、核心设计理念 |
| [02-Backend系统设计.md](./02-Backend系统设计.md) | BackendProtocol、存储后端实现、文件操作抽象 |
| [03-Middleware系统设计.md](./03-Middleware系统设计.md) | 中间件架构、核心中间件详解、组合策略 |
| [04-Agent创建流程详解.md](./04-Agent创建流程详解.md) | `create_deep_agent()` 完整流程、设计决策分析 |

## 快速导航

### 如果你想知道...

- **DeepAgents 是什么？** → 阅读 [01-架构设计.md - 项目概述](./01-架构设计.md#1-项目概述)
- **如何自定义存储后端？** → 阅读 [02-Backend系统设计.md - 扩展开发](./02-Backend系统设计.md#8-扩展开发)
- **如何开发自定义中间件？** → 阅读 [03-Middleware系统设计.md - 自定义 Middleware 开发](./03-Middleware系统设计.md#5-自定义-middleware-开发)
- **`create_deep_agent` 内部做了什么？** → 阅读 [04-Agent创建流程详解.md - 详细流程解析](./04-Agent创建流程详解.md#4-详细流程解析)

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 应用层                              │
├─────────────────────────────────────────────────────────────┤
│  create_deep_agent() - 核心入口函数                          │
│  - 模型配置 / 工具注册 / Middleware 组装                      │
├─────────────────────────────────────────────────────────────┤
│                    Middleware 层                             │
│  TodoList → Filesystem → SubAgent → Summarization → ...     │
├─────────────────────────────────────────────────────────────┤
│                    Backend 层                                │
│  StateBackend / FilesystemBackend / StoreBackend / Sandbox  │
└─────────────────────────────────────────────────────────────┘
```

## 核心概念

### 1. Backend 系统

Backend 是存储抽象层，提供统一的文件操作接口：

- **StateBackend**：内存存储，适合短期对话
- **FilesystemBackend**：本地文件系统，持久化存储
- **StoreBackend**：外部存储（Redis/Postgres），适合生产环境
- **SandboxBackendProtocol**：支持命令执行的沙箱后端

### 2. Middleware 系统

Middleware 是功能扩展机制，每个 Middleware 负责特定功能：

- **TodoListMiddleware**：任务规划（`write_todos`/`read_todos`）
- **FilesystemMiddleware**：文件操作（`ls`/`read_file`/`write_file`/`edit_file`/`glob`/`grep`/`execute`）
- **SubAgentMiddleware**：子代理委派（`task`）
- **SummarizationMiddleware**：上下文压缩和摘要
- **SkillsMiddleware**：技能系统
- **MemoryMiddleware**：记忆加载
- **PatchToolCallsMiddleware**：工具调用修复

### 3. Agent 创建

`create_deep_agent()` 组装所有组件：

1. 初始化模型（默认 Claude Sonnet 4.5）
2. 配置 Backend（默认 StateBackend）
3. 计算 Summarization 默认值
4. 构建通用子代理
5. 处理用户自定义子代理
6. 组装中间件栈
7. 创建 LangGraph 并返回

## 代码结构

```
deepagents/libs/deepagents/deepagents/
├── __init__.py              # 导出公共 API
├── graph.py                 # 核心入口：create_deep_agent()
├── _version.py              # 版本信息
├── backends/                # Backend 系统
│   ├── __init__.py
│   ├── protocol.py          # BackendProtocol 定义
│   ├── state.py             # StateBackend 实现
│   ├── filesystem.py        # FilesystemBackend 实现
│   ├── store.py             # StoreBackend 实现
│   ├── composite.py         # CompositeBackend 实现
│   ├── local_shell.py       # LocalShellBackend 实现
│   ├── sandbox.py           # Sandbox 基类
│   └── utils.py             # 工具函数
└── middleware/              # Middleware 系统
    ├── __init__.py
    ├── filesystem.py        # FilesystemMiddleware
    ├── subagents.py         # SubAgentMiddleware
    ├── summarization.py     # SummarizationMiddleware
    ├── skills.py            # SkillsMiddleware
    ├── memory.py            # MemoryMiddleware
    ├── patch_tool_calls.py  # PatchToolCallsMiddleware
    └── _utils.py            # 中间件工具函数
```

## 设计原则

1. **开箱即用**：合理的默认配置，无需复杂设置即可使用
2. **可扩展性**：通过 Backend 和 Middleware 系统灵活扩展
3. **关注点分离**：每个组件负责单一功能，职责清晰
4. **可组合性**：Middleware 可以按需堆叠组合
5. **模型无关**：支持多种 LLM 提供商

## 扩展点

### 自定义 Backend

```python
from deepagents.backends.protocol import BackendProtocol, WriteResult

class MyBackend(BackendProtocol):
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        # 实现读取逻辑
        pass
    
    def write(self, file_path: str, content: str) -> WriteResult:
        # 实现写入逻辑
        return WriteResult(path=file_path, files_update=None)
```

### 自定义 Middleware

```python
from langchain.agents.middleware.types import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.system_prompt = "Instructions..."
    
    def before_agent(self, state, runtime, config):
        return {"my_data": {}}
    
    def modify_request(self, request):
        return request
```

## 参考资料

- [DeepAgents GitHub](https://github.com/langchain-ai/deepagents)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 文档](https://python.langchain.com/)

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
