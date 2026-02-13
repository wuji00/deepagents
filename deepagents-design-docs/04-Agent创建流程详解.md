# Agent 创建流程详解

## 1. 概述

`create_deep_agent()` 是 DeepAgents SDK 的核心入口函数，负责组装和配置一个功能完整的 Agent。本文档详细解析其内部工作流程和设计决策。

**文件位置**：`deepagents/graph.py`

## 2. 函数签名与参数

```python
def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
```

### 2.1 参数分类

| 类别 | 参数 | 说明 |
|------|------|------|
| **核心配置** | `model` | LLM 模型（字符串或实例） |
| | `tools` | 自定义工具列表 |
| | `system_prompt` | 自定义系统提示 |
| **扩展配置** | `middleware` | 额外中间件 |
| | `subagents` | 自定义子代理 |
| | `skills` | 技能路径列表 |
| | `memory` | 记忆文件路径 |
| **存储配置** | `backend` | 存储后端 |
| | `checkpointer` | 状态检查点 |
| | `store` | 持久化存储 |
| **控制配置** | `interrupt_on` | 人工审核点 |
| | `debug` | 调试模式 |
| **其他** | `response_format` | 结构化输出 |
| | `name` | Agent 名称 |

## 3. 创建流程总览

```
create_deep_agent(params)
    │
    ├──▶ 步骤 1: 模型初始化
    │
    ├──▶ 步骤 2: Backend 配置
    │
    ├──▶ 步骤 3: 计算 Summarization 默认值
    │
    ├──▶ 步骤 4: 构建通用子代理
    │
    ├──▶ 步骤 5: 处理用户子代理
    │
    ├──▶ 步骤 6: 合并子代理列表
    │
    ├──▶ 步骤 7: 组装主 Agent 中间件栈
    │
    ├──▶ 步骤 8: 合并 System Prompt
    │
    └──▶ 步骤 9: 创建并返回 LangGraph
```

## 4. 详细流程解析

### 4.1 步骤 1: 模型初始化

```python
if model is None:
    model = get_default_model()  # Claude Sonnet 4.5
elif isinstance(model, str):
    model = init_chat_model(model)  # "openai:gpt-4o" → ChatOpenAI
```

**默认模型配置**：
```python
def get_default_model() -> ChatAnthropic:
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )
```

**模型字符串格式**：
- `"openai:gpt-4o"` → OpenAI GPT-4o
- `"anthropic:claude-3-5-sonnet"` → Anthropic Claude
- `"google:gemini-pro"` → Google Gemini

### 4.2 步骤 2: Backend 配置

```python
# 默认使用 StateBackend
backend = backend if backend is not None else (lambda rt: StateBackend(rt))
```

**Backend 类型**：
- `BackendProtocol` 实例：直接使用
- `BackendFactory` 函数：延迟创建（运行时传入 ToolRuntime）

### 4.3 步骤 3: 计算 Summarization 默认值

```python
summarization_defaults = _compute_summarization_defaults(model)
```

**智能默认值策略**：

```python
def _compute_summarization_defaults(model: BaseChatModel) -> SummarizationDefaults:
    # 检查模型是否有 max_input_tokens 配置
    has_profile = (
        model.profile is not None
        and "max_input_tokens" in model.profile
    )
    
    if has_profile:
        # 有配置：使用比例策略
        return {
            "trigger": ("fraction", 0.85),    # 85% 触发摘要
            "keep": ("fraction", 0.10),       # 保留最后 10%
            "truncate_args_settings": {
                "trigger": ("fraction", 0.85),
                "keep": ("fraction", 0.10),
            },
        }
    else:
        # 无配置：使用固定值策略
        return {
            "trigger": ("tokens", 170000),    # 17万 tokens 触发
            "keep": ("messages", 6),          # 保留最后 6 条消息
            "truncate_args_settings": {
                "trigger": ("messages", 20),
                "keep": ("messages", 20),
            },
        }
```

### 4.4 步骤 4: 构建通用子代理

**通用子代理（general-purpose）** 是默认创建的子代理，具有与主 Agent 相同的能力。

```python
# 通用子代理中间件栈
gp_middleware: list[AgentMiddleware] = [
    TodoListMiddleware(),
    FilesystemMiddleware(backend=backend),
    SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=summarization_defaults["trigger"],
        keep=summarization_defaults["keep"],
        trim_tokens_to_summarize=None,
        truncate_args_settings=summarization_defaults["truncate_args_settings"],
    ),
    AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
    PatchToolCallsMiddleware(),
]

# 可选：添加 SkillsMiddleware
if skills is not None:
    gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))

# 可选：添加 HumanInTheLoopMiddleware
if interrupt_on is not None:
    gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

# 通用子代理定义
general_purpose_spec: SubAgent = {
    **GENERAL_PURPOSE_SUBAGENT,  # 预定义模板
    "model": model,
    "tools": tools or [],
    "middleware": gp_middleware,
}
```

**GENERAL_PURPOSE_SUBAGENT 模板**：
```python
DEFAULT_GENERAL_PURPOSE_DESCRIPTION = """
General-purpose agent for researching complex questions, searching for files 
and content, and executing multi-step tasks.
"""

GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}
```

### 4.5 步骤 5: 处理用户子代理

```python
processed_subagents: list[SubAgent | CompiledSubAgent] = []

for spec in subagents or []:
    if "runnable" in spec:
        # CompiledSubAgent：直接使用
        processed_subagents.append(spec)
    else:
        # SubAgent：填充默认值
        processed_spec = _process_subagent_spec(spec, model, tools, backend)
        processed_subagents.append(processed_spec)
```

**子代理配置继承规则**：

```python
def _process_subagent_spec(spec, default_model, default_tools, backend):
    # 1. 继承或覆盖模型
    subagent_model = spec.get("model", default_model)
    if isinstance(subagent_model, str):
        subagent_model = init_chat_model(subagent_model)
    
    # 2. 继承或覆盖工具
    subagent_tools = spec.get("tools", default_tools or [])
    
    # 3. 构建中间件栈（基础 + 用户自定义）
    subagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(...),
        AnthropicPromptCachingMiddleware(...),
        PatchToolCallsMiddleware(),
    ]
    
    # 添加技能中间件（如指定）
    if spec.get("skills"):
        subagent_middleware.append(
            SkillsMiddleware(backend=backend, sources=spec["skills"])
        )
    
    # 添加用户自定义中间件
    subagent_middleware.extend(spec.get("middleware", []))
    
    return {
        **spec,
        "model": subagent_model,
        "tools": subagent_tools,
        "middleware": subagent_middleware,
    }
```

### 4.6 步骤 6: 合并子代理列表

```python
# 通用子代理 + 用户子代理
all_subagents: list[SubAgent | CompiledSubAgent] = [
    general_purpose_spec,      # 必须放在第一个
    *processed_subagents
]
```

**注意**：通用子代理必须放在第一个，因为它的 `name="general-purpose"` 是 `task` 工具的默认选择。

### 4.7 步骤 7: 组装主 Agent 中间件栈

```python
deepagent_middleware: list[AgentMiddleware] = [
    # 1. 任务管理
    TodoListMiddleware(),
]

# 2. 记忆加载（可选）
if memory is not None:
    deepagent_middleware.append(
        MemoryMiddleware(backend=backend, sources=memory)
    )

# 3. 技能加载（可选）
if skills is not None:
    deepagent_middleware.append(
        SkillsMiddleware(backend=backend, sources=skills)
    )

# 4. 核心功能中间件
deepagent_middleware.extend([
    FilesystemMiddleware(backend=backend),
    SubAgentMiddleware(backend=backend, subagents=all_subagents),
    SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=summarization_defaults["trigger"],
        keep=summarization_defaults["keep"],
        truncate_args_settings=summarization_defaults["truncate_args_settings"],
    ),
    AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
    PatchToolCallsMiddleware(),
])

# 5. 用户自定义中间件
deepagent_middleware.extend(middleware)

# 6. 人工审核（可选）
if interrupt_on is not None:
    deepagent_middleware.append(
        HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
    )
```

### 4.8 步骤 8: 合并 System Prompt

```python
BASE_AGENT_PROMPT = """In order to complete the objective that the user asks of you, 
you have access to a number of standard tools."""

if system_prompt is None:
    final_system_prompt = BASE_AGENT_PROMPT
elif isinstance(system_prompt, SystemMessage):
    # SystemMessage：追加到 content_blocks
    new_content = [
        *system_prompt.content_blocks,
        {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
    ]
    final_system_prompt = SystemMessage(content=new_content)
else:
    # 字符串：简单拼接
    final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT
```

### 4.9 步骤 9: 创建 LangGraph

```python
return create_agent(
    model,
    system_prompt=final_system_prompt,
    tools=tools,
    middleware=deepagent_middleware,
    response_format=response_format,
    context_schema=context_schema,
    checkpointer=checkpointer,
    store=store,
    debug=debug,
    name=name,
    cache=cache,
).with_config({"recursion_limit": 1000})  # 默认递归限制
```

## 5. 创建流程时序图

```
用户调用
    │
    ▼
┌──────────────────────────────────────────────┐
│         create_deep_agent()                  │
│                                              │
│  1. resolve_model()                          │
│     ├──▶ None → Claude Sonnet 4.5            │
│     └──▶ str → init_chat_model()             │
│                                              │
│  2. resolve_backend()                        │
│     └──▶ None → StateBackend factory         │
│                                              │
│  3. compute_summarization_defaults()         │
│     ├──▶ has_profile → fraction strategy     │
│     └──▶ no_profile → fixed strategy         │
│                                              │
│  4. build_general_purpose_subagent()         │
│     ├──▶ TodoListMiddleware                  │
│     ├──▶ FilesystemMiddleware                │
│     ├──▶ SummarizationMiddleware             │
│     ├──▶ AnthropicPromptCachingMiddleware    │
│     ├──▶ PatchToolCallsMiddleware            │
│     └──▶ [SkillsMiddleware?]                 │
│                                              │
│  5. process_user_subagents()                 │
│     ├──▶ for each subagent:                  │
│     │       ├──▶ resolve model               │
│     │       ├──▶ resolve tools               │
│     │       └──▶ build middleware stack      │
│     └──▶ collect all subagents               │
│                                              │
│  6. merge_subagents()                        │
│     └──▶ [gp_subagent, ...user_subagents]    │
│                                              │
│  7. build_middleware_stack()                 │
│     ├──▶ TodoListMiddleware                  │
│     ├──▶ [MemoryMiddleware?]                 │
│     ├──▶ [SkillsMiddleware?]                 │
│     ├──▶ FilesystemMiddleware                │
│     ├──▶ SubAgentMiddleware                  │
│     ├──▶ SummarizationMiddleware             │
│     ├──▶ AnthropicPromptCachingMiddleware    │
│     ├──▶ PatchToolCallsMiddleware            │
│     ├──▶ [...user_middleware]                │
│     └──▶ [HumanInTheLoopMiddleware?]         │
│                                              │
│  8. merge_system_prompt()                    │
│     └──▶ user_prompt + BASE_AGENT_PROMPT     │
│                                              │
│  9. create_agent()                           │
│     └──▶ CompiledStateGraph                  │
│                                              │
└──────────────────────────────────────────────┘
    │
    ▼
返回 CompiledStateGraph
```

## 6. 设计决策分析

### 6.1 为什么使用工厂模式创建 Backend？

```python
# 方式 1：直接传入实例（问题：无法获取运行时信息）
backend = StateBackend(runtime=?)  # runtime 还不知道

# 方式 2：工厂函数（运行时创建，可获取 runtime）
def backend_factory(runtime: ToolRuntime) -> BackendProtocol:
    thread_id = runtime.config.get("configurable", {}).get("thread_id")
    return StoreBackend(runtime, namespace=("threads", thread_id))

middleware = FilesystemMiddleware(backend=backend_factory)
```

### 6.2 为什么子代理要继承主 Agent 配置？

- **一致性**：子代理和主 Agent 有相同的文件操作能力
- **简化配置**：用户只需在顶层配置一次
- **可覆盖**：需要时可以通过子代理参数覆盖

### 6.3 为什么中间件有固定顺序？

```
TodoList → Memory/Skills → Filesystem → SubAgent → Summarization → Caching → Patch

原因：
├─ TodoList 最先：任务规划优先
├─ Memory/Skills 在 Filesystem 前：上下文指导文件操作
├─ SubAgent 在 Summarization 前：子代理需要完整上下文
├─ Summarization 在 Caching 前：压缩后缓存更高效
└─ Patch 最后：修复所有潜在问题
```

### 6.4 为什么默认 recursion_limit=1000？

- 防止无限循环（Agent 反复调用工具）
- 足够大以支持复杂任务
- 可通过 `.with_config()` 覆盖

## 7. 配置示例

### 7.1 最小配置

```python
from deepagents import create_deep_agent

agent = create_deep_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

### 7.2 完整配置

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.tools import tool

@tool
def my_tool(query: str) -> str:
    """My custom tool."""
    return f"Result for {query}"

agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[my_tool],
    system_prompt="You are a helpful assistant.",
    backend=FilesystemBackend("/data"),
    skills=["/skills/research", "/skills/coding"],
    memory=["/project/AGENTS.md"],
    subagents=[
        {
            "name": "researcher",
            "description": "Research specialist",
            "system_prompt": "You are a research expert.",
            "model": "anthropic:claude-3-haiku",
            "tools": [web_search_tool],
        }
    ],
    interrupt_on={"edit_file": True},
    checkpointer=sqlite_saver,
)
```

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
