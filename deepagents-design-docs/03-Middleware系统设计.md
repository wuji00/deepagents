# Middleware 系统设计文档

## 1. 概述

Middleware 系统是 DeepAgents 的功能扩展机制，基于 LangChain Agents 的 Middleware 架构。每个 Middleware 负责特定的横切关注点（cross-cutting concern），通过组合多个 Middleware 构建完整的 Agent 功能。

### 1.1 设计目标

- **关注点分离**：每个 Middleware 负责单一功能领域
- **可组合性**：Middleware 可以按需堆叠和组合
- **可插拔**：新功能通过添加 Middleware 实现，不影响现有代码
- **顺序执行**：Middleware 按照定义的顺序依次处理请求
- **状态管理**：通过 State 共享数据，通过 Reducer 合并更新

### 1.2 架构位置

```
┌─────────────────────────────────────────┐
│           create_deep_agent()           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Middleware Chain (链式调用)      │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐ │
│  │ MW1 │──▶│ MW2 │──▶│ MW3 │──▶│ MW4 │ │
│  └─────┘   └─────┘   └─────┘   └─────┘ │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│            LangGraph Runtime            │
└─────────────────────────────────────────┘
```

## 2. 核心概念

### 2.1 AgentMiddleware 基类

**来源**：`langchain.agents.middleware.types.AgentMiddleware`

所有 DeepAgents Middleware 都继承自 `AgentMiddleware` 基类：

```python
class AgentMiddleware:
    """Middleware 基类定义生命周期钩子"""
    
    # 可选：定义 State 结构
    state_schema: type[AgentState] | None = None
    
    # Middleware 提供的工具
    tools: list[BaseTool] = []
    
    # 注入到 System Prompt 的指令
    system_prompt: str | None = None
    
    # ========== 生命周期钩子 ==========
    
    def before_agent(self, state, runtime, config) -> dict | None:
        """Agent 执行前调用，用于初始化 State"""
        pass
    
    def before_model(self, state, runtime, config) -> dict | None:
        """调用 LLM 前调用，用于修改消息/触发摘要"""
        pass
    
    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """修改模型请求（添加 tools、修改 system_message）"""
        return request
    
    def wrap_model_call(self, request, handler) -> ModelResponse:
        """包装模型调用，可以修改请求和响应"""
        return handler(request)
    
    def wrap_tool_call(self, request, handler) -> ToolMessage | Command:
        """包装工具调用，可以处理工具结果"""
        return handler(request)
```

### 2.2 执行顺序

```
Agent.invoke()
    │
    ├──▶ before_agent()  # 所有 Middleware，按注册顺序
    │
    ├──▶ modify_request() # 所有 Middleware，按注册顺序
    │
    ├──▶ wrap_model_call() # 嵌套调用，从外到内
    │       │
    │       ├──▶ Middleware1.wrap_model_call()
    │       │       └──▶ Middleware2.wrap_model_call()
    │       │               └──▶ ... 实际模型调用
    │       │
    │       └──▶ 响应从内部返回，从里到外
    │
    ├──▶ 模型生成 Tool Calls
    │
    ├──▶ wrap_tool_call() # 每个工具调用
    │       │
    │       ├──▶ Middleware1.wrap_tool_call()
    │       │       └──▶ Middleware2.wrap_tool_call()
    │       │               └──▶ ... 实际工具执行
    │       │
    │       └──▶ 结果从内部返回，从里到外
    │
    └──▶ before_model() # 下一个迭代开始前（Summarization）
```

## 3. DeepAgents 核心 Middleware

### 3.1 TodoListMiddleware

**文件位置**：`langchain.agents.middleware.TodoListMiddleware`

**功能**：任务规划与进度管理

#### 提供的工具
- `write_todos` - 创建/更新待办事项列表
- `read_todos` - 读取当前待办事项

#### State 结构
```python
class TodoState(AgentState):
    todos: Annotated[list[TodoItem], add_todo_item]

class TodoItem(TypedDict):
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"]
    created_at: str
```

#### 工作原理
```
用户输入
    │
    ▼
Agent 调用 write_todos 创建任务
    │
    ▼
任务存储在 State.todos 中
    │
    ▼
后续迭代可以 read_todos 查看进度
    │
    ▼
任务完成时更新状态为 completed
```

### 3.2 FilesystemMiddleware

**文件位置**：`deepagents/middleware/filesystem.py`

**功能**：文件操作和可选的命令执行

#### 提供的工具
| 工具 | 功能 |
|------|------|
| `ls` | 列出目录内容 |
| `read_file` | 读取文件（支持 offset/limit 分页） |
| `write_file` | 写入新文件 |
| `edit_file` | 编辑文件（字符串替换） |
| `glob` | 文件模式匹配 |
| `grep` | 文件内容搜索 |
| `execute` | 执行 shell 命令（需 Sandbox 后端） |

#### State 结构
```python
class FilesystemState(AgentState):
    files: Annotated[dict[str, FileData], _file_data_reducer]

class FileData(TypedDict):
    content: list[str]        # 按行存储
    created_at: str           # ISO 8601 时间戳
    modified_at: str          # ISO 8601 时间戳
```

#### 核心机制

**1. 路径安全验证**
```python
def _validate_path(path: str, allowed_prefixes: Sequence[str] | None = None) -> str:
    # 防止目录遍历攻击
    if ".." in path or path.startswith("~"):
        raise ValueError("Path traversal not allowed")
    
    # 拒绝 Windows 绝对路径
    if re.match(r"^[a-zA-Z]:", path):
        raise ValueError("Windows absolute paths are not supported")
    
    # 规范化路径
    normalized = os.path.normpath(path).replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    
    return normalized
```

**2. 大内容处理**
```python
# 超过 token 限制的工具结果自动转存
NUM_CHARS_PER_TOKEN = 4
TOOL_TOKEN_LIMIT = 20000

# 处理流程
if len(content) > NUM_CHARS_PER_TOKEN * TOOL_TOKEN_LIMIT:
    # 1. 写入 Backend
    file_path = f"/large_tool_results/{tool_call_id}"
    backend.write(file_path, content)
    
    # 2. 返回预览
    return TOO_LARGE_TOOL_MSG.format(
        tool_call_id=tool_call_id,
        file_path=file_path,
        content_sample=create_preview(content)
    )
```

### 3.3 SubAgentMiddleware

**文件位置**：`deepagents/middleware/subagents.py`

**功能**：子代理委派与任务并行化

#### 核心概念

**SubAgent**：子代理定义
```python
class SubAgent(TypedDict):
    name: str                   # 子代理标识
    description: str            # 用途描述（LLM 用）
    system_prompt: str          # 系统提示
    tools: NotRequired[Sequence[BaseTool]]  # 专属工具
    model: NotRequired[str | BaseChatModel] # 专用模型
    middleware: NotRequired[list[AgentMiddleware]]  # 专属中间件
    skills: NotRequired[list[str]]  # 技能路径
    interrupt_on: NotRequired[dict] # HITL 配置
```

**CompiledSubAgent**：预编译子代理
```python
class CompiledSubAgent(TypedDict):
    name: str
    description: str
    runnable: Runnable          # 预编译的 LangGraph
```

#### 提供的工具
- `task` - 委派任务给子代理

#### 任务调用流程
```
父 Agent 调用 task 工具
        │
        ├──▶ description: 任务描述
        ├──▶ subagent_type: 子代理类型
        │
        ▼
┌───────────────────┐
│ 查找对应 SubAgent │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 准备子代理 State   │
│ - 过滤敏感 State   │
│ - 设置任务消息     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  执行子代理        │
│  (同步/异步)      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 提取最终消息      │
│ 作为 ToolMessage  │
│ 返回给父 Agent    │
└───────────────────┘
```

#### 状态隔离策略
```python
# 从父 State 中排除的键（不传递给子代理）
_EXCLUDED_STATE_KEYS = {
    "messages",           # 重新设置
    "todos",              # 子代理有自己的任务
    "structured_response", # 结构化响应
    "skills_metadata",    # 子代理自己加载技能
    "memory_contents",    # 子代理自己加载记忆
}
```

### 3.4 SummarizationMiddleware

**文件位置**：`deepagents/middleware/summarization.py`

**功能**：对话历史摘要和上下文压缩

#### 触发条件
```python
# 三种触发策略
ContextSize = tuple[Literal["messages", "tokens", "fraction"], int | float]

# 示例：
("messages", 50)      # 消息数超过 50 条
("tokens", 100000)    # Token 数超过 10 万
("fraction", 0.85)    # 超过上下文窗口 85%
```

#### 处理流程
```
消息历史
    │
    ▼
检查是否触发摘要
    │
    ├──▶ 未触发 ──▶ 返回 None
    │
    └──▶ 触发
            │
            ▼
    ┌───────────────────┐
    │  1. 参数截断      │
    │  大工具参数截断   │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  2. 内容卸载      │
    │  写入 Backend     │
    │  /conversation_history/{thread_id}.md
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  3. 生成摘要      │
    │  使用轻量级模型   │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  4. 替换消息      │
    │  旧消息 → 摘要    │
    └─────────┬─────────┘
              │
              ▼
    返回新消息列表给模型
```

#### 参数截断策略
```python
class TruncateArgsSettings(TypedDict):
    trigger: ContextSize | None    # 触发阈值
    keep: ContextSize              # 保留策略
    max_length: int                # 参数最大长度（默认 2000）
    truncation_text: str           # 截断提示文本

# 只截断 write_file/edit_file 的大参数
if tool_call["name"] in {"write_file", "edit_file"}:
    for key, value in args.items():
        if isinstance(value, str) and len(value) > max_length:
            truncated_args[key] = value[:20] + "...(argument truncated)"
```

### 3.5 SkillsMiddleware

**文件位置**：`deepagents/middleware/skills.py`

**功能**：技能系统，动态加载能力模块

#### Skill 结构
```
/skills/
├── skill-a/
│   ├── SKILL.md          # 技能定义和说明
│   ├── agents.yaml       # (可选) 子代理定义
│   └── resources/        # (可选) 资源文件
└── skill-b/
    └── SKILL.md
```

#### Skill 元数据
```python
class SkillMetadata(TypedDict):
    name: str
    description: str
    version: str
    author: str
    tools: list[str]          # 提供的工具
    agents: list[dict]        # 子代理定义
```

#### 加载流程
```
before_agent()
    │
    ▼
遍历 skills 路径
    │
    ├──▶ 解析 SKILL.md
    ├──▶ 提取元数据
    ├──▶ 加载 agents.yaml
    │
    ▼
合并到 State.skills_metadata
    │
    ▼
注入到 System Prompt
"""
## Available Skills

- skill-a: Description of skill A
- skill-b: Description of skill B
"""
```

### 3.6 MemoryMiddleware

**文件位置**：`deepagents/middleware/memory.py`

**功能**：记忆加载，从 AGENTS.md 文件注入上下文

#### 工作原理
```python
class MemoryMiddleware(AgentMiddleware):
    def __init__(self, backend: BackendProtocol, sources: list[str]):
        # sources: ["/memory/AGENTS.md", "/project/AGENTS.md"]
        self.sources = sources
    
    def before_agent(self, state, runtime, config):
        # 从 Backend 读取记忆文件
        memory_contents = {}
        for source in self.sources:
            content = backend.read(source)
            memory_contents[source] = content
        
        return {"memory_contents": memory_contents}
    
    def modify_request(self, request: ModelRequest):
        # 将记忆内容注入 System Message
        memory_text = format_memory(request.state["memory_contents"])
        new_system = append_to_system_message(
            request.system_message, 
            f"## Project Context\n{memory_text}"
        )
        return request.override(system_message=new_system)
```

### 3.7 PatchToolCallsMiddleware

**文件位置**：`deepagents/middleware/patch_tool_calls.py`

**功能**：修复工具调用 ID 格式问题

#### 背景问题
某些模型返回的 `tool_call_id` 格式不规范，可能导致后续消息匹配失败。

#### 修复策略
```python
def before_agent(self, state, runtime):
    messages = state["messages"]
    
    # 修复工具调用 ID 格式
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc["id"] = sanitize_tool_call_id(tc["id"])
        
        if isinstance(msg, ToolMessage):
            msg.tool_call_id = sanitize_tool_call_id(msg.tool_call_id)
```

## 4. Middleware 组合策略

### 4.1 标准中间件栈（create_deep_agent）

```python
# 通用子代理中间件栈
gp_middleware = [
    TodoListMiddleware(),                    # 1. 任务管理
    FilesystemMiddleware(backend=backend),   # 2. 文件操作
    SummarizationMiddleware(                 # 3. 上下文压缩
        model=model,
        backend=backend,
        trigger=summarization_defaults["trigger"],
        keep=summarization_defaults["keep"],
    ),
    AnthropicPromptCachingMiddleware(),      # 4. Prompt 缓存
    PatchToolCallsMiddleware(),              # 5. Tool ID 修复
]

# 主 Agent 中间件栈
deepagent_middleware = [
    TodoListMiddleware(),                    # 1. 任务管理
    MemoryMiddleware(backend=backend, sources=memory),  # 2. 记忆（可选）
    SkillsMiddleware(backend=backend, sources=skills),  # 3. 技能（可选）
    FilesystemMiddleware(backend=backend),   # 4. 文件操作
    SubAgentMiddleware(                      # 5. 子代理
        backend=backend,
        subagents=all_subagents,
    ),
    SummarizationMiddleware(...),            # 6. 上下文压缩
    AnthropicPromptCachingMiddleware(),      # 7. Prompt 缓存
    PatchToolCallsMiddleware(),              # 8. Tool ID 修复
    # + 用户自定义 middleware
    # + HumanInTheLoopMiddleware（如启用）
]
```

### 4.2 执行顺序的重要性

```
正确顺序：

TodoList → Memory/Skills → Filesystem → SubAgent → Summarization → Caching → Patch

原因：
- TodoList 最先：任务规划应该在所有操作之前
- Memory/Skills 在 Filesystem 之前：提供上下文指导文件操作
- SubAgent 在 Summarization 之前：子代理需要完整上下文
- Summarization 在 Caching 之前：压缩后再缓存更高效
- Patch 最后：修复所有中间件可能产生的问题
```

## 5. 自定义 Middleware 开发

### 5.1 基础模板

```python
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict, Annotated

class MyMiddlewareState(AgentState):
    my_data: Annotated[dict, my_reducer]

class MyMiddleware(AgentMiddleware):
    state_schema = MyMiddlewareState
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.tools = [self._create_my_tool()]
        self.system_prompt = "## My Middleware\nInstructions..."
    
    def _create_my_tool(self) -> BaseTool:
        @tool
        def my_tool(param: str) -> str:
            """Tool description."""
            return f"Result: {param}"
        return my_tool
    
    def before_agent(self, state, runtime, config):
        # 初始化 State
        return {"my_data": {}}
    
    def modify_request(self, request):
        # 修改请求
        return request
    
    def wrap_model_call(self, request, handler):
        # 包装模型调用
        response = handler(request)
        return response
```

### 5.2 使用自定义 Middleware

```python
from deepagents import create_deep_agent
from my_middleware import MyMiddleware

agent = create_deep_agent(
    middleware=[
        MyMiddleware(config={"key": "value"})
    ]
)
```

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
