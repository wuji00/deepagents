# 15 - MemoryMiddleware 详解

## 1. 概述

`MemoryMiddleware` 实现了从 `AGENTS.md` 文件加载持久化记忆（Memory）的功能。与 Skills（按需工作流）不同，Memory 提供**始终加载**的持久上下文，帮助AI代理更好地理解项目背景和用户偏好。

## 2. 核心概念

### 2.1 AGENTS.md 规范

AGENTS.md 文件是标准 Markdown 格式，常见章节包括：
- 项目概览
- 构建/测试命令
- 代码风格指南
- 架构说明

### 2.2 Memory 与 Skill 的区别

| 特性 | Memory (AGENTS.md) | Skill (SKILL.md) |
|------|-------------------|------------------|
| 加载时机 | 始终加载 | 按需读取 |
| 内容类型 | 持久上下文、偏好设置 | 任务工作流、最佳实践 |
| 更新频率 | 低（学习积累） | 中（技能开发） |
| 格式 | 纯 Markdown | YAML Frontmatter + Markdown |

## 3. 核心数据类型

### 3.1 MemoryState 定义

```python
class MemoryState(AgentState):
    """MemoryMiddleware 的状态模式。"""
    
    memory_contents: NotRequired[
        Annotated[dict[str, str], PrivateStateAttr]
    ]
    """源路径到内容的映射，标记为私有状态。"""


class MemoryStateUpdate(TypedDict):
    """MemoryMiddleware 的状态更新。"""
    
    memory_contents: dict[str, str]
```

### 3.2 状态设计说明

```python
# 使用 dict[str, str] 而非 list 的原因：
# 1. 保留源路径信息，便于追踪记忆来源
# 2. 支持多源合并时去重和排序
# 3. 方便调试和日志记录

memory_contents = {
    "~/.deepagents/AGENTS.md": "# User Preferences...",
    "./.deepagents/AGENTS.md": "# Project Guidelines...",
}
```

## 4. 记忆加载机制

### 4.1 同步加载

```python
def _load_memory_from_backend_sync(
    self,
    backend: BackendProtocol,
    path: str,
) -> str | None:
    """从后端同步加载单个记忆文件。"""
    results = backend.download_files([path])
    
    # 严格检查：必须返回恰好一个结果
    if len(results) != 1:
        raise AssertionError(
            f"Expected 1 response for path {path}, got {len(results)}"
        )
    
    response = results[0]
    
    if response.error is not None:
        # file_not_found 是预期的，静默跳过
        if response.error == "file_not_found":
            return None
        # 其他错误抛出异常
        raise ValueError(f"Failed to download {path}: {response.error}")
    
    if response.content is not None:
        return response.content.decode("utf-8")
    
    return None
```

### 4.2 异步加载

```python
async def _load_memory_from_backend(
    self,
    backend: BackendProtocol,
    path: str,
) -> str | None:
    """异步版本使用 await 调用 adownload_files。"""
    results = await backend.adownload_files([path])
    # 其余逻辑与同步版本相同
    ...
```

### 4.3 批量加载流程

```python
def before_agent(self, state, runtime, config) -> MemoryStateUpdate | None:
    """在代理执行前加载所有记忆源。
    
    特点：
    - 只加载一次（检查 state 中是否已存在）
    - 多源按配置顺序加载
    - file_not_found 错误静默处理
    """
    # 跳过已加载的情况
    if "memory_contents" in state:
        return None
    
    backend = self._get_backend(state, runtime, config)
    contents: dict[str, str] = {}
    
    for path in self.sources:
        content = self._load_memory_from_backend_sync(backend, path)
        if content:
            contents[path] = content
            logger.debug(f"Loaded memory from: {path}")
    
    return MemoryStateUpdate(memory_contents=contents)
```

## 5. 系统提示注入

### 5.1 记忆格式化

```python
def _format_agent_memory(self, contents: dict[str, str]) -> str:
    """将记忆内容格式化为系统提示。
    
    输出格式：
    <agent_memory>
    {path}
    {content}
    
    {path}
    {content}
    </agent_memory>
    
    <memory_guidelines>
    ...使用指南...
    </memory_guidelines>
    """
    if not contents:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")
    
    # 按配置顺序组合各源内容
    sections = []
    for path in self.sources:
        if contents.get(path):
            sections.append(f"{path}\n{contents[path]}")
    
    if not sections:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")
    
    memory_body = "\n\n".join(sections)
    return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)
```

### 5.2 记忆使用指南

```python
MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem. As you learn from your interactions with the user, you can save new knowledge by calling the `edit_file` tool.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions with the user. These learnings can be implicit or explicit. This means that in the future, you will remember this important information.
    - When you need to remember something, updating memory must be your FIRST, IMMEDIATE action - before responding to the user, before calling other tools, before doing anything else. Just update memory immediately.
    - When user says something is better/worse, capture WHY and encode it as a pattern.
    - Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions.
    - A great opportunity to update your memories is when the user interrupts a tool call and provides feedback. You should update your memories immediately before revising the tool call.
    - Look for the underlying principle behind corrections, not just the specific mistake.
    - The user might not explicitly ask you to remember something, but if they provide information that is useful for future use, you should update your memories immediately.

    **Asking for information:**
    - If you lack context to perform an action (e.g. send a Slack DM, requires a user ID/email) you should explicitly ask the user for this information.
    - It is preferred for you to ask for information, don't assume anything that you do not know!
    - When the user provides information that is useful for future use, you should update your memories immediately.

    **When to update memories:**
    - When the user explicitly asks you to remember something (e.g., "remember my email", "save this preference")
    - When the user describes your role or how you should behave (e.g., "you are a web researcher", "always do X")
    - When the user gives feedback on your work - capture what was wrong and how to improve
    - When the user provides information required for tool use (e.g., slack channel ID, email addresses)
    - When the user provides context useful for future tasks, such as how to use tools, or which actions to take in a particular situation
    - When you discover new patterns or preferences (coding styles, conventions, workflows)

    **When to NOT update memories:**
    - When the information is temporary or transient (e.g., "I'm running late", "I'm on my phone right now")
    - When the information is a one-time task request (e.g., "Find me a recipe", "What's 25 * 4?")
    - When the information is a simple question that doesn't reveal lasting preferences (e.g., "What day is it?", "Can you explain X?")
    - When the information is an acknowledgment or small talk (e.g., "Sounds good!", "Hello", "Thanks for that")
    - When the information is stale or irrelevant in future conversations
    - Never store API keys, access tokens, passwords, or any other credentials in any file, memory, or system prompt.
    - If the user asks where to put API keys or provides an API key, do NOT echo or save it.

    **Examples:**
    Example 1 (remembering user information):
    User: Can you connect to my google account?
    Agent: Sure, I'll connect to your google account, what's your google account email?
    User: john@example.com
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user's google account email is john@example.com

    Example 2 (remembering implicit user preferences):
    User: Can you write me an example for creating a deep agent in LangChain?
    Agent: Sure, I'll write you an example for creating a deep agent in LangChain <example code in Python>
    User: Can you do this in JavaScript
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user prefers to get LangChaincode examples in JavaScript
    Agent: Sure, here is the JavaScript example<example code in JavaScript>

    Example 3 (do not remember transient information):
    User: I'm going to play basketball tonight so I will be offline for a few hours.
    Agent: Okay I'll add a black to your calendar.
    Tool Call: create_calendar_event(...) -> just calls a tool, does not commit anything to memory, as it is transient information
</memory_guidelines>
"""
```

## 6. 记忆更新指南设计

### 6.1 何时更新记忆

指南中明确列出了应该更新记忆的场景：

1. **用户明确要求记住** - "remember my email"
2. **用户描述角色期望** - "you are a web researcher"
3. **用户反馈** - 捕获错误原因和改进方法
4. **工具使用信息** - 渠道ID、邮箱地址等
5. **任务上下文** - 工具使用方法、特定情况下的操作
6. **发现的新模式** - 编码风格、约定、工作流

### 6.2 何时不更新记忆

1. **临时信息** - "I'm running late"
2. **一次性任务** - "Find me a recipe"
3. **简单问题** - "What day is it?"
4. **寒暄** - "Hello", "Thanks"
5. **过时信息** - 未来对话中不再相关
6. **敏感信息** - API密钥、密码（**绝不要存储**）

### 6.3 设计示例

指南包含三个详细示例，教导代理：
- **示例1**：存储用户信息（显式请求）
- **示例2**：学习隐式偏好（语言选择）
- **示例3**：忽略临时信息（活动安排）

## 7. 请求修改流程

```python
def modify_request(self, request: ModelRequest) -> ModelRequest:
    """将记忆内容注入系统消息。"""
    
    # 从状态获取记忆内容
    contents = request.state.get("memory_contents", {})
    
    # 格式化记忆（包含使用指南）
    agent_memory = self._format_agent_memory(contents)
    
    # 追加到系统消息
    new_system_message = append_to_system_message(
        request.system_message, 
        agent_memory
    )
    
    return request.override(system_message=new_system_message)
```

## 8. 使用示例

### 8.1 基本用法

```python
from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend

# 注意：FilesystemBackend 允许读写整个文件系统
# 建议在沙箱中使用或添加 HIL 审批
backend = FilesystemBackend(root_dir="/")

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        "~/.deepagents/AGENTS.md",      # 用户级记忆
        "./.deepagents/AGENTS.md",      # 项目级记忆
    ],
)

agent = create_deep_agent(middleware=[middleware])
```

### 8.2 使用 StateBackend

```python
from deepagents.backends.state import StateBackend

middleware = MemoryMiddleware(
    backend=lambda rt: StateBackend(rt),
    sources=["/memory/AGENTS.md"],
)
```

## 9. 关键设计决策

### 9.1 为什么使用 PrivateStateAttr？

```python
memory_contents: NotRequired[
    Annotated[dict[str, str], PrivateStateAttr]
]
```

- **不传播给父代理**：记忆是代理私有的
- **避免状态污染**：子代理的记忆不应影响父代理
- **按需加载**：只在需要时加载，不持久化

### 9.2 为什么静默处理 file_not_found？

```python
if response.error == "file_not_found":
    return None  # 静默跳过
```

记忆文件是**可选的**，允许优雅降级：
- 用户可以逐步添加记忆文件
- 不需要记忆的场景不会报错
- 多源配置中部分源可能不存在

### 9.3 为什么优先更新记忆？

```
When you need to remember something, updating memory must be your 
FIRST, IMMEDIATE action - before responding to the user, before 
calling other tools, before doing anything else.
```

确保关键信息不会丢失：
- 代理可能在后续步骤中崩溃
- 长对话可能触发摘要，丢失细节
- 用户反馈如果不立即记录可能遗忘

## 10. 与 create_deep_agent 集成

```python
def create_deep_agent(
    memory: list[str] | None = None,  # AGENTS.md 路径列表
    ...
):
    deepagent_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]
    
    if memory is not None:
        deepagent_middleware.append(
            MemoryMiddleware(backend=backend, sources=memory)
        )
    
    # MemoryMiddleware 在 FilesystemMiddleware 之前
    # 确保记忆内容在系统提示中先于文件工具描述
```

## 11. 总结

MemoryMiddleware 实现了智能的记忆管理系统：

1. **多源支持**：可从多个 AGENTS.md 文件加载
2. **可选文件**：file_not_found 静默处理，支持渐进式配置
3. **状态隔离**：使用 PrivateStateAttr 防止状态污染
4. **使用指南**：详细的指南教导代理何时/如何更新记忆
5. **安全第一**：明确禁止存储敏感信息
6. **优先级高**：指导代理优先更新记忆，防止信息丢失
