# SubAgentMiddleware 详解

## 1. 概述

`SubAgentMiddleware` 提供子代理委派功能，通过 `task` 工具允许 Agent 创建独立的子代理来处理特定任务。它位于 `deepagents/middleware/subagents.py`。

## 2. 核心类型

### 2.1 SubAgent - 子代理规范

```python
class SubAgent(TypedDict):
    """Specification for an agent."""
    
    name: str
    """Unique identifier for the subagent."""
    
    description: str
    """What this subagent does. The main agent uses this to decide when to delegate."""
    
    system_prompt: str
    """Instructions for the subagent."""
    
    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """Tools the subagent can use. If not specified, inherits from main agent."""
    
    model: NotRequired[str | BaseChatModel]
    """Override the main agent's model. Use 'provider:model-name' format."""
    
    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior."""
    
    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Configure human-in-the-loop for specific tools."""
    
    skills: NotRequired[list[str]]
    """Skill source paths for SkillsMiddleware."""
```

### 2.2 CompiledSubAgent - 预编译子代理

```python
class CompiledSubAgent(TypedDict):
    """A pre-compiled agent spec."""
    
    name: str
    """Unique identifier for the subagent."""
    
    description: str
    """What this subagent does."""
    
    runnable: Runnable
    """A custom agent implementation.
    
    Note: The runnable's state schema must include a 'messages' key.
    """
```

### 2.3 内部规范

```python
class _SubagentSpec(TypedDict):
    """Internal spec for building the task tool."""
    name: str
    description: str
    runnable: Runnable
```

## 3. 通用子代理模板

```python
DEFAULT_GENERAL_PURPOSE_DESCRIPTION = """
General-purpose agent for researching complex questions, searching for files 
and content, and executing multi-step tasks.
"""

DEFAULT_SUBAGENT_PROMPT = """
In order to complete the objective that the user asks of you, you have access 
to a number of standard tools.
"""

GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}
```

## 4. SubAgentMiddleware 类

### 4.1 初始化

```python
class SubAgentMiddleware(AgentMiddleware):
    """Middleware for providing subagents via a `task` tool.
    
    This middleware adds a `task` tool to the agent that can be used to invoke
    subagents. Subagents are useful for handling complex tasks that require 
    multiple steps or isolated context windows.
    """

    _VALID_DEPRECATED_KWARGS = frozenset({
        "default_model", "default_tools", "default_middleware",
        "default_interrupt_on", "general_purpose_agent",
    })

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        task_description: str | None = None,
        **deprecated_kwargs: Unpack[_DeprecatedKwargs],
    ) -> None:
        """Initialize the SubAgentMiddleware."""
        super().__init__()

        # 处理废弃参数
        default_model = deprecated_kwargs.get("default_model")
        default_tools = deprecated_kwargs.get("default_tools")
        default_middleware = deprecated_kwargs.get("default_middleware")
        default_interrupt_on = deprecated_kwargs.get("default_interrupt_on")
        general_purpose_agent = deprecated_kwargs.get("general_purpose_agent", True)

        # 检测使用哪个 API
        using_new_api = backend is not None
        using_old_api = default_model is not None

        if using_old_api and not using_new_api:
            # 旧 API - 从废弃参数构建子代理
            subagent_specs = _get_subagents_legacy(
                default_model=default_model,
                default_tools=default_tools or [],
                default_middleware=default_middleware,
                default_interrupt_on=default_interrupt_on,
                subagents=subagents or [],
                general_purpose_agent=general_purpose_agent,
            )
        elif using_new_api:
            if not subagents:
                raise ValueError("At least one subagent must be specified")
            self._backend = backend
            self._subagents = subagents
            subagent_specs = self._get_subagents()
        else:
            raise ValueError(
                "SubAgentMiddleware requires either `backend` (new API) "
                "or `default_model` (deprecated API)"
            )

        # 构建 task 工具
        task_tool = _build_task_tool(subagent_specs, task_description)

        # 构建系统提示词
        if system_prompt and subagent_specs:
            agents_desc = "\n".join(
                f"- {s['name']}: {s['description']}" for s in subagent_specs
            )
            self.system_prompt = system_prompt + "\n\nAvailable subagent types:\n" + agents_desc
        else:
            self.system_prompt = system_prompt

        self.tools = [task_tool]
```

### 4.2 构建子代理列表（新 API）

```python
def _get_subagents(self) -> list[_SubagentSpec]:
    """Create runnable agents from specs."""
    specs: list[_SubagentSpec] = []

    for spec in self._subagents:
        if "runnable" in spec:
            # CompiledSubAgent - 直接使用
            compiled = cast("CompiledSubAgent", spec)
            specs.append({
                "name": compiled["name"],
                "description": compiled["description"],
                "runnable": compiled["runnable"]
            })
            continue

        # SubAgent - 验证必需字段
        if "model" not in spec:
            raise ValueError(f"SubAgent '{spec['name']}' must specify 'model'")
        if "tools" not in spec:
            raise ValueError(f"SubAgent '{spec['name']}' must specify 'tools'")

        # 解析模型
        model = spec["model"]
        if isinstance(model, str):
            model = init_chat_model(model)

        # 使用提供的中间件
        middleware: list[AgentMiddleware] = list(spec.get("middleware", []))

        # 添加 HITL（如果配置）
        interrupt_on = spec.get("interrupt_on")
        if interrupt_on:
            middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        # 创建子代理
        specs.append({
            "name": spec["name"],
            "description": spec["description"],
            "runnable": create_agent(
                model,
                system_prompt=spec["system_prompt"],
                tools=spec["tools"],
                middleware=middleware,
                name=spec["name"],
            ),
        })

    return specs
```

## 5. task 工具构建

```python
def _build_task_tool(
    subagents: list[_SubagentSpec],
    task_description: str | None = None,
) -> BaseTool:
    """Create a task tool from pre-built subagent graphs."""
    
    # 构建子代理字典
    subagent_graphs: dict[str, Runnable] = {
        spec["name"]: spec["runnable"] for spec in subagents
    }
    
    # 构建子代理描述
    subagent_description_str = "\n".join(
        f"- {s['name']}: {s['description']}" for s in subagents
    )

    # 构建工具描述
    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(
            available_agents=subagent_description_str
        )
    elif "{available_agents}" in task_description:
        description = task_description.format(
            available_agents=subagent_description_str
        )
    else:
        description = task_description

    def _return_command_with_state_update(
        result: dict,
        tool_call_id: str
    ) -> Command:
        """Convert subagent result to Command with state update."""
        if "messages" not in result:
            raise ValueError(
                "CompiledSubAgent must return a state containing a 'messages' key."
            )

        # 过滤不需要的 state 键
        state_update = {
            k: v for k, v in result.items()
            if k not in _EXCLUDED_STATE_KEYS
        }
        
        # 提取最后一条消息
        message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
        
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(
        subagent_type: str,
        description: str,
        runtime: ToolRuntime
    ) -> tuple[Runnable, dict]:
        """Prepare state for subagent invocation."""
        subagent = subagent_graphs[subagent_type]
        
        # 创建新的 state dict，过滤敏感键
        subagent_state = {
            k: v for k, v in runtime.state.items()
            if k not in _EXCLUDED_STATE_KEYS
        }
        subagent_state["messages"] = [HumanMessage(content=description)]
        
        return subagent, subagent_state

    def task(
        description: Annotated[str, "Task description"],
        subagent_type: Annotated[str, "Subagent type"],
        runtime: ToolRuntime,
    ) -> str | Command:
        """Execute task with subagent."""
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"Subagent {subagent_type} does not exist. Allowed: {allowed_types}"

        subagent, subagent_state = _validate_and_prepare_state(
            subagent_type, description, runtime
        )
        
        result = subagent.invoke(subagent_state)
        
        if not runtime.tool_call_id:
            raise ValueError("Tool call ID is required")
            
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: Annotated[str, "Task description"],
        subagent_type: Annotated[str, "Subagent type"],
        runtime: ToolRuntime,
    ) -> str | Command:
        """Async version of task."""
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"Subagent {subagent_type} does not exist. Allowed: {allowed_types}"

        subagent, subagent_state = _validate_and_prepare_state(
            subagent_type, description, runtime
        )
        
        result = await subagent.ainvoke(subagent_state)
        
        if not runtime.tool_call_id:
            raise ValueError("Tool call ID is required")
            
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
    )
```

## 6. State 过滤

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

## 7. 系统提示词

```python
TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle 
isolated tasks. These agents are ephemeral — they live only for the duration 
of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage
- When sandboxing improves reliability (e.g. code execution, structured searches)
- When you only care about the output of the subagent, not the intermediate steps

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching

## Important Task Tool Usage Notes:
- Launch multiple agents concurrently whenever possible
- Each agent invocation is stateless - you cannot send additional messages
- Clearly tell the agent whether you expect it to create content, perform analysis, or just do research"""
```

## 8. 使用示例

### 8.1 基本使用（create_deep_agent 自动创建）

```python
from deepagents import create_deep_agent

# general-purpose 子代理会自动创建
agent = create_deep_agent()
```

### 8.2 自定义子代理

```python
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent

subagents: list[SubAgent] = [
    {
        "name": "researcher",
        "description": "Research specialist for deep investigation",
        "system_prompt": "You are a research expert. Conduct thorough research...",
        "model": "openai:gpt-4o",
        "tools": [web_search_tool, read_file_tool],
    },
    {
        "name": "coder",
        "description": "Code specialist for implementation tasks",
        "system_prompt": "You are an expert programmer...",
        "model": "anthropic:claude-3-5-sonnet",
        "tools": [file_tools],
    },
]

agent = create_deep_agent(subagents=subagents)
```

### 8.3 使用 CompiledSubAgent

```python
from deepagents.middleware.subagents import CompiledSubAgent
from langgraph.graph import StateGraph

# 预编译子代理
workflow = StateGraph(...)
# ... 配置 workflow ...
compiled = workflow.compile()

subagents: list[CompiledSubAgent] = [
    {
        "name": "custom-agent",
        "description": "Custom pre-compiled agent",
        "runnable": compiled,
    }
]

agent = create_deep_agent(subagents=subagents)
```

## 9. 设计要点

### 9.1 上下文隔离
- 子代理有独立的 `messages` 列表
- 排除敏感 state 键
- 子代理无法访问父代理的完整历史

### 9.2 结果传递
- 子代理的最后一条消息作为 ToolMessage 返回
- 可以传递部分 state 更新
- 父代理负责整合结果

### 9.3 并行执行
- 多个子代理可以同时启动
- 各自独立执行，互不干扰
- 结果分别返回

## 10. 注意事项

1. **Tool Call ID**：子代理调用必须有 `tool_call_id`
2. **Messages 键**：CompiledSubAgent 必须返回包含 `messages` 的 state
3. **State 继承**：子代理可以选择性继承父 state
4. **错误处理**：子代理类型不存在时返回错误消息

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
