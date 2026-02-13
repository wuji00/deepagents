# 16 - PatchToolCallsMiddleware 详解

## 1. 概述

`PatchToolCallsMiddleware` 是一个轻量级中间件，用于修复消息历史中的**悬空工具调用（dangling tool calls）**。当 AI 消息中包含工具调用请求，但后续没有对应的工具响应消息时，该中间件会自动插入占位响应，确保消息序列的完整性。

## 2. 问题场景

### 2.1 悬空工具调用是如何产生的？

```python
# 场景1：用户中断
AI: 我将为您搜索相关信息。  
    [工具调用: search_web(query="...")]  
    ← 用户在此处发送新消息中断
User: 等等，先别搜索

# 场景2：流式响应中断
AI: 让我分析这个文件。  
    [工具调用: read_file(path="...")]  
    ← 网络中断或超时

# 场景3：错误处理
AI: 我来执行这个命令。  
    [工具调用: execute(command="...")]  
    ← 执行器抛出异常，未生成 ToolMessage
```

### 2.2 为什么需要修复？

许多 LLM 提供商要求消息序列满足以下约束：
- 每个 `AIMessage` 中的 `tool_calls` 必须有对应的 `ToolMessage` 响应
- 工具调用的 `tool_call_id` 必须在后续消息中被回应
- 违反此约束会导致 API 错误或非预期行为

## 3. 核心实现

### 3.1 完整代码

```python
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


class PatchToolCallsMiddleware(AgentMiddleware):
    """Middleware to patch dangling tool calls in the messages history."""

    def before_agent(
        self, 
        state: AgentState, 
        runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Before the agent runs, handle dangling tool calls from any AIMessage."""
        messages = state["messages"]
        
        # 空消息列表，无需处理
        if not messages or len(messages) == 0:
            return None

        patched_messages = []
        
        # 遍历消息，查找悬空工具调用
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            
            # 只处理 AI 消息且有工具调用的情况
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 在后续消息中查找对应的 ToolMessage
                    corresponding_tool_msg = next(
                        (
                            msg for msg in messages[i:]
                            if msg.type == "tool" 
                            and msg.tool_call_id == tool_call["id"]
                        ),
                        None,
                    )
                    
                    # 未找到对应响应，创建占位消息
                    if corresponding_tool_msg is None:
                        tool_msg_content = (
                            f"Tool call {tool_call['name']} with id "
                            f"{tool_call['id']} was cancelled - another "
                            f"message came in before it could be completed."
                        )
                        
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg_content,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        # 使用 Overwrite 完全替换消息列表
        return {"messages": Overwrite(patched_messages)}
```

## 4. 算法详解

### 4.1 处理流程

```
输入: messages = [msg1, msg2, msg3, ...]
输出: 修补后的消息列表（可能包含新增的 ToolMessage）

for i, msg in enumerate(messages):
    1. 将当前消息添加到结果列表
    
    2. 如果当前消息是 AIMessage 且有 tool_calls:
        for each tool_call in msg.tool_calls:
            a. 在 messages[i:] 中搜索 tool_call_id 匹配的 ToolMessage
            b. 如果未找到:
               - 创建占位 ToolMessage
               - 插入到结果列表中
```

### 4.2 时间复杂度

- **最坏情况**: O(n × m)，其中 n 是消息数，m 是平均工具调用数
- **实际场景**: m 通常很小（1-3个工具调用），接近 O(n)
- **优化**: 使用 `next()` 和生成器表达式，在找到匹配时立即终止搜索

### 4.3 示例演示

```python
# 输入消息序列
messages = [
    HumanMessage(content="Search for Python docs"),
    AIMessage(
        content="I'll search for you",
        tool_calls=[
            {"id": "call_1", "name": "search", "args": {"q": "Python"}},
            {"id": "call_2", "name": "search", "args": {"q": "docs"}},
        ]
    ),
    # call_1 的响应缺失！
    ToolMessage(
        content="Found docs",
        tool_call_id="call_2",
        name="search"
    ),
    HumanMessage(content="Thanks"),
]

# 处理后
patched_messages = [
    HumanMessage(content="Search for Python docs"),
    AIMessage(...),  # 原始 AI 消息
    # 插入的占位消息
    ToolMessage(
        content="Tool call search with id call_1 was cancelled...",
        name="search",
        tool_call_id="call_1",
    ),
    ToolMessage(content="Found docs", ...),  # 原始响应
    HumanMessage(content="Thanks"),
]
```

## 5. 关键设计决策

### 5.1 为什么使用 Overwrite？

```python
return {"messages": Overwrite(patched_messages)}
```

- **完整控制**：完全替换消息列表，而非追加
- **顺序保证**：确保占位消息插入到正确的位置
- **无副作用**：不修改原始状态，遵循不可变性原则

### 5.2 为什么使用简单文本内容？

```python
tool_msg_content = (
    f"Tool call {tool_call['name']} with id {tool_call['id']} "
    f"was cancelled - another message came in before it could be completed."
)
```

- **人类可读**：便于调试和理解代理行为
- **无状态依赖**：不需要访问后端或其他资源
- **轻量快速**：同步处理，无 I/O 操作

### 5.3 为什么不尝试重新执行？

- **安全性**：重新执行可能产生副作用
- **不确定性**：原始上下文可能已丢失
- **用户意图**：中断通常是有意的，不应自动继续

## 6. 在代理栈中的位置

```python
# create_deep_agent 中的中间件顺序
deepagent_middleware = [
    TodoListMiddleware(),
    MemoryMiddleware(...),      # 注入记忆
    SkillsMiddleware(...),      # 注入技能
    FilesystemMiddleware(...),  # 文件工具
    SubAgentMiddleware(...),    # 子代理
    SummarizationMiddleware(...),  # 摘要管理
    AnthropicPromptCachingMiddleware(...),
    PatchToolCallsMiddleware(),  # ← 最后执行，确保所有前置修改都被修复
]
```

### 6.1 位置策略

将 `PatchToolCallsMiddleware` 放在栈的最后确保：
1. 所有其他中间件已完成消息修改
2. 包括 `SummarizationMiddleware` 的历史重写
3. 所有悬空调用都在最终执行前被修复

## 7. 边界情况处理

### 7.1 空消息列表

```python
if not messages or len(messages) == 0:
    return None  # 无需处理
```

### 7.2 无工具调用的消息

```python
if msg.type == "ai" and msg.tool_calls:
    # 只有 AI 消息且有工具调用时才处理
```

### 7.3 多个工具调用

```python
for tool_call in msg.tool_calls:
    # 为每个工具调用独立查找对应响应
    # 可能产生多个占位消息
```

### 7.4 链式悬空

```python
# 场景：连续的 AI 消息都有悬空工具调用
messages = [
    AIMessage(tool_calls=[call1]),  # 悬空
    AIMessage(tool_calls=[call2]),  # 也悬空
]

# 处理结果：为每个调用插入占位消息
patched = [
    AIMessage(tool_calls=[call1]),
    ToolMessage(call1_id, "cancelled..."),  # 占位
    AIMessage(tool_calls=[call2]),
    ToolMessage(call2_id, "cancelled..."),  # 占位
]
```

## 8. 使用示例

### 8.1 基本用法

```python
from deepagents import create_deep_agent
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

# 自动包含在 create_deep_agent 中
agent = create_deep_agent()

# 手动添加到自定义中间件栈
agent = create_deep_agent(
    middleware=[
        MyCustomMiddleware(),
        PatchToolCallsMiddleware(),  # 建议放在最后
    ]
)
```

### 8.2 与其他中间件配合

```python
middleware = [
    # 修改消息的中间件
    SummarizationMiddleware(...),
    
    # 确保消息完整性的中间件
    PatchToolCallsMiddleware(),
]
```

## 9. 性能考量

### 9.1 轻量级设计

- 无 I/O 操作
- 无外部依赖
- 纯内存处理
- 线性时间复杂度

### 9.2 触发频率

- 正常运行时：不触发（无悬空调用）
- 中断场景：触发并快速修复
- 最坏情况：每条 AI 消息都触发，但仍保持 O(n) 性能

## 10. 总结

`PatchToolCallsMiddleware` 是一个关键的防御性组件：

1. **问题修复**：自动检测并修复悬空工具调用
2. **轻量高效**：无 I/O、无依赖、线性复杂度
3. **位置敏感**：应放在中间件栈的最后
4. **透明处理**：占位消息对人类和LLM都可读
5. **安全第一**：不尝试重新执行，仅标记为取消

虽然是小组件，但确保了代理对话的完整性和稳定性，是生产环境中不可或缺的防护机制。
