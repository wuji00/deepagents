# 13 - SummarizationMiddleware 详解

## 1. 概述

`SummarizationMiddleware` 是一个关键的对话管理中间件，负责在对话历史过长时触发摘要生成，并将完整对话历史卸载到后端存储。这解决了LLM上下文窗口限制的问题，同时保留了对话历史的可恢复性。

## 2. 核心功能

### 2.1 双重处理机制

```python
class SummarizationMiddleware(BaseSummarizationMiddleware):
    """Summarization middleware with backend for conversation history offloading."""
    
    def before_model(self, state, runtime):
        """处理消息：参数截断 -> 检查摘要条件 -> 卸载历史 -> 生成摘要"""
        # Step 1: 截断工具参数
        truncated_messages, args_were_truncated = self._truncate_args(messages)
        
        # Step 2: 检查是否需要摘要
        total_tokens = self.token_counter(truncated_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)
        
        # Step 3: 执行摘要流程
        ...
```

## 3. 触发策略详解

### 3.1 ContextSize 类型

```python
ContextSize = Union[
    Tuple[Literal["messages"], int],      # ("messages", 50) - 保留50条消息
    Tuple[Literal["tokens"], int],        # ("tokens", 100000) - 保留10万token
    Tuple[Literal["fraction"], float],    # ("fraction", 0.85) - 保留85%上下文
]
```

### 3.2 默认策略计算

```python
def _compute_summarization_defaults(model: BaseChatModel) -> SummarizationDefaults:
    """根据模型配置计算默认摘要策略。"""
    has_profile = (
        model.profile is not None
        and "max_input_tokens" in model.profile
    )
    
    if has_profile:
        # 有模型配置：使用比例策略
        return {
            "trigger": ("fraction", 0.85),      # 达到85%触发摘要
            "keep": ("fraction", 0.10),         # 保留最近10%
            "truncate_args_settings": {
                "trigger": ("fraction", 0.85),
                "keep": ("fraction", 0.10),
            },
        }
    else:
        # 无模型配置：使用固定值
        return {
            "trigger": ("tokens", 170000),      # 17万token触发
            "keep": ("messages", 6),            # 保留6条消息
            "truncate_args_settings": {
                "trigger": ("messages", 20),
                "keep": ("messages", 20),
            },
        }
```

## 4. 参数截断机制

### 4.1 TruncateArgsSettings 配置

```python
class TruncateArgsSettings(TypedDict, total=False):
    """工具参数截断设置。"""
    
    trigger: ContextSize | None    # 触发阈值，None表示禁用
    keep: ContextSize              # 保留策略
    max_length: int                # 最大字符长度（默认2000）
    truncation_text: str           # 截断替换文本（默认"...(argument truncated)"）
```

### 4.2 参数截断实现

```python
def _truncate_args(self, messages: list[AnyMessage]) -> tuple[list[AnyMessage], bool]:
    """截断旧消息中的大型工具参数。"""
    
    # 检查是否应该截断
    if not self._should_truncate_args(messages, total_tokens):
        return messages, False
    
    # 确定截断截止索引
    cutoff_index = self._determine_truncate_cutoff_index(messages)
    
    # 只处理截止索引之前的AI消息
    for i, msg in enumerate(messages):
        if i < cutoff_index and isinstance(msg, AIMessage) and msg.tool_calls:
            # 截断 write_file 和 edit_file 的参数
            for tool_call in msg.tool_calls:
                if tool_call["name"] in {"write_file", "edit_file"}:
                    truncated_call = self._truncate_tool_call(tool_call)
```

### 4.3 工具调用参数截断

```python
def _truncate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
    """截断单个工具调用的参数。"""
    args = tool_call.get("args", {})
    truncated_args = {}
    modified = False
    
    for key, value in args.items():
        if isinstance(value, str) and len(value) > self._max_arg_length:
            # 保留前20个字符 + 截断标记
            truncated_args[key] = value[:20] + self._truncation_text
            modified = True
        else:
            truncated_args[key] = value
    
    if modified:
        return {**tool_call, "args": truncated_args}
    return tool_call
```

## 5. 历史卸载机制

### 5.1 存储路径生成

```python
def _get_history_path(self) -> str:
    """生成历史存储路径：/conversation_history/{thread_id}.md"""
    thread_id = self._get_thread_id()
    return f"{self._history_path_prefix}/{thread_id}.md"

def _get_thread_id(self) -> str:
    """从LangGraph配置获取thread_id。"""
    try:
        config = get_config()
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is not None:
            return str(thread_id)
    except RuntimeError:
        pass
    
    # 回退：生成会话ID
    return f"session_{uuid.uuid4().hex[:8]}"
```

### 5.2 消息过滤

```python
def _is_summary_message(self, msg: AnyMessage) -> bool:
    """检查消息是否为之前的摘要消息（避免重复存储）。"""
    if not isinstance(msg, HumanMessage):
        return False
    return msg.additional_kwargs.get("lc_source") == "summarization"

def _filter_summary_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
    """过滤掉之前的摘要消息。"""
    return [msg for msg in messages if not self._is_summary_message(msg)]
```

### 5.3 卸载到后端

```python
def _offload_to_backend(
    self,
    backend: BackendProtocol,
    messages: list[AnyMessage],
) -> str | None:
    """将消息持久化到后端存储。"""
    path = self._get_history_path()
    
    # 过滤之前的摘要消息
    filtered_messages = self._filter_summary_messages(messages)
    
    # 生成带时间戳的章节
    timestamp = datetime.now(UTC).isoformat()
    new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages)}\n\n"
    
    # 读取现有内容并追加
    existing_content = ""
    try:
        responses = backend.download_files([path])
        if responses and responses[0].content is not None:
            existing_content = responses[0].content.decode("utf-8")
    except Exception:
        pass  # 文件不存在，视为新文件
    
    combined_content = existing_content + new_section
    
    # 写入后端
    if existing_content:
        result = backend.edit(path, existing_content, combined_content)
    else:
        result = backend.write(path, combined_content)
    
    return path if result and not result.error else None
```

## 6. 摘要消息构建

### 6.1 带路径引用的摘要消息

```python
def _build_new_messages_with_path(
    self, 
    summary: str, 
    file_path: str | None
) -> list[AnyMessage]:
    """构建包含历史路径引用的摘要消息。"""
    
    if file_path is not None:
        content = f"""\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""
    else:
        content = f"Here is a summary of the conversation to date:\n\n{summary}"
    
    return [
        HumanMessage(
            content=content,
            additional_kwargs={"lc_source": "summarization"},
        )
    ]
```

## 7. 完整处理流程

```python
def before_model(self, state, runtime):
    """完整的预处理流程。"""
    messages = state["messages"]
    
    # 1. 参数截断
    truncated_messages, args_were_truncated = self._truncate_args(messages)
    
    # 2. 检查摘要条件
    total_tokens = self.token_counter(truncated_messages)
    should_summarize = self._should_summarize(truncated_messages, total_tokens)
    
    # 3. 仅截断，不摘要
    if args_were_truncated and not should_summarize:
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *truncated_messages,
            ]
        }
    
    # 4. 无需处理
    if not should_summarize:
        return None
    
    # 5. 执行摘要
    cutoff_index = self._determine_cutoff_index(truncated_messages)
    messages_to_summarize, preserved = self._partition_messages(truncated_messages, cutoff_index)
    
    # 6. 先卸载到后端（防止数据丢失）
    backend = self._get_backend(state, runtime)
    file_path = self._offload_to_backend(backend, messages_to_summarize)
    if file_path is None:
        warnings.warn("Offloading conversation history failed")
    
    # 7. 生成摘要
    summary = self._create_summary(messages_to_summarize)
    new_messages = self._build_new_messages_with_path(summary, file_path)
    
    # 8. 返回更新后的消息列表
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages,
            *preserved,
        ]
    }
```

## 8. 关键设计决策

### 8.1 为什么先卸载后摘要？

```python
# 先卸载，防止数据丢失
file_path = self._offload_to_backend(backend, messages_to_summarize)
if file_path is None:
    warnings.warn("Offloading failed...")  # 警告但继续

# 后生成摘要
summary = self._create_summary(messages_to_summarize)
```

### 8.2 异步支持

```python
async def abefore_model(self, state, runtime):
    """异步版本使用 await 调用异步后端方法。"""
    file_path = await self._aoffload_to_backend(backend, messages_to_summarize)
    summary = await self._acreate_summary(messages_to_summarize)
    ...
```

## 9. 使用示例

```python
from deepagents import create_deep_agent
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="/data")

middleware = SummarizationMiddleware(
    model="gpt-4o-mini",
    backend=backend,
    trigger=("fraction", 0.85),  # 85%上下文时触发
    keep=("fraction", 0.10),     # 保留最近10%
    truncate_args_settings={
        "trigger": ("messages", 50),
        "keep": ("messages", 20),
        "max_length": 2000,
        "truncation_text": "...(truncated)",
    }
)

agent = create_deep_agent(middleware=[middleware])
```

## 10. 总结

SummarizationMiddleware通过以下机制解决了长对话的上下文管理问题：

1. **双重触发策略**：参数截断 + 消息摘要
2. **智能历史卸载**：将完整对话保存到后端，提供可恢复性
3. **链式摘要防护**：过滤之前的摘要消息，避免重复存储
4. **模型自适应**：根据模型profile自动选择合适的触发策略
5. **渐进式处理**：先截断参数减轻负担，必要时再执行完整摘要
