# 21 - Middleware 工具函数详解

## 1. 概述

`middleware/_utils.py` 提供了中间件共享的实用函数，目前主要包含系统消息追加功能。虽然代码量不大，但它是多个中间件（MemoryMiddleware、SkillsMiddleware 等）的基础组件。

## 2. 核心函数

### 2.1 append_to_system_message - 系统消息追加

```python
from langchain_core.messages import SystemMessage


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """向系统消息追加文本内容。
    
    这是多个中间件（Memory、Skills、Filesystem）共用的核心函数，
    用于将额外的指令、记忆、技能描述等注入到系统提示中。
    
    Args:
        system_message: 现有的系统消息或 None
        text: 要追加的文本内容
    
    Returns:
        包含追加内容的新 SystemMessage
    """
    # 1. 提取现有内容块
    # 如果 system_message 为 None，从空列表开始
    # 使用 content_blocks 属性获取结构化内容
    new_content: list[str | dict[str, str]] = (
        list(system_message.content_blocks) if system_message else []
    )
    
    # 2. 添加分隔符（如果已有内容）
    # 在已有内容和新内容之间添加空行，提高可读性
    if new_content:
        text = f"\n\n{text}"
    
    # 3. 追加新的文本块
    # 使用字典格式表示文本内容块，符合 LangChain 消息格式规范
    new_content.append({"type": "text", "text": text})
    
    # 4. 创建新的 SystemMessage
    # 返回新对象，不修改原始消息（不可变性原则）
    return SystemMessage(content=new_content)
```

## 3. 使用模式分析

### 3.1 在 MemoryMiddleware 中使用

```python
class MemoryMiddleware(AgentMiddleware):
    def modify_request(self, request: ModelRequest) -> ModelRequest:
        # 格式化记忆内容（包含使用指南）
        agent_memory = self._format_agent_memory(contents)
        
        # 使用 append_to_system_message 注入记忆
        new_system_message = append_to_system_message(
            request.system_message, 
            agent_memory
        )
        
        return request.override(system_message=new_system_message)
```

**注入内容示例：**
```xml
<agent_memory>
~/.deepagents/AGENTS.md
# User Preferences
- Prefers TypeScript over Python
- Likes concise responses

./.deepagents/AGENTS.md  
# Project Guidelines
- Use asyncio for async code
- Follow PEP 8 style
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files...
    [详细的使用指南]
</memory_guidelines>
```

### 3.2 在 SkillsMiddleware 中使用

```python
class SkillsMiddleware(AgentMiddleware):
    def modify_request(self, request: ModelRequest) -> ModelRequest:
        # 渲染技能段落
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )
        
        # 追加到系统消息
        new_system_message = append_to_system_message(
            request.system_message, 
            skills_section
        )
        
        return request.override(system_message=new_system_message)
```

**注入内容示例：**
```markdown

## Skills System

You have access to a skills library...

**Available Skills:**

- **web-research**: Structured approach to web research (License: MIT)
  -> Read `/skills/user/web-research/SKILL.md` for full instructions

**How to Use Skills (Progressive Disclosure):**
...
```

### 3.3 在 create_deep_agent 中的使用

```python
def create_deep_agent(...):
    # 组合 system_prompt 与 BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        # SystemMessage: 使用 content_blocks 追加
        new_content = [
            *system_prompt.content_blocks,
            {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
        ]
        final_system_message = SystemMessage(content=new_content)
    else:
        # 字符串：简单拼接
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT
```

## 4. 设计决策分析

### 4.1 为什么选择 content_blocks？

```python
# 使用 content_blocks 而非 content
new_content = list(system_message.content_blocks)
```

**原因：**
1. **结构化支持**：`content_blocks` 支持混合内容（文本 + 图像 + 其他）
2. **LangChain 规范**：符合 `SystemMessage` 的最新设计
3. **可扩展性**：未来可支持多模态内容注入

### 4.2 不可变性原则

```python
# 创建新对象，不修改原始消息
return SystemMessage(content=new_content)
```

**好处：**
1. **避免副作用**：原始请求保持不变
2. **可预测性**：调用者知道消息未被修改
3. **调试友好**：可以比较修改前后的消息

### 4.3 分隔符处理

```python
if new_content:
    text = f"\n\n{text}"
```

**设计考虑：**
- 在已有内容和新内容之间添加空行
- 提高系统提示的可读性
- 视觉分离不同来源的内容

## 5. 内容格式详解

### 5.1 LangChain 消息内容格式

```python
# 简单字符串格式（传统）
SystemMessage(content="You are a helpful assistant.")

# 结构化内容块格式（现代）
SystemMessage(content=[
    {"type": "text", "text": "You are a helpful assistant."},
])

# 多模态内容（未来扩展）
SystemMessage(content=[
    {"type": "text", "text": "Analyze this image:"},
    {"type": "image", "source": {"type": "base64", "data": "..."}},
])
```

### 5.2 append_to_system_message 生成的格式

```python
# 输入
system_message = SystemMessage(content="Base prompt")
text = "Additional instructions"

# 输出
SystemMessage(content=[
    {"type": "text", "text": "Base prompt"},
    {"type": "text", "text": "\n\nAdditional instructions"},
])
```

## 6. 边界情况处理

### 6.1 None 输入

```python
system_message = None
text = "New content"

# 结果：直接创建包含 text 的 SystemMessage
SystemMessage(content=[
    {"type": "text", "text": "New content"}
])
```

### 6.2 空内容列表

```python
system_message = SystemMessage(content=[])
text = "New content"

# 结果：不添加分隔符（因为是空列表）
SystemMessage(content=[
    {"type": "text", "text": "New content"}
])
```

### 6.3 多段内容追加

```python
# 第一次追加
msg1 = append_to_system_message(None, "Memory content")

# 第二次追加
msg2 = append_to_system_message(msg1, "Skills content")

# 第三次追加
msg3 = append_to_system_message(msg2, "Filesystem instructions")

# 最终结果
SystemMessage(content=[
    {"type": "text", "text": "Memory content"},
    {"type": "text", "text": "\n\nSkills content"},
    {"type": "text", "text": "\n\nFilesystem instructions"},
])
```

## 7. 扩展性考虑

### 7.1 未来可能的扩展

```python
def append_to_system_message(
    system_message: SystemMessage | None,
    content: str | dict | list[dict],  # 支持多种内容类型
    separator: str = "\n\n",  # 可配置分隔符
    position: Literal["append", "prepend"] = "append",  # 插入位置
) -> SystemMessage:
    """增强版系统消息追加函数。"""
    ...
```

### 7.2 与中间件模式的关系

```
┌─────────────────────────────────────┐
│        AgentMiddleware              │
│  ┌───────────────────────────────┐  │
│  │   modify_request()            │  │
│  │   ┌───────────────────────┐   │  │
│  │   │ append_to_system_     │   │  │
│  │   │ _message()            │   │  │
│  │   └───────────────────────┘   │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## 8. 测试策略

### 8.1 单元测试示例

```python
def test_append_to_system_message():
    # 测试 None 输入
    result = append_to_system_message(None, "text")
    assert result.content == [{"type": "text", "text": "text"}]
    
    # 测试追加
    base = SystemMessage(content="base")
    result = append_to_system_message(base, "extra")
    assert len(result.content) == 2
    assert result.content[1]["text"].startswith("\n\n")
    
    # 测试不可变性
    original_content = base.content.copy()
    append_to_system_message(base, "extra")
    assert base.content == original_content  # 原对象未被修改
```

## 9. 与其他组件的关系

### 9.1 调用链

```
create_deep_agent()
    ├── MemoryMiddleware.modify_request()
    │   └── append_to_system_message()  # 注入记忆
    ├── SkillsMiddleware.modify_request()
    │   └── append_to_system_message()  # 注入技能
    └── FilesystemMiddleware.modify_request()
        └── append_to_system_message()  # 注入文件工具说明
```

### 9.2 最终系统消息结构

```
┌─────────────────────────────────────────┐
│  SystemMessage                          │
│  ├─ content[0]: Base system prompt      │
│  ├─ content[1]: \n\n + Memory section   │
│  ├─ content[2]: \n\n + Skills section   │
│  └─ content[3]: \n\n + Filesystem docs │
└─────────────────────────────────────────┘
```

## 10. 总结

虽然 `append_to_system_message` 是一个简单的工具函数，但它在 DeepAgents 架构中扮演重要角色：

1. **一致性**：所有中间件使用相同的方式修改系统消息
2. **可组合性**：多个中间件的修改可以安全地叠加
3. **可扩展性**：支持未来的多模态内容注入
4. **不可变性**：遵循函数式编程原则，避免副作用

这个函数体现了"小而美"的设计哲学——用最小的代码实现最大的价值。
