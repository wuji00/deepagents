# FilesystemMiddleware 详解

## 1. 概述

`FilesystemMiddleware` 是 DeepAgents 的核心中间件，提供文件系统操作工具和可选的命令执行能力。它位于 `deepagents/middleware/filesystem.py`，基于 `BackendProtocol` 实现统一的文件操作接口。

## 2. 提供的工具

| 工具名称 | 功能 | 需要 SandboxBackend |
|---------|------|-------------------|
| `ls` | 列出目录内容 | 否 |
| `read_file` | 读取文件内容（支持分页） | 否 |
| `write_file` | 写入新文件 | 否 |
| `edit_file` | 编辑文件内容 | 否 |
| `glob` | 文件模式匹配 | 否 |
| `grep` | 文件内容搜索 | 否 |
| `execute` | 执行 shell 命令 | **是** |

## 3. 类定义与初始化

```python
class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem and optional execution tools.
    
    This middleware adds filesystem tools to the agent: `ls`, `read_file`,
    `write_file`, `edit_file`, `glob`, and `grep`.
    
    If the backend implements `SandboxBackendProtocol`, an `execute` tool
    is also added for running shell commands.
    
    This middleware also automatically evicts large tool results to the file
    system when they exceed a token threshold.
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """Initialize the filesystem middleware.
        
        Args:
            backend: Backend for file storage and optional execution.
                If not provided, defaults to `StateBackend`.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
            tool_token_limit_before_evict: Token limit before evicting a tool
                result to the filesystem. When exceeded, writes the result
                using the configured backend and replaces it with a truncated
                preview and file reference.
        """
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict

        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]
```

## 4. Backend 解析

```python
def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
    """Get the resolved backend instance from backend or factory.
    
    Supports both direct BackendProtocol instances and factory callables.
    """
    if callable(self.backend):
        return self.backend(runtime)
    return self.backend
```

## 5. 工具创建详解

### 5.1 ls 工具

```python
def _create_ls_tool(self) -> BaseTool:
    """Create the ls (list files) tool."""
    tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

    def sync_ls(
        runtime: ToolRuntime[None, FilesystemState],
        path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
    ) -> str:
        resolved_backend = self._get_backend(runtime)
        try:
            validated_path = _validate_path(path)
        except ValueError as e:
            return f"Error: {e}"
        
        infos = resolved_backend.ls_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_ls(
        runtime: ToolRuntime[None, FilesystemState],
        path: Annotated[str, "Absolute path to the directory to list."],
    ) -> str:
        resolved_backend = self._get_backend(runtime)
        try:
            validated_path = _validate_path(path)
        except ValueError as e:
            return f"Error: {e}"
        
        infos = await resolved_backend.als_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="ls",
        description=tool_description,
        func=sync_ls,
        coroutine=async_ls,
    )
```

**关键特性：**
- 同步和异步版本
- 路径验证
- 结果截断（防止 token 溢出）

### 5.2 read_file 工具

```python
def _create_read_file_tool(self) -> BaseTool:
    """Create the read_file tool."""
    tool_description = self._custom_tool_descriptions.get("read_file") or READ_FILE_TOOL_DESCRIPTION
    token_limit = self._tool_token_limit_before_evict

    def sync_read_file(
        file_path: Annotated[str, "Absolute path to the file to read."],
        runtime: ToolRuntime[None, FilesystemState],
        offset: Annotated[int, "Line number to start reading from (0-indexed)."] = 0,
        limit: Annotated[int, "Maximum number of lines to read."] = 100,
    ) -> str:
        resolved_backend = self._get_backend(runtime)
        try:
            validated_path = _validate_path(file_path)
        except ValueError as e:
            return f"Error: {e}"
        
        result = resolved_backend.read(validated_path, offset=offset, limit=limit)

        lines = result.splitlines(keepends=True)
        if len(lines) > limit:
            lines = lines[:limit]
            result = "".join(lines)

        # 如果结果超过 token 阈值，截断
        if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
            truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
            max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
            result = result[:max_content_length]
            result += truncation_msg

        return result

    return StructuredTool.from_function(
        name="read_file",
        description=tool_description,
        func=sync_read_file,
        coroutine=async_read_file,
    )
```

**关键特性：**
- 支持分页（offset/limit）
- 自动截断超长内容
- 提供截断提示

### 5.3 write_file 工具

```python
def _create_write_file_tool(self) -> BaseTool:
    """Create the write_file tool."""
    tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

    def sync_write_file(
        file_path: Annotated[str, "Absolute path where the file should be created."],
        content: Annotated[str, "The text content to write to the file."],
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        resolved_backend = self._get_backend(runtime)
        try:
            validated_path = _validate_path(file_path)
        except ValueError as e:
            return f"Error: {e}"
        
        res: WriteResult = resolved_backend.write(validated_path, content)
        if res.error:
            return res.error
        
        # 如果 backend 返回 state 更新，包装成 Command
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Updated file {res.path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Updated file {res.path}"

    return StructuredTool.from_function(
        name="write_file",
        description=tool_description,
        func=sync_write_file,
        coroutine=async_write_file,
    )
```

**关键特性：**
- 文件已存在时返回错误
- StateBackend 返回 `Command` 更新 state
- 外部存储返回简单字符串

### 5.4 edit_file 工具

```python
def _create_edit_file_tool(self) -> BaseTool:
    """Create the edit_file tool."""
    tool_description = self._custom_tool_descriptions.get("edit_file") or EDIT_FILE_TOOL_DESCRIPTION

    def sync_edit_file(
        file_path: Annotated[str, "Absolute path to the file to edit."],
        old_string: Annotated[str, "The exact text to find and replace."],
        new_string: Annotated[str, "The text to replace old_string with."],
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: Annotated[bool, "If True, replace all occurrences."] = False,
    ) -> Command | str:
        resolved_backend = self._get_backend(runtime)
        try:
            validated_path = _validate_path(file_path)
        except ValueError as e:
            return f"Error: {e}"
        
        res: EditResult = resolved_backend.edit(
            validated_path, old_string, new_string, replace_all=replace_all
        )
        if res.error:
            return res.error
        
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully replaced {res.occurrences} instance(s)",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Successfully replaced {res.occurrences} instance(s)"
```

**关键特性：**
- 精确字符串匹配
- `replace_all` 参数控制替换范围
- 返回替换次数

### 5.5 glob 工具

```python
def _create_glob_tool(self) -> BaseTool:
    """Create the glob tool."""
    tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

    def sync_glob(
        pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py')."],
        runtime: ToolRuntime[None, FilesystemState],
        path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
    ) -> str:
        resolved_backend = self._get_backend(runtime)
        infos = resolved_backend.glob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="glob",
        description=tool_description,
        func=sync_glob,
        coroutine=async_glob,
    )
```

### 5.6 grep 工具

```python
def _create_grep_tool(self) -> BaseTool:
    """Create the grep tool."""
    tool_description = self._custom_tool_descriptions.get("grep") or GREP_TOOL_DESCRIPTION

    def sync_grep(
        pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
        runtime: ToolRuntime[None, FilesystemState],
        path: Annotated[str | None, "Directory to search in."] = None,
        glob: Annotated[str | None, "Glob pattern to filter files."] = None,
        output_mode: Annotated[
            Literal["files_with_matches", "content", "count"],
            "Output format."
        ] = "files_with_matches",
    ) -> str:
        resolved_backend = self._get_backend(runtime)
        raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)

    return StructuredTool.from_function(
        name="grep",
        description=tool_description,
        func=sync_grep,
        coroutine=async_grep,
    )
```

**输出模式：**
- `files_with_matches`：仅返回文件路径
- `content`：返回匹配行及上下文
- `count`：返回每个文件的匹配数量

### 5.7 execute 工具

```python
def _create_execute_tool(self) -> BaseTool:
    """Create the execute tool for sandbox command execution."""
    tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

    def sync_execute(
        command: Annotated[str, "Shell command to execute in the sandbox environment."],
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        resolved_backend = self._get_backend(runtime)

        # 运行时检查是否支持执行
        if not _supports_execution(resolved_backend):
            return (
                "Error: Execution not available. This agent's backend "
                "does not support command execution (SandboxBackendProtocol)."
            )

        try:
            result = resolved_backend.execute(command)
        except NotImplementedError as e:
            return f"Error: Execution not available. {e}"

        # 格式化输出
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)

    return StructuredTool.from_function(
        name="execute",
        description=tool_description,
        func=sync_execute,
        coroutine=async_execute,
    )
```

**关键特性：**
- 运行时检查后端是否支持执行
- 优雅降级（返回错误消息而不是崩溃）
- 格式化输出（退出状态、截断提示）

## 6. 执行能力检测

```python
def _supports_execution(backend: BackendProtocol) -> bool:
    """Check if a backend supports command execution.
    
    For CompositeBackend, checks if the default backend supports execution.
    For other backends, checks if they implement SandboxBackendProtocol.
    """
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)
    return isinstance(backend, SandboxBackendProtocol)
```

## 7. 模型调用包装

### 7.1 wrap_model_call

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Update the system prompt and filter tools based on backend capabilities.
    
    1. Checks if execute tool should be filtered out (backend doesn't support it)
    2. Builds dynamic system prompt based on available tools
    """
    # 检查是否有 execute 工具且后端是否支持
    has_execute_tool = any(
        (tool.name if hasattr(tool, "name") else tool.get("name")) == "execute"
        for tool in request.tools
    )

    backend_supports_execution = False
    if has_execute_tool:
        backend = self._get_backend(request.runtime)
        backend_supports_execution = _supports_execution(backend)

        # 如果后端不支持执行，过滤掉 execute 工具
        if not backend_supports_execution:
            filtered_tools = [
                tool for tool in request.tools
                if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"
            ]
            request = request.override(tools=filtered_tools)
            has_execute_tool = False

    # 构建动态 system prompt
    if self._custom_system_prompt is not None:
        system_prompt = self._custom_system_prompt
    else:
        prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]
        if has_execute_tool and backend_supports_execution:
            prompt_parts.append(EXECUTION_SYSTEM_PROMPT)
        system_prompt = "\n\n".join(prompt_parts)

    if system_prompt:
        new_system_message = append_to_system_message(
            request.system_message, system_prompt
        )
        request = request.override(system_message=new_system_message)

    return handler(request)
```

## 8. 大内容转存机制

### 8.1 内容预览生成

```python
def _create_content_preview(
    content_str: str,
    *,
    head_lines: int = 5,
    tail_lines: int = 5
) -> str:
    """Create a preview of content showing head and tail with truncation marker.
    
    Example output:
        1	First line
        2	Second line
        ...
        5	Fifth line
        
        ... [50 lines truncated] ...
        
        56	Fifty-sixth line
        57	Fifty-seventh line
    """
    lines = content_str.splitlines()

    if len(lines) <= head_lines + tail_lines:
        preview_lines = [line[:1000] for line in lines]
        return format_content_with_line_numbers(preview_lines, start_line=1)

    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]

    head_sample = format_content_with_line_numbers(head, start_line=1)
    truncation_notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return head_sample + truncation_notice + tail_sample
```

### 8.2 大消息处理

```python
def _process_large_message(
    self,
    message: ToolMessage,
    resolved_backend: BackendProtocol,
) -> tuple[ToolMessage, dict[str, FileData] | None]:
    """Process a large ToolMessage by evicting its content to filesystem.
    
    Returns:
        Tuple of (processed_message, files_update):
        - processed_message: New ToolMessage with truncated content and file reference
        - files_update: Dict of file updates to apply to state, or None
    """
    if not self._tool_token_limit_before_evict:
        return message, None

    # 转换内容为字符串
    if isinstance(message.content, str):
        content_str = message.content
    else:
        content_str = str(message.content)

    # 检查是否超过阈值
    if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
        return message, None

    # 写入文件系统
    sanitized_id = sanitize_tool_call_id(message.tool_call_id)
    file_path = f"/large_tool_results/{sanitized_id}"
    result = resolved_backend.write(file_path, content_str)
    if result.error:
        return message, None

    # 创建预览
    content_sample = _create_content_preview(content_str)
    replacement_text = TOO_LARGE_TOOL_MSG.format(
        tool_call_id=message.tool_call_id,
        file_path=file_path,
        content_sample=content_sample,
    )

    processed_message = ToolMessage(
        content=replacement_text,
        tool_call_id=message.tool_call_id,
        name=message.name,
    )
    return processed_message, result.files_update
```

### 8.3 工具调用拦截

```python
# 工具被排除在大结果转存之外（有内置截断或不适用）
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls", "glob", "grep",      # 内置截断
    "read_file",               # 分页读取
    "edit_file", "write_file", # 结果很小
)

def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Check the size of the tool call result and evict to filesystem if too large."""
    if (self._tool_token_limit_before_evict is None or
        request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION):
        return handler(request)

    tool_result = handler(request)
    return self._intercept_large_tool_result(tool_result, request.runtime)
```

## 9. 系统提示词

### 9.1 文件系统提示词

```python
FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files"""
```

### 9.2 执行提示词

```python
EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""
```

## 10. 使用示例

### 10.1 基本使用

```python
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware

agent = create_deep_agent(
    middleware=[FilesystemMiddleware()]
)
```

### 10.2 自定义 Backend

```python
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

middleware = FilesystemMiddleware(
    backend=FilesystemBackend(
        root_dir="/workspace",
        virtual_mode=True,
    )
)

agent = create_deep_agent(middleware=[middleware])
```

### 10.3 启用命令执行

```python
from deepagents.backends import LocalShellBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

middleware = FilesystemMiddleware(
    backend=LocalShellBackend(root_dir="/workspace")
)

agent = create_deep_agent(middleware=[middleware])
```

### 10.4 调整转存阈值

```python
middleware = FilesystemMiddleware(
    tool_token_limit_before_evict=50000,  # 50k tokens
)
```

## 11. 设计要点

### 11.1 工具与 Backend 解耦
- Middleware 提供工具定义
- Backend 提供具体实现
- 同一套工具可工作在不同 Backend 上

### 11.2 大结果处理
- 自动检测超大结果
- 转存到文件系统
- 返回预览和文件路径引用

### 11.3 执行能力动态检测
- 运行时检查 Backend 能力
- 不支持时过滤工具
- 优雅降级而不是崩溃

## 12. 注意事项

1. **路径验证**：所有路径必须通过 `_validate_path()` 验证
2. **结果截断**：列表类结果会自动截断
3. **State 更新**：StateBackend 返回 `Command`，外部存储返回字符串
4. **异步支持**：所有工具都有同步和异步版本
5. **Token 估算**：使用 `NUM_CHARS_PER_TOKEN = 4` 估算 token 数

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
