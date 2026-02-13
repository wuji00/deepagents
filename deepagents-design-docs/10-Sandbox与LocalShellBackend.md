# Sandbox 与 LocalShellBackend 详解

## 1. 概述

DeepAgents 通过 `SandboxBackendProtocol` 支持命令执行功能。本文档涵盖：
- `BaseSandbox`：沙箱基类，通过 shell 命令实现所有文件操作
- `LocalShellBackend`：本地 shell 后端，在主机上直接执行命令
- `SandboxProvider`：第三方沙箱提供者的抽象接口

## 2. SandboxBackendProtocol

```python
class SandboxBackendProtocol(BackendProtocol):
    """Extension of BackendProtocol that adds shell command execution.
    
    Designed for backends running in isolated environments (containers, VMs,
    remote hosts).
    
    Adds `execute()`/`aexecute()` for shell commands and an `id` property.
    """

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend instance."""
        raise NotImplementedError

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the process.
        
        Args:
            command: Full shell command string to execute.
        
        Returns:
            ExecuteResponse with combined output, exit code, optional signal,
            and truncation flag.
        """
        raise NotImplementedError
```

## 3. BaseSandbox 基类

`BaseSandbox` 位于 `deepagents/backends/sandbox.py`，提供通过 shell 命令实现所有文件操作的基类。

### 3.1 设计理念

子类只需要实现 `execute()` 方法，即可获得完整的文件操作能力：

```python
class MySandbox(BaseSandbox):
    def execute(self, command: str) -> ExecuteResponse:
        # 在沙箱环境中执行命令
        return self._run_in_container(command)
```

### 3.2 命令模板

BaseSandbox 使用 Python 脚本模板在沙箱中执行文件操作：

#### Glob 模板
```python
_GLOB_COMMAND_TEMPLATE = """python3 -c "
import glob
import os
import json
import base64

path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

os.chdir(path)
matches = sorted(glob.glob(pattern, recursive=True))
for m in matches:
    stat = os.stat(m)
    result = {{
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }}
    print(json.dumps(result))
" 2>/dev/null"""
```

#### 写入模板（使用 heredoc 避免 ARG_MAX 限制）
```python
_WRITE_COMMAND_TEMPLATE = """python3 -c "
import os
import sys
import base64
import json

# 从 stdin 读取 base64 编码的 payload
payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received', file=sys.stderr)
    sys.exit(1)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    content = base64.b64decode(data['content']).decode('utf-8')
except Exception as e:
    print(f'Error: Failed to decode: {e}', file=sys.stderr)
    sys.exit(1)

# 检查文件是否存在
if os.path.exists(file_path):
    print(f'Error: File already exists', file=sys.stderr)
    sys.exit(1)

# 创建父目录
parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)

with open(file_path, 'w') as f:
    f.write(content)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""
```

**使用 heredoc 的原因：**
- `ARG_MAX` 限制命令行参数总大小（通常 ~2MB）
- Heredoc 通过 stdin 传递数据，绕过此限制
- 支持大文件写入（base64 编码后可能很大）

#### 编辑模板
```python
_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys
import base64
import json
import os

payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received', file=sys.stderr)
    sys.exit(4)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    old = data['old']
    new = data['new']
except Exception as e:
    print(f'Error: Failed to decode: {e}', file=sys.stderr)
    sys.exit(4)

# 检查文件是否存在
if not os.path.isfile(file_path):
    sys.exit(3)  # 文件不存在

with open(file_path, 'r') as f:
    text = f.read()

# 计数匹配
count = text.count(old)

if count == 0:
    sys.exit(1)  # 字符串未找到
elif count > 1 and not {replace_all}:
    sys.exit(2)  # 多处匹配但未指定 replace_all

# 执行替换
if {replace_all}:
    result = text.replace(old, new)
else:
    result = text.replace(old, new, 1)

with open(file_path, 'w') as f:
    f.write(result)

print(count)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""
```

**退出码约定：**
- 0：成功
- 1：字符串未找到
- 2：多处匹配（需要 `replace_all=True`）
- 3：文件不存在
- 4：payload 解码失败

#### 读取模板
```python
_READ_COMMAND_TEMPLATE = """python3 -c "
import os
import sys

file_path = '{file_path}'
offset = {offset}
limit = {limit}

if not os.path.isfile(file_path):
    print('Error: File not found')
    sys.exit(1)

if os.path.getsize(file_path) == 0:
    print('System reminder: File exists but has empty contents')
    sys.exit(0)

with open(file_path, 'r') as f:
    lines = f.readlines()

start_idx = offset
end_idx = offset + limit
selected_lines = lines[start_idx:end_idx]

# 格式化输出行号
for i, line in enumerate(selected_lines):
    line_num = offset + i + 1
    line_content = line.rstrip('\\n')
    print(f'{{line_num:6d}}\\t{{line_content}}')
" 2>&1"""
```

### 3.3 BaseSandbox 方法实现

#### ls_info - 列出目录
```python
def ls_info(self, path: str) -> list[FileInfo]:
    """Structured listing with file metadata using os.scandir."""
    cmd = f"""python3 -c "
import os
import json

path = '{path}'

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': os.path.join(path, entry.name),
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    pass
except PermissionError:
    pass
" 2>/dev/null"""

    result = self.execute(cmd)

    file_infos: list[FileInfo] = []
    for line in result.output.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
        except json.JSONDecodeError:
            continue

    return file_infos
```

#### read - 读取文件
```python
def read(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content with line numbers using a single shell command."""
    cmd = _READ_COMMAND_TEMPLATE.format(
        file_path=file_path,
        offset=offset,
        limit=limit
    )
    result = self.execute(cmd)

    output = result.output.rstrip()
    exit_code = result.exit_code

    if exit_code != 0 or "Error: File not found" in output:
        return f"Error: File '{file_path}' not found"

    return output
```

#### write - 写入文件
```python
def write(
    self,
    file_path: str,
    content: str,
) -> WriteResult:
    """Create a new file."""
    # Base64 编码参数
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    payload = json.dumps({"path": file_path, "content": content_b64})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

    cmd = _WRITE_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
    result = self.execute(cmd)

    if result.exit_code != 0 or "Error:" in result.output:
        error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
        return WriteResult(error=error_msg)

    return WriteResult(path=file_path, files_update=None)
```

#### edit - 编辑文件
```python
def edit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditResult:
    """Edit a file by replacing string occurrences."""
    payload = json.dumps({"path": file_path, "old": old_string, "new": new_string})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

    cmd = _EDIT_COMMAND_TEMPLATE.format(
        payload_b64=payload_b64,
        replace_all=replace_all
    )
    result = self.execute(cmd)

    exit_code = result.exit_code
    output = result.output.strip()

    # 映射退出码到错误消息
    error_messages = {
        1: f"Error: String not found in file: '{old_string}'",
        2: f"Error: String '{old_string}' appears multiple times...",
        3: f"Error: File '{file_path}' not found",
        4: f"Error: Failed to decode edit payload: {output}",
    }
    if exit_code in error_messages:
        return EditResult(error=error_messages[exit_code])
    if exit_code != 0:
        return EditResult(error=f"Error editing file (exit code {exit_code})")

    count = int(output)
    return EditResult(path=file_path, files_update=None, occurrences=count)
```

#### grep_raw - 搜索文件
```python
def grep_raw(
    self,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """Structured search results or error string for invalid input."""
    search_path = shlex.quote(path or ".")

    # 构建 grep 命令
    grep_opts = "-rHnF"  # recursive, with filename, with line number, fixed-strings

    glob_pattern = ""
    if glob:
        glob_pattern = f"--include='{glob}'"

    pattern_escaped = shlex.quote(pattern)

    cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null || true"
    result = self.execute(cmd)

    output = result.output.rstrip()
    if not output:
        return []

    # 解析 grep 输出
    matches: list[GrepMatch] = []
    for line in output.split("\n"):
        parts = line.split(":", 2)
        if len(parts) >= 3:
            matches.append({
                "path": parts[0],
                "line": int(parts[1]),
                "text": parts[2],
            })

    return matches
```

## 4. LocalShellBackend

`LocalShellBackend` 位于 `deepagents/backends/local_shell.py`，继承自 `FilesystemBackend` 和 `SandboxBackendProtocol`，在本地主机上提供无限制的 shell 执行能力。

### 4.1 安全警告

```python
class LocalShellBackend(FilesystemBackend, SandboxBackendProtocol):
    """Filesystem backend with unrestricted local shell command execution.
    
    !!! warning "Security Warning"
        This backend grants agents BOTH direct filesystem access AND unrestricted
        shell execution on your local machine. Use with extreme caution and only
        in appropriate environments.
    """
```

**安全风险：**
- Agent 可以执行 **任意 shell 命令**
- Agent 可以读取 **任何可访问的文件**（包括密钥、凭证）
- 结合网络工具，可能通过 SSRF 攻击外泄机密
- **无进程隔离** - 命令直接在主机上运行
- **无资源限制** - 命令可能消耗无限 CPU、内存、磁盘

### 4.2 初始化

```python
def __init__(
    self,
    root_dir: str | Path | None = None,
    *,
    virtual_mode: bool = False,
    timeout: float = 120.0,
    max_output_bytes: int = 100_000,
    env: dict[str, str] | None = None,
    inherit_env: bool = False,
) -> None:
    """Initialize local shell backend.
    
    Args:
        root_dir: Working directory for both filesystem operations and shell commands.
        virtual_mode: Enable virtual path mode for filesystem operations.
            **Note:** This does NOT restrict shell commands.
        timeout: Maximum time in seconds to wait for shell command execution.
        max_output_bytes: Maximum number of bytes to capture from command output.
        env: Environment variables for shell commands.
        inherit_env: Whether to inherit the parent process's environment variables.
    """
    super().__init__(
        root_dir=root_dir,
        virtual_mode=virtual_mode,
        max_file_size_mb=10,
    )

    self._timeout = timeout
    self._max_output_bytes = max_output_bytes

    # 构建环境变量
    if inherit_env:
        self._env = os.environ.copy()
        if env is not None:
            self._env.update(env)
    else:
        self._env = env if env is not None else {}

    # 生成唯一 ID
    self._sandbox_id = f"local-{uuid.uuid4().hex[:8]}"
```

### 4.3 命令执行

```python
def execute(self, command: str) -> ExecuteResponse:
    """Execute a shell command directly on the host system.
    
    !!! danger "Unrestricted Execution"
        Commands are executed directly on your host system using `subprocess.run()`
        with `shell=True`. There is **no sandboxing, isolation, or security
        restrictions**.
    
    The command is executed using the system shell with the working directory
    set to the backend's `root_dir`. Stdout and stderr are combined into a
    single output stream.
    """
    if not command or not isinstance(command, str):
        return ExecuteResponse(
            output="Error: Command must be a non-empty string.",
            exit_code=1,
            truncated=False,
        )

    try:
        result = subprocess.run(
            command,
            check=False,
            shell=True,  # 允许 shell 特性（管道、重定向等）
            capture_output=True,
            text=True,
            timeout=self._timeout,
            env=self._env,
            cwd=str(self.cwd),
        )

        # 合并 stdout 和 stderr
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

        output = "\n".join(output_parts) if output_parts else "<no output>"

        # 检查截断
        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[:self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        # 添加退出码信息
        if result.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

        return ExecuteResponse(
            output=output,
            exit_code=result.returncode,
            truncated=truncated,
        )

    except subprocess.TimeoutExpired:
        return ExecuteResponse(
            output=f"Error: Command timed out after {self._timeout:.1f} seconds.",
            exit_code=124,
            truncated=False,
        )
    except Exception as e:
        return ExecuteResponse(
            output=f"Error executing command: {e}",
            exit_code=1,
            truncated=False,
        )
```

**输出格式：**
- stdout 正常输出
- stderr 每行前缀 `[stderr]`
- 非零退出码追加到输出末尾
- 超时的退出码为 124

### 4.4 ID 属性

```python
@property
def id(self) -> str:
    """Unique identifier for this backend instance."""
    return self._sandbox_id  # 例如: "local-a1b2c3d4"
```

## 5. SandboxProvider 接口

`SandboxProvider` 是第三方沙箱 SDK 实现的抽象基类，定义沙箱生命周期管理接口。

### 5.1 核心类型

```python
class SandboxInfo(TypedDict, Generic[MetadataT]):
    """Metadata for a single sandbox instance."""
    sandbox_id: str
    metadata: NotRequired[MetadataT]  # 提供者特定的元数据

class SandboxListResponse(TypedDict, Generic[MetadataT]):
    """Paginated response from a sandbox list operation."""
    items: list[SandboxInfo[MetadataT]]
    cursor: str | None  # 分页游标
```

### 5.2 接口定义

```python
class SandboxProvider(ABC, Generic[MetadataT]):
    """Abstract base class for third-party sandbox provider implementations."""

    @abstractmethod
    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MetadataT]:
        """List available sandboxes with optional filtering and pagination."""

    @abstractmethod
    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing sandbox or create a new one.
        
        Important: If sandbox_id is provided but doesn't exist, raises an error.
        Only when sandbox_id is None should a new sandbox be created.
        """

    @abstractmethod
    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a sandbox instance.
        
        Idempotency: Calling delete on a non-existent sandbox should succeed.
        """

    # 异步版本（默认使用线程池）
    async def alist(self, *, cursor: str | None = None, **kwargs: Any) -> SandboxListResponse[MetadataT]:
        return await asyncio.to_thread(self.list, cursor=cursor, **kwargs)

    async def aget_or_create(self, *, sandbox_id: str | None = None, **kwargs: Any) -> SandboxBackendProtocol:
        return await asyncio.to_thread(self.get_or_create, sandbox_id=sandbox_id, **kwargs)

    async def adelete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)
```

### 5.3 实现示例

```python
class CustomMetadata(TypedDict, total=False):
    status: Literal["running", "stopped"]
    template: str
    created_at: str

class CustomSandboxProvider(SandboxProvider[CustomMetadata]):
    def list(
        self,
        *,
        cursor: str | None = None,
        status: Literal["running", "stopped"] | None = None,
        **kwargs: Any
    ) -> SandboxListResponse[CustomMetadata]:
        # 查询提供者 API
        return {"items": [...], "cursor": None}

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        template_id: str = "default",
        **kwargs: Any
    ) -> SandboxBackendProtocol:
        if sandbox_id:
            # 连接到现有沙箱
            return CustomSandbox(sandbox_id)
        # 创建新沙箱
        new_id = self._create_new(template_id)
        return CustomSandbox(new_id)

    def delete(self, *, sandbox_id: str, force: bool = False, **kwargs: Any) -> None:
        self._client.delete(sandbox_id, force=force)
```

## 6. 使用示例

### 6.1 LocalShellBackend 基本使用

```python
from deepagents.backends import LocalShellBackend

# 创建后端
backend = LocalShellBackend(
    root_dir="/home/user/project",
    virtual_mode=True,  # 路径隔离
    timeout=60.0,
    inherit_env=True,   # 继承父进程环境变量
)

# 执行命令
result = backend.execute("python --version")
print(result.output)  # Python 3.11.0
print(result.exit_code)  # 0

# 文件操作（继承自 FilesystemBackend）
backend.write("/test.txt", "Hello")
content = backend.read("/test.txt")
```

### 6.2 在 CompositeBackend 中使用

```python
from deepagents.backends import CompositeBackend, LocalShellBackend, StoreBackend

backend = CompositeBackend(
    default=LocalShellBackend(
        root_dir="/workspace",
        virtual_mode=True,
    ),
    routes={
        "/memories/": StoreBackend,
    }
)

# 执行命令（通过 LocalShellBackend）
result = backend.execute("ls -la /workspace")

# 存储持久化数据（通过 StoreBackend）
backend.write("/memories/note.txt", "persistent data")
```

## 7. 安全最佳实践

### 7.1 开发环境

```python
# 启用 virtual_mode，但仍然有风险
backend = LocalShellBackend(
    root_dir="/home/user/project",
    virtual_mode=True,
)

# **强烈推荐**：使用 Human-in-the-Loop
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_deep_agent(
    backend=backend,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"execute": True})
    ]
)
```

### 7.2 CI/CD 环境

- 使用专门的构建镜像
- 排除敏感文件（`.env`, `secrets/`）
- 限制网络访问
- 考虑使用 `StateBackend` 代替

### 7.3 生产环境

**不推荐**使用 `LocalShellBackend`，考虑：
- Docker 容器（通过 `BaseSandbox` 实现）
- 云沙箱服务（Daytona, Modal, Runloop）
- VM 隔离

## 8. 优缺点分析

### BaseSandbox
| 优点 | 缺点 |
|------|------|
| 只需实现 `execute()` | 性能开销（每次操作都是 shell 调用）|
| 自动获得所有文件操作 | 依赖目标环境的 Python 3 |
| 通过命令模板保证行为一致 | 大文件传输较慢 |

### LocalShellBackend
| 优点 | 缺点 |
|------|------|
| 简单易用 | **无安全隔离** |
| 完整 shell 功能 | 可以访问整个文件系统 |
| 继承 FilesystemBackend 的所有功能 | 不适合生产环境 |
| | 需要额外安全措施（HITL） |

## 9. 注意事项

1. **virtual_mode 限制**：只影响文件操作路径，不影响 shell 命令
2. **环境变量**：`inherit_env=False` 时，命令可能缺少必要的环境变量
3. **超时设置**：长时间运行的命令需要增加 timeout
4. **输出截断**：大输出会被截断，检查 `truncated` 标志
5. **编码问题**：假设 UTF-8，其他编码可能导致错误

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
