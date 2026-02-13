# BackendProtocol 详解

## 1. 概述

`BackendProtocol` 是 DeepAgents 存储抽象层的核心协议，定义了所有存储后端必须实现的标准接口。它位于 `deepagents/backends/protocol.py`，为文件操作提供统一的抽象。

## 2. 核心协议类

### 2.1 BackendProtocol (ABC)

```python
class BackendProtocol(abc.ABC):
    """Protocol for pluggable memory backends."""
    
    # 文件列表
    def ls_info(self, path: str) -> list["FileInfo"]
    async def als_info(self, path: str) -> list["FileInfo"]
    
    # 文件读取
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str
    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str
    
    # 文件搜索
    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list["GrepMatch"] | str
    async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list["GrepMatch"] | str
    
    # 文件匹配
    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]
    async def aglob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]
    
    # 文件写入
    def write(self, file_path: str, content: str) -> WriteResult
    async def awrite(self, file_path: str, content: str) -> WriteResult
    
    # 文件编辑
    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult
    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult
    
    # 批量文件操作
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]
    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]
    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]
```

### 2.2 SandboxBackendProtocol

扩展 `BackendProtocol`，添加命令执行能力：

```python
class SandboxBackendProtocol(BackendProtocol):
    """Extension of BackendProtocol that adds shell command execution."""
    
    @property
    def id(self) -> str
    
    def execute(self, command: str) -> ExecuteResponse
    async def aexecute(self, command: str) -> ExecuteResponse
```

## 3. 数据结构详解

### 3.1 FileInfo - 文件元数据

```python
class FileInfo(TypedDict):
    """Structured file listing info."""
    path: str                    # 文件路径（必需）
    is_dir: NotRequired[bool]    # 是否为目录
    size: NotRequired[int]       # 文件大小（字节）
    modified_at: NotRequired[str] # ISO 8601 时间戳
```

**使用场景：**
- `ls_info()` 返回目录列表
- `glob_info()` 返回匹配文件列表

### 3.2 WriteResult - 写入结果

```python
@dataclass
class WriteResult:
    """Result from backend write operations."""
    error: str | None = None                    # 错误信息
    path: str | None = None                     # 写入的文件路径
    files_update: dict[str, Any] | None = None  # State更新（StateBackend使用）
```

**关键设计：**
- `files_update` 用于 StateBackend 返回 State 更新
- 外部存储（FilesystemBackend/StoreBackend）设置为 `None`

**示例：**
```python
# Checkpoint storage (StateBackend)
WriteResult(path="/f.txt", files_update={"/f.txt": {...}})

# External storage (FilesystemBackend)
WriteResult(path="/f.txt", files_update=None)

# Error
WriteResult(error="File exists")
```

### 3.3 EditResult - 编辑结果

```python
@dataclass
class EditResult:
    """Result from backend edit operations."""
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None              # 替换次数
```

### 3.4 GrepMatch - 搜索结果

```python
class GrepMatch(TypedDict):
    """Structured grep match entry."""
    path: str    # 文件路径
    line: int    # 行号（1-indexed）
    text: str    # 匹配行内容
```

### 3.5 FileUploadResponse / FileDownloadResponse

```python
@dataclass
class FileUploadResponse:
    path: str
    error: FileOperationError | None = None

@dataclass
class FileDownloadResponse:
    path: str
    content: bytes | None = None
    error: FileOperationError | None = None
```

## 4. 错误码系统

```python
FileOperationError = Literal[
    "file_not_found",      # 文件不存在
    "permission_denied",   # 权限拒绝
    "is_directory",        # 尝试下载目录
    "invalid_path",        # 路径格式错误
]
```

**设计原则：**
- 标准化错误码便于 LLM 理解和处理
- 支持批量操作的部分成功
- 错误信息结构化，可用于程序处理

## 5. ExecuteResponse - 命令执行结果

```python
@dataclass
class ExecuteResponse:
    """Result of code execution."""
    output: str           # 合并的 stdout 和 stderr
    exit_code: int | None = None  # 进程退出码
    truncated: bool = False       # 输出是否被截断
```

## 6. BackendFactory 类型

```python
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
BACKEND_TYPES = BackendProtocol | BackendFactory
```

**使用场景：**
- 允许延迟创建 Backend，在运行时才获取必要的上下文
- 工厂函数接收 `ToolRuntime` 参数，可以访问 state、config 等

**示例：**
```python
def create_backend(runtime: ToolRuntime) -> BackendProtocol:
    thread_id = runtime.config.get("configurable", {}).get("thread_id")
    return StoreBackend(runtime, namespace=("threads", thread_id))

middleware = FilesystemMiddleware(backend=create_backend)
```

## 7. 同步/异步方法约定

所有 BackendProtocol 方法都提供同步和异步版本：

| 同步方法 | 异步方法 | 默认实现 |
|---------|---------|---------|
| `ls_info()` | `als_info()` | 异步调用 `asyncio.to_thread()` |
| `read()` | `aread()` | 异步调用 `asyncio.to_thread()` |
| `grep_raw()` | `agrep_raw()` | 异步调用 `asyncio.to_thread()` |
| `glob_info()` | `aglob_info()` | 异步调用 `asyncio.to_thread()` |
| `write()` | `awrite()` | 异步调用 `asyncio.to_thread()` |
| `edit()` | `aedit()` | 异步调用 `asyncio.to_thread()` |
| `upload_files()` | `aupload_files()` | 异步调用 `asyncio.to_thread()` |
| `download_files()` | `adownload_files()` | 异步调用 `asyncio.to_thread()` |
| `execute()` | `aexecute()` | 异步调用 `asyncio.to_thread()` |

**设计说明：**
- 基类提供默认的异步实现，通过线程池调用同步方法
- 子类可以覆盖异步方法以提供优化的原生异步实现
- StoreBackend 提供了原生的异步实现（使用 `store.aget()` / `store.aput()`）

## 8. 路径规范

所有 Backend 使用统一的路径规范：

1. **绝对路径**：必须以 `/` 开头
2. **POSIX 风格**：使用 `/` 作为分隔符
3. **规范化**：通过 `os.path.normpath()` 和自定义逻辑处理

**路径验证：**
```python
def _validate_path(path: str) -> str:
    if ".." in path or path.startswith("~"):
        raise ValueError("Path traversal not allowed")
    
    # 拒绝 Windows 绝对路径
    if re.match(r"^[a-zA-Z]:", path):
        raise ValueError("Windows absolute paths are not supported")
    
    normalized = os.path.normpath(path).replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    
    return normalized
```

## 9. 文件数据格式

StateBackend 中文件数据的内部表示：

```python
{
    "/path/to/file.txt": {
        "content": ["line1", "line2", "line3"],  # 按行存储
        "created_at": "2025-01-15T10:30:00Z",
        "modified_at": "2025-01-15T10:35:00Z",
    }
}
```

## 10. 扩展开发指南

### 10.1 实现自定义 Backend

```python
from deepagents.backends.protocol import BackendProtocol, WriteResult, EditResult

class MyCustomBackend(BackendProtocol):
    def __init__(self, config: dict):
        self.config = config
    
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        # 实现读取逻辑
        pass
    
    def write(self, file_path: str, content: str) -> WriteResult:
        # 实现写入逻辑
        return WriteResult(path=file_path, files_update=None)
    
    # ... 实现其他方法
```

### 10.2 支持命令执行

```python
from deepagents.backends.protocol import SandboxBackendProtocol, ExecuteResponse

class MySandboxBackend(SandboxBackendProtocol):
    @property
    def id(self) -> str:
        return "my-sandbox-001"
    
    def execute(self, command: str) -> ExecuteResponse:
        output = run_in_sandbox(command)
        return ExecuteResponse(
            output=output,
            exit_code=0,
            truncated=False
        )
```

## 11. 关键设计决策

### 11.1 为什么使用 TypedDict 而不是 dataclass？

- **灵活性**：TypedDict 允许部分字段缺失（使用 `NotRequired`）
- **兼容性**：更容易与 JSON 数据交互
- **性能**：字典操作在某些场景下更快

### 11.2 为什么区分 `files_update`？

- **StateBackend**：需要返回 State 更新供 LangGraph 处理
- **外部存储**：已经持久化到磁盘/数据库，不需要 State 更新
- **统一接口**：所有 Backend 返回相同的结构，调用方按需处理

### 11.3 错误码为什么使用 Literal 类型？

- **类型安全**：IDE 和类型检查器可以验证错误码
- **文档化**：代码本身说明可能的错误类型
- **LLM 友好**：结构化错误便于 LLM 理解和处理

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
