# Backend 系统设计文档

## 1. 概述

Backend 系统是 DeepAgents 的存储抽象层，提供统一的文件操作接口。它解耦了 Agent 的文件操作逻辑与具体的存储实现，支持多种存储后端的无缝切换。

### 1.1 设计目标

- **统一接口**：所有存储后端提供一致的文件操作 API
- **存储无关**：支持内存、磁盘、数据库、远程存储等多种后端
- **执行支持**：通过 SandboxBackendProtocol 支持命令执行
- **异步支持**：所有操作提供同步和异步两种版本
- **错误标准化**：使用结构化错误码便于 LLM 理解和处理

## 2. 核心协议

### 2.1 BackendProtocol

**文件位置**：`deepagents/backends/protocol.py`

`BackendProtocol` 是所有存储后端必须实现的抽象基类，定义了文件操作的标准接口。

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

### 2.3 数据结构

#### FileInfo - 文件元数据
```python
class FileInfo(TypedDict):
    path: str                    # 文件路径（必需）
    is_dir: NotRequired[bool]    # 是否为目录
    size: NotRequired[int]       # 文件大小（字节）
    modified_at: NotRequired[str] # ISO 8601 时间戳
```

#### WriteResult / EditResult - 操作结果
```python
@dataclass
class WriteResult:
    error: str | None = None                    # 错误信息
    path: str | None = None                     # 写入的文件路径
    files_update: dict[str, Any] | None = None  # 状态更新（StateBackend 使用）

@dataclass
class EditResult:
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None              # 替换次数
```

#### 错误码
```python
FileOperationError = Literal[
    "file_not_found",      # 文件不存在
    "permission_denied",   # 权限拒绝
    "is_directory",        # 尝试下载目录
    "invalid_path",        # 路径格式错误
]
```

## 3. 后端实现

### 3.1 StateBackend

**文件位置**：`deepagents/backends/state.py`

使用 LangGraph 的 State 存储文件，数据在对话线程内持久化，但跨线程不保留。

```python
class StateBackend(BackendProtocol):
    def __init__(self, runtime: "ToolRuntime"):
        self.runtime = runtime  # 通过 runtime 访问 state
```

#### 存储格式
```python
# State 中的 files 结构
{
    "/path/to/file.txt": {
        "content": ["line1", "line2", "line3"],  # 按行存储
        "created_at": "2025-01-15T10:30:00Z",
        "modified_at": "2025-01-15T10:35:00Z",
    }
}
```

#### 特点
- ✅ 无需外部依赖
- ✅ 自动随 StateGraph checkpoint 持久化
- ✅ 适合短期对话场景
- ❌ 跨线程数据丢失
- ❌ 不支持大文件存储

### 3.2 FilesystemBackend

**文件位置**：`deepagents/backends/filesystem.py`

将文件存储在本地文件系统。

```python
class FilesystemBackend(BackendProtocol):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
```

#### 特点
- ✅ 数据持久化存储
- ✅ 支持大文件
- ✅ 可与其他工具共享文件
- ✅ grep 操作支持 ripgrep 加速
- ❌ 需要文件系统权限
- ❌ 无内置命令执行（需配合 LocalShellBackend）

### 3.3 StoreBackend

**文件位置**：`deepagents/backends/store.py`

使用 LangGraph 的 Store API 进行持久化存储，适合生产环境。

```python
class StoreBackend(BackendProtocol):
    def __init__(
        self, 
        runtime: "ToolRuntime",
        *, 
        namespace: NamespaceFactory | None = None
    )
```

#### 特点
- ✅ 持久化存储
- ✅ 支持命名空间隔离
- ✅ 适合多租户场景
- ❌ 需要配置 Store（如 Redis、Postgres）

### 3.4 CompositeBackend

**文件位置**：`deepagents/backends/composite.py`

组合多个后端，根据路径前缀路由到不同后端。

```python
class CompositeBackend(BackendProtocol):
    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol] | None = None
    )
    # 示例：routes={"/memories/": store_backend, "/sandbox/": sandbox_backend}
```

#### 特点
- ✅ 灵活的路由配置
- ✅ 混合存储策略（如临时文件用 State，持久数据用 Filesystem）
- ✅ 支持执行能力的动态检测

### 3.5 沙箱后端（Sandbox Providers）

通过 `SandboxBackendProtocol` 支持远程执行环境：

| 实现 | 提供商 | 说明 |
|------|--------|------|
| HarborSandbox | DeepAgents Harbor | 评估框架内置 |
| DaytonaSandbox | Daytona | 远程开发环境 |
| ModalSandbox | Modal | 无服务器计算 |
| RunloopSandbox | Runloop | 快速启动沙箱 |

## 4. 后端对比

| 特性 | StateBackend | FilesystemBackend | StoreBackend | CompositeBackend |
|------|-------------|-------------------|--------------|------------------|
| 存储位置 | 内存（State） | 本地磁盘 | 外部存储 | 混合 |
| 持久化 | 仅同线程 | 是 | 是 | 取决于配置 |
| 大文件支持 | 有限 | 是 | 是 | 取决于配置 |
| 命令执行 | ❌ | ❌ | ❌ | 取决于配置 |
| 配置复杂度 | 低 | 低 | 中 | 高 |
| 适用场景 | 开发/测试 | 本地开发 | 生产环境 | 复杂需求 |

## 5. 工具集成

### 5.1 在 Middleware 中使用 Backend

```python
class FilesystemMiddleware(AgentMiddleware):
    def __init__(self, backend: BackendProtocol | BackendFactory | None = None):
        # 支持直接传入 Backend 实例或工厂函数
        self.backend = backend or (lambda rt: StateBackend(rt))
    
    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        # 运行时解析 Backend
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend
```

### 5.2 Backend Factory 模式

允许延迟创建 Backend，在运行时才获取必要的上下文：

```python
# 工厂函数签名
BackendFactory = Callable[[ToolRuntime], BackendProtocol]

# 使用示例
def create_backend(runtime: ToolRuntime) -> BackendProtocol:
    thread_id = runtime.config.get("configurable", {}).get("thread_id")
    return StoreBackend(runtime, namespace=("threads", thread_id))

middleware = FilesystemMiddleware(backend=create_backend)
```

## 6. 工具执行支持

### 6.1 执行能力检测

```python
def _supports_execution(backend: BackendProtocol) -> bool:
    # CompositeBackend 特殊处理
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)
    
    # 其他 Backend 直接检查
    return isinstance(backend, SandboxBackendProtocol)
```

### 6.2 执行流程

```
用户调用 execute 工具
        │
        ▼
┌─────────────────┐
│ 检查 Backend    │ ──是否实现 SandboxBackendProtocol？
│ 执行能力        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  不支持      支持
    │         │
    ▼         ▼
返回错误   执行命令
           │
           ▼
    返回 ExecuteResponse
    (output, exit_code, truncated)
```

## 7. 工具结果大内容处理

### 7.1 内容过大检测

```python
NUM_CHARS_PER_TOKEN = 4  # 保守估计

def is_content_too_large(content: str, token_limit: int) -> bool:
    return len(content) > NUM_CHARS_PER_TOKEN * token_limit
```

### 7.2 内容转存流程

当工具结果超过 `tool_token_limit_before_evict`（默认 20000 tokens）时：

1. 将完整内容写入 Backend（路径：`/large_tool_results/{tool_call_id}`）
2. 创建预览（显示头部和尾部内容）
3. 返回包含文件路径引用的摘要消息
4. LLM 可以通过 `read_file` 读取完整内容

```python
TOO_LARGE_TOOL_MSG = """Tool result too large, saved at: {file_path}

Preview:
{content_sample}
"""
```

## 8. 扩展开发

### 8.1 实现自定义 Backend

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

### 8.2 支持命令执行

```python
from deepagents.backends.protocol import SandboxBackendProtocol, ExecuteResponse

class MySandboxBackend(SandboxBackendProtocol):
    @property
    def id(self) -> str:
        return "my-sandbox-001"
    
    def execute(self, command: str) -> ExecuteResponse:
        # 在沙箱中执行命令
        output = run_in_sandbox(command)
        return ExecuteResponse(
            output=output,
            exit_code=0,
            truncated=False
        )
```

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
