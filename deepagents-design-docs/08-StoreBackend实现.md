# StoreBackend 实现详解

## 1. 概述

`StoreBackend` 是 DeepAgents 的持久化存储后端，使用 LangGraph 的 `BaseStore` API 将文件存储在外部存储系统（如 Redis、Postgres）中。它位于 `deepagents/backends/store.py`，提供跨线程、跨会话的持久化存储能力。

## 2. 核心特性

| 特性 | 说明 |
|------|------|
| 存储位置 | 外部存储（Redis、Postgres 等） |
| 持久化 | 跨线程持久化 |
| 命名空间 | 支持多租户隔离 |
| 异步支持 | 原生异步方法实现 |
| 适用场景 | 生产环境 |

## 3. 核心类与类型

### 3.1 BackendContext

```python
@dataclass
class BackendContext(Generic[StateT, ContextT]):
    """Context passed to namespace factory functions."""
    state: StateT
    runtime: "Runtime[ContextT]"

# Type alias for namespace factory functions
NamespaceFactory = Callable[[BackendContext[Any, Any]], tuple[str, ...]]
```

**用途：**
- 传递给命名空间工厂函数，用于动态构建命名空间
- 包含当前 state 和 runtime 上下文

### 3.2 命名空间验证

```python
_NAMESPACE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")

def _validate_namespace(namespace: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a namespace tuple returned by a NamespaceFactory.
    
    Each component must be a non-empty string containing only safe characters:
    alphanumeric (a-z, A-Z, 0-9), hyphen (-), underscore (_), dot (.),
    at sign (@), plus (+), colon (:), and tilde (~).
    
    Characters like `*`, `?`, `[`, `]`, `{`, `}`, etc. are rejected
    to prevent wildcard or glob injection in store lookups.
    """
    if not namespace:
        raise ValueError("Namespace tuple must not be empty.")

    for i, component in enumerate(namespace):
        if not isinstance(component, str):
            raise TypeError(f"Namespace component at index {i} must be a string")
        if not component:
            raise ValueError(f"Namespace component at index {i} must not be empty")
        if not _NAMESPACE_COMPONENT_RE.match(component):
            raise ValueError(
                f"Namespace component at index {i} contains disallowed characters: {component!r}. "
                f"Only alphanumeric characters, hyphens, underscores, dots, @, +, colons, and tildes are allowed."
            )

    return namespace
```

**安全设计：**
- 拒绝通配符字符（`*`, `?`, `[`, `]`, `{`, `}`）
- 防止在 store 查找中的注入攻击

## 4. StoreBackend 类

### 4.1 初始化

```python
class StoreBackend(BackendProtocol):
    """Backend that stores files in LangGraph's BaseStore (persistent).
    
    Uses LangGraph's Store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.
    """

    def __init__(
        self,
        runtime: "ToolRuntime",
        *,
        namespace: NamespaceFactory | None = None
    ):
        """Initialize StoreBackend with runtime.
        
        Args:
            runtime: The ToolRuntime instance providing store access.
            namespace: Optional callable that takes a BackendContext and returns
                a namespace tuple. If None, uses legacy assistant_id detection.
                
        Example:
            namespace=lambda ctx: ("filesystem", ctx.runtime.context.user_id)
        """
        self.runtime = runtime
        self._namespace = namespace
```

### 4.2 Store 获取

```python
def _get_store(self) -> BaseStore:
    """Get the store instance from runtime."""
    store = self.runtime.store
    if store is None:
        raise ValueError("Store is required but not available in runtime")
    return store
```

### 4.3 命名空间获取

```python
def _get_namespace(self) -> tuple[str, ...]:
    """Get the namespace for store operations.
    
    If namespace was provided at init, calls it with a BackendContext.
    Otherwise, uses legacy assistant_id detection from metadata (deprecated).
    """
    if self._namespace is not None:
        state = getattr(self.runtime, "state", None)
        ctx = BackendContext(state=state, runtime=self.runtime)
        return _validate_namespace(self._namespace(ctx))

    return self._get_namespace_legacy()

def _get_namespace_legacy(self) -> tuple[str, ...]:
    """Legacy namespace resolution: check metadata for assistant_id.
    
    Preference order:
    1) Use `self.runtime.config` if present (tests pass this explicitly).
    2) Fallback to `langgraph.config.get_config()` if available.
    3) Default to ("filesystem",).
    """
    warnings.warn(
        "StoreBackend without explicit `namespace` is deprecated. "
        "Pass `namespace=lambda ctx: (...)` to StoreBackend.",
        DeprecationWarning,
        stacklevel=3,
    )
    namespace = "filesystem"

    # 优先使用 runtime 提供的 config
    runtime_cfg = getattr(self.runtime, "config", None)
    if isinstance(runtime_cfg, dict):
        assistant_id = runtime_cfg.get("metadata", {}).get("assistant_id")
        if assistant_id:
            return (assistant_id, namespace)
        return (namespace,)

    # 回退到 langgraph 的 context
    try:
        cfg = get_config()
    except Exception:
        return (namespace,)

    try:
        assistant_id = cfg.get("metadata", {}).get("assistant_id")
    except Exception:
        assistant_id = None

    if assistant_id:
        return (assistant_id, namespace)
    return (namespace,)
```

## 5. 数据转换

### 5.1 Store Item 转 FileData

```python
def _convert_store_item_to_file_data(self, store_item: Item) -> dict[str, Any]:
    """Convert a store Item to FileData format.
    
    Validates that the store item contains required fields:
    - content: list of strings (file lines)
    - created_at: ISO 8601 timestamp string
    - modified_at: ISO 8601 timestamp string
    """
    if "content" not in store_item.value or not isinstance(store_item.value["content"], list):
        raise ValueError(f"Store item does not contain valid content field")
    if "created_at" not in store_item.value or not isinstance(store_item.value["created_at"], str):
        raise ValueError(f"Store item does not contain valid created_at field")
    if "modified_at" not in store_item.value or not isinstance(store_item.value["modified_at"], str):
        raise ValueError(f"Store item does not contain valid modified_at field")
    
    return {
        "content": store_item.value["content"],
        "created_at": store_item.value["created_at"],
        "modified_at": store_item.value["modified_at"],
    }
```

### 5.2 FileData 转 Store Value

```python
def _convert_file_data_to_store_value(self, file_data: dict[str, Any]) -> dict[str, Any]:
    """Convert FileData to a dict suitable for store.put()."""
    return {
        "content": file_data["content"],
        "created_at": file_data["created_at"],
        "modified_at": file_data["modified_at"],
    }
```

## 6. 分页搜索

```python
def _search_store_paginated(
    self,
    store: BaseStore,
    namespace: tuple[str, ...],
    *,
    query: str | None = None,
    filter: dict[str, Any] | None = None,
    page_size: int = 100,
) -> list[Item]:
    """Search store with automatic pagination to retrieve all results.
    
    Continues fetching pages until no more results are available.
    """
    all_items: list[Item] = []
    offset = 0
    while True:
        page_items = store.search(
            namespace,
            query=query,
            filter=filter,
            limit=page_size,
            offset=offset,
        )
        if not page_items:
            break
        all_items.extend(page_items)
        if len(page_items) < page_size:
            break
        offset += page_size

    return all_items
```

## 7. 核心方法实现

### 7.1 ls_info - 列出目录

```python
def ls_info(self, path: str) -> list[FileInfo]:
    """List files and directories in the specified directory (non-recursive)."""
    store = self._get_store()
    namespace = self._get_namespace()

    # 检索所有 items 并在本地过滤
    items = self._search_store_paginated(store, namespace)
    infos: list[FileInfo] = []
    subdirs: set[str] = set()

    normalized_path = path if path.endswith("/") else path + "/"

    for item in items:
        if not str(item.key).startswith(normalized_path):
            continue

        relative = str(item.key)[len(normalized_path):]

        # 如果在子目录中
        if "/" in relative:
            subdir_name = relative.split("/")[0]
            subdirs.add(normalized_path + subdir_name + "/")
            continue

        # 当前目录下的文件
        try:
            fd = self._convert_store_item_to_file_data(item)
        except ValueError:
            continue
        size = len("\n".join(fd.get("content", [])))
        infos.append({
            "path": item.key,
            "is_dir": False,
            "size": int(size),
            "modified_at": fd.get("modified_at", ""),
        })

    # 添加目录
    for subdir in sorted(subdirs):
        infos.append({
            "path": subdir,
            "is_dir": True,
            "size": 0,
            "modified_at": "",
        })

    infos.sort(key=lambda x: x.get("path", ""))
    return infos
```

### 7.2 read / aread - 读取文件

```python
def read(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content with line numbers (sync version)."""
    store = self._get_store()
    namespace = self._get_namespace()
    item: Item | None = store.get(namespace, file_path)

    if item is None:
        return f"Error: File '{file_path}' not found"

    try:
        file_data = self._convert_store_item_to_file_data(item)
    except ValueError as e:
        return f"Error: {e}"

    return format_read_response(file_data, offset, limit)

async def aread(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content with line numbers (async version).
    
    Uses native store async methods to avoid sync calls in async context.
    """
    store = self._get_store()
    namespace = self._get_namespace()
    item: Item | None = await store.aget(namespace, file_path)

    if item is None:
        return f"Error: File '{file_path}' not found"

    try:
        file_data = self._convert_store_item_to_file_data(item)
    except ValueError as e:
        return f"Error: {e}"

    return format_read_response(file_data, offset, limit)
```

### 7.3 write / awrite - 写入文件

```python
def write(
    self,
    file_path: str,
    content: str,
) -> WriteResult:
    """Create a new file with content."""
    store = self._get_store()
    namespace = self._get_namespace()

    # 检查文件是否存在
    existing = store.get(namespace, file_path)
    if existing is not None:
        return WriteResult(
            error=f"Cannot write to {file_path} because it already exists. "
                  f"Read and then make an edit, or write to a new path."
        )

    # 创建新文件
    file_data = create_file_data(content)
    store_value = self._convert_file_data_to_store_value(file_data)
    store.put(namespace, file_path, store_value)
    return WriteResult(path=file_path, files_update=None)

async def awrite(
    self,
    file_path: str,
    content: str,
) -> WriteResult:
    """Create a new file with content (async version)."""
    store = self._get_store()
    namespace = self._get_namespace()

    # 使用 async 方法检查文件是否存在
    existing = await store.aget(namespace, file_path)
    if existing is not None:
        return WriteResult(
            error=f"Cannot write to {file_path} because it already exists."
        )

    file_data = create_file_data(content)
    store_value = self._convert_file_data_to_store_value(file_data)
    await store.aput(namespace, file_path, store_value)
    return WriteResult(path=file_path, files_update=None)
```

### 7.4 edit / aedit - 编辑文件

```python
def edit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditResult:
    """Edit a file by replacing string occurrences."""
    store = self._get_store()
    namespace = self._get_namespace()

    item = store.get(namespace, file_path)
    if item is None:
        return EditResult(error=f"Error: File '{file_path}' not found")

    try:
        file_data = self._convert_store_item_to_file_data(item)
    except ValueError as e:
        return EditResult(error=f"Error: {e}")

    content = file_data_to_string(file_data)
    result = perform_string_replacement(content, old_string, new_string, replace_all)

    if isinstance(result, str):
        return EditResult(error=result)

    new_content, occurrences = result
    new_file_data = update_file_data(file_data, new_content)

    # 更新 store
    store_value = self._convert_file_data_to_store_value(new_file_data)
    store.put(namespace, file_path, store_value)
    return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))

async def aedit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditResult:
    """Edit a file by replacing string occurrences (async version)."""
    store = self._get_store()
    namespace = self._get_namespace()

    # 使用 async 方法获取文件
    item = await store.aget(namespace, file_path)
    if item is None:
        return EditResult(error=f"Error: File '{file_path}' not found")

    try:
        file_data = self._convert_store_item_to_file_data(item)
    except ValueError as e:
        return EditResult(error=f"Error: {e}")

    content = file_data_to_string(file_data)
    result = perform_string_replacement(content, old_string, new_string, replace_all)

    if isinstance(result, str):
        return EditResult(error=result)

    new_content, occurrences = result
    new_file_data = update_file_data(file_data, new_content)

    # 使用 async 方法更新 store
    store_value = self._convert_file_data_to_store_value(new_file_data)
    await store.aput(namespace, file_path, store_value)
    return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
```

### 7.5 grep_raw / agrep_raw - 搜索文件

```python
def grep_raw(
    self,
    pattern: str,
    path: str = "/",
    glob: str | None = None,
) -> list[GrepMatch] | str:
    store = self._get_store()
    namespace = self._get_namespace()
    items = self._search_store_paginated(store, namespace)
    
    files: dict[str, Any] = {}
    for item in items:
        try:
            files[item.key] = self._convert_store_item_to_file_data(item)
        except ValueError:
            continue
    
    return grep_matches_from_files(files, pattern, path, glob)
```

### 7.6 glob_info / aglob_info - 文件匹配

```python
def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
    store = self._get_store()
    namespace = self._get_namespace()
    items = self._search_store_paginated(store, namespace)
    
    files: dict[str, Any] = {}
    for item in items:
        try:
            files[item.key] = self._convert_store_item_to_file_data(item)
        except ValueError:
            continue
    
    result = _glob_search_files(files, pattern, path)
    if result == "No files found":
        return []
    
    paths = result.split("\n")
    infos: list[FileInfo] = []
    for p in paths:
        fd = files.get(p)
        size = len("\n".join(fd.get("content", []))) if fd else 0
        infos.append({
            "path": p,
            "is_dir": False,
            "size": int(size),
            "modified_at": fd.get("modified_at", "") if fd else "",
        })
    return infos
```

### 7.7 upload_files / aupload_files - 上传文件

```python
def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
    """Upload multiple files to the store."""
    store = self._get_store()
    namespace = self._get_namespace()
    responses: list[FileUploadResponse] = []

    for path, content in files:
        content_str = content.decode("utf-8")
        file_data = create_file_data(content_str)
        store_value = self._convert_file_data_to_store_value(file_data)

        store.put(namespace, path, store_value)
        responses.append(FileUploadResponse(path=path, error=None))

    return responses
```

### 7.8 download_files / adownload_files - 下载文件

```python
def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
    """Download multiple files from the store."""
    store = self._get_store()
    namespace = self._get_namespace()
    responses: list[FileDownloadResponse] = []

    for path in paths:
        item = store.get(namespace, path)

        if item is None:
            responses.append(FileDownloadResponse(
                path=path, content=None, error="file_not_found"
            ))
            continue

        file_data = self._convert_store_item_to_file_data(item)
        content_str = file_data_to_string(file_data)
        content_bytes = content_str.encode("utf-8")

        responses.append(FileDownloadResponse(
            path=path, content=content_bytes, error=None
        ))

    return responses
```

## 8. 使用示例

### 8.1 基本使用

```python
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langgraph.store.postgres import PostgresStore

# 创建 store 实例
store = PostgresStore(
    conn_string="postgresql://user:pass@localhost/dbname"
)

# 创建 Agent
agent = create_deep_agent(
    store=store,
    backend=lambda rt: StoreBackend(
        rt,
        namespace=lambda ctx: ("files", "user_123")
    )
)
```

### 8.2 多租户场景

```python
from deepagents.backends import StoreBackend

def create_backend_for_user(runtime):
    user_id = runtime.config.get("configurable", {}).get("user_id")
    return StoreBackend(
        runtime,
        namespace=lambda ctx: ("tenant", user_id, "files")
    )

agent = create_deep_agent(
    store=store,
    backend=create_backend_for_user
)
```

### 8.3 使用 Legacy API（不推荐）

```python
# 使用 assistant_id 自动检测命名空间（已废弃）
agent = create_deep_agent(
    store=store,
    backend=lambda rt: StoreBackend(rt)  # 会发出 DeprecationWarning
)
```

## 9. 与 StateBackend 对比

| 特性 | StateBackend | StoreBackend |
|------|-------------|--------------|
| 存储位置 | 内存 (State) | 外部存储 |
| 持久化 | 仅同线程 | 跨线程 |
| 外部依赖 | 无 | 需要 Store |
| 配置复杂度 | 低 | 中 |
| 适用场景 | 开发/测试 | 生产环境 |
| 多租户支持 | 否 | 是（通过 namespace） |
| 异步方法 | 线程池 | 原生实现 |

## 10. 优缺点分析

### 优点
- ✅ 持久化存储，跨线程可用
- ✅ 支持命名空间隔离
- ✅ 原生异步方法，性能更好
- ✅ 适合多租户场景
- ✅ 支持多种存储后端（Redis、Postgres 等）

### 缺点
- ❌ 需要配置外部存储
- ❌ 配置复杂度较高
- ❌ 依赖 LangGraph Store API
- ❌ 需要理解命名空间概念

## 11. 适用场景

1. **生产环境**：需要持久化的应用
2. **多租户系统**：需要数据隔离的场景
3. **长时间运行**：跨会话保持数据
4. **分布式部署**：多个实例共享存储

## 12. 注意事项

1. **命名空间设计**：合理规划命名空间结构，避免数据混乱
2. **Store 配置**：确保 Store 正确配置并可用
3. **字符编码**：上传/下载时统一使用 UTF-8
4. **大文件**：考虑 Store 的大小限制
5. **并发控制**：Store 通常有内置的并发控制

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
