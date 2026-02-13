# CompositeBackend 实现详解

## 1. 概述

`CompositeBackend` 是 DeepAgents 的组合存储后端，根据路径前缀将文件操作路由到不同的后端实现。它位于 `deepagents/backends/composite.py`，允许在同一个 Agent 中使用多种存储策略。

## 2. 核心特性

| 特性 | 说明 |
|------|------|
| 路由机制 | 基于路径前缀的路由 |
| 多后端 | 支持组合多个后端 |
| 混合存储 | 临时数据用 State，持久数据用 Store |
| 执行能力 | 动态检测默认后端是否支持命令执行 |

## 3. 使用场景

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

# 配置：临时文件用 State，持久数据用 Store
composite = CompositeBackend(
    default=StateBackend(runtime),
    routes={
        "/memories/": StoreBackend(runtime),
        "/cache/": StoreBackend(runtime),
    }
)

# 写入临时文件（StateBackend）
composite.write("/temp.txt", "ephemeral data")

# 写入持久数据（StoreBackend）
composite.write("/memories/note.md", "persistent data")
```

## 4. 类定义与初始化

```python
class CompositeBackend(BackendProtocol):
    """Routes file operations to different backends by path prefix.
    
    Matches paths against route prefixes (longest first) and delegates to the
    corresponding backend. Unmatched paths use the default backend.
    
    Attributes:
        default: Backend for paths that don't match any route.
        routes: Map of path prefixes to backends.
        sorted_routes: Routes sorted by length (longest first) for correct matching.
    """

    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        """Initialize composite backend.
        
        Args:
            default: Backend for paths that don't match any route.
            routes: Map of path prefixes to backends. Prefixes must start with "/"
                and should end with "/" (e.g., "/memories/").
        """
        # 默认后端
        self.default = default

        # 虚拟路由
        self.routes = routes

        # 按长度排序（最长的优先），确保正确匹配
        self.sorted_routes = sorted(
            routes.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
```

## 5. 路由机制

### 5.1 _get_backend_and_key

```python
def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
    """Get backend for path and strip route prefix.
    
    Args:
        key: File path to route.
    
    Returns:
        Tuple of (backend, stripped_path). The stripped path has the route
        prefix removed but keeps the leading slash.
    
    Examples:
        "/memories/notes.txt" → (store_backend, "/notes.txt")
        "/memories/" → (store_backend, "/")
        "/other/file.txt" → (default_backend, "/other/file.txt")
    """
    # 按长度排序检查路由（最长的优先）
    for prefix, backend in self.sorted_routes:
        if key.startswith(prefix):
            # 去掉前缀，确保保留前导斜杠
            # 例如："/memories/notes.txt" → "/notes.txt"
            #       "/memories/" → "/"
            suffix = key[len(prefix):]
            stripped_key = f"/{suffix}" if suffix else "/"
            return backend, stripped_key

    return self.default, key
```

### 5.2 路由匹配逻辑

```
路径: /memories/user/profile.txt
路由表:
  /memories/ → StoreBackend
  /memories/user/ → StoreBackend (更具体的匹配)

匹配过程:
1. 检查 /memories/user/ → 匹配！→ 使用对应 Backend
2. （不会检查 /memories/，因为已经找到匹配）
```

## 6. 核心方法实现

### 6.1 ls_info / als_info - 列出目录

```python
def ls_info(self, path: str) -> list[FileInfo]:
    """List directory contents (non-recursive).
    
    Special cases:
    - Path matches a route: lists only that backend
    - Path is "/": aggregates default backend plus virtual route directories
    - Path doesn't match any route: lists only default backend
    """
    # 检查路径是否匹配特定路由
    for route_prefix, backend in self.sorted_routes:
        if path.startswith(route_prefix.rstrip("/")):
            # 只查询匹配的路由后端
            suffix = path[len(route_prefix):]
            search_path = f"/{suffix}" if suffix else "/"
            infos = backend.ls_info(search_path)
            
            # 恢复前缀到返回的路径
            prefixed: list[FileInfo] = []
            for fi in infos:
                fi = dict(fi)
                fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                prefixed.append(fi)
            return prefixed

    # 在根目录，聚合默认后端和所有路由
    if path == "/":
        results: list[FileInfo] = []
        results.extend(self.default.ls_info(path))
        
        for route_prefix, backend in self.sorted_routes:
            # 添加路由本身作为目录（例如 /memories/）
            results.append({
                "path": route_prefix,
                "is_dir": True,
                "size": 0,
                "modified_at": "",
            })

        results.sort(key=lambda x: x.get("path", ""))
        return results

    # 路径不匹配任何路由：只查询默认后端
    return self.default.ls_info(path)
```

### 6.2 read / aread - 读取文件

```python
def read(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content, routing to appropriate backend."""
    backend, stripped_key = self._get_backend_and_key(file_path)
    return backend.read(stripped_key, offset=offset, limit=limit)

async def aread(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Async version of read."""
    backend, stripped_key = self._get_backend_and_key(file_path)
    return await backend.aread(stripped_key, offset=offset, limit=limit)
```

### 6.3 write / awrite - 写入文件

```python
def write(
    self,
    file_path: str,
    content: str,
) -> WriteResult:
    """Create a new file, routing to appropriate backend."""
    backend, stripped_key = self._get_backend_and_key(file_path)
    res = backend.write(stripped_key, content)
    
    # 如果是 state-backed 更新且默认后端有 state，合并更新
    # 这样列表可以反映变化
    if res.files_update:
        try:
            runtime = getattr(self.default, "runtime", None)
            if runtime is not None:
                state = runtime.state
                files = state.get("files", {})
                files.update(res.files_update)
                state["files"] = files
        except Exception:
            pass
    
    return res

async def awrite(
    self,
    file_path: str,
    content: str,
) -> WriteResult:
    """Async version of write."""
    backend, stripped_key = self._get_backend_and_key(file_path)
    res = await backend.awrite(stripped_key, content)
    
    # 同上，合并 state 更新
    if res.files_update:
        try:
            runtime = getattr(self.default, "runtime", None)
            if runtime is not None:
                state = runtime.state
                files = state.get("files", {})
                files.update(res.files_update)
                state["files"] = files
        except Exception:
            pass
    
    return res
```

**关键设计：**
- 将 StateBackend 的更新同步到默认后端的 state
- 确保 `ls_info("/")` 能正确显示所有文件

### 6.4 edit / aedit - 编辑文件

```python
def edit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditResult:
    """Edit a file, routing to appropriate backend."""
    backend, stripped_key = self._get_backend_and_key(file_path)
    res = backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
    
    # 同步 state 更新到默认后端
    if res.files_update:
        try:
            runtime = getattr(self.default, "runtime", None)
            if runtime is not None:
                state = runtime.state
                files = state.get("files", {})
                files.update(res.files_update)
                state["files"] = files
        except Exception:
            pass
    
    return res
```

### 6.5 grep_raw / agrep_raw - 搜索文件

```python
def grep_raw(
    self,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """Search files for literal text pattern.
    
    Routing behavior:
    - Specific route path: searches one backend
    - "/" or None: searches all backends and merges results
    - Other paths: searches only default backend
    """
    # 如果路径匹配特定路由，只搜索该后端
    for route_prefix, backend in self.sorted_routes:
        if path is not None and path.startswith(route_prefix.rstrip("/")):
            search_path = path[len(route_prefix) - 1:]
            raw = backend.grep_raw(
                pattern,
                search_path if search_path else "/",
                glob
            )
            if isinstance(raw, str):
                return raw
            # 恢复路径前缀
            return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

    # 如果路径是 None 或 "/"，搜索所有后端并合并结果
    if path is None or path == "/":
        all_matches: list[GrepMatch] = []
        
        # 搜索默认后端
        raw_default = self.default.grep_raw(pattern, path, glob)
        if isinstance(raw_default, str):
            return raw_default
        all_matches.extend(raw_default)

        # 搜索所有路由后端
        for route_prefix, backend in self.routes.items():
            raw = backend.grep_raw(pattern, "/", glob)
            if isinstance(raw, str):
                return raw
            # 添加路由前缀到匹配路径
            all_matches.extend(
                {**m, "path": f"{route_prefix[:-1]}{m['path']}"}
                for m in raw
            )

        return all_matches

    # 指定路径但不匹配任何路由 - 只搜索默认后端
    return self.default.grep_raw(pattern, path, glob)
```

### 6.6 glob_info / aglob_info - 文件匹配

```python
def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
    """Find files matching a glob pattern across backends."""
    results: list[FileInfo] = []

    # 基于路径路由，而不是模式
    for route_prefix, backend in self.sorted_routes:
        if path.startswith(route_prefix.rstrip("/")):
            suffix = path[len(route_prefix) - 1:]
            infos = backend.glob_info(pattern, search_path if suffix else "/")
            # 恢复前缀
            return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

    # 路径不匹配任何特定路由 - 搜索默认后端和所有路由后端
    results.extend(self.default.glob_info(pattern, path))

    for route_prefix, backend in self.routes.items():
        infos = backend.glob_info(pattern, "/")
        results.extend(
            {**fi, "path": f"{route_prefix[:-1]}{fi['path']}"}
            for fi in infos
        )

    # 确定性排序
    results.sort(key=lambda x: x.get("path", ""))
    return results
```

## 7. 命令执行支持

CompositeBackend 支持命令执行，但仅限于默认后端：

```python
def execute(self, command: str) -> ExecuteResponse:
    """Execute shell command via default backend.
    
    Raises:
        NotImplementedError: If default backend doesn't implement SandboxBackendProtocol.
    """
    if isinstance(self.default, SandboxBackendProtocol):
        return self.default.execute(command)

    raise NotImplementedError(
        "Default backend doesn't support command execution (SandboxBackendProtocol). "
        "To enable execution, provide a default backend that implements SandboxBackendProtocol."
    )

async def aexecute(self, command: str) -> ExecuteResponse:
    """Async version of execute."""
    if isinstance(self.default, SandboxBackendProtocol):
        return await self.default.aexecute(command)

    raise NotImplementedError(
        "Default backend doesn't support command execution (SandboxBackendProtocol)."
    )
```

**设计说明：**
- 命令执行只通过默认后端进行
- 需要默认后端实现 `SandboxBackendProtocol`
- 这是安全设计，避免跨后端执行命令

## 8. 批量文件操作

### 8.1 upload_files / aupload_files

```python
def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
    """Upload multiple files, batching by backend for efficiency.
    
    Groups files by their target backend, calls each backend's upload_files
    once with all files for that backend, then merges results in original order.
    """
    from collections import defaultdict

    # 预分配结果列表
    results: list[FileUploadResponse | None] = [None] * len(files)

    # 按后端分组文件，记录原始索引
    backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

    for idx, (path, content) in enumerate(files):
        backend, stripped_path = self._get_backend_and_key(path)
        backend_batches[backend].append((idx, stripped_path, content))

    # 处理每个后端的批次
    for backend, batch in backend_batches.items():
        # 提取数据
        indices, stripped_paths, contents = zip(*batch, strict=False)
        batch_files = list(zip(stripped_paths, contents, strict=False))

        # 调用后端上传
        batch_responses = backend.upload_files(batch_files)

        # 将响应放回原始位置
        for i, orig_idx in enumerate(indices):
            results[orig_idx] = FileUploadResponse(
                path=files[orig_idx][0],  # 原始路径
                error=batch_responses[i].error if i < len(batch_responses) else None,
            )

    return results  # type: ignore[return-value]
```

**优化点：**
- 按后端批量处理，减少后端调用次数
- 保持原始顺序返回结果
- 使用原始路径（而非 stripped 路径）返回

### 8.2 download_files / adownload_files

```python
def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
    """Download multiple files, batching by backend for efficiency."""
    results: list[FileDownloadResponse | None] = [None] * len(paths)
    backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

    # 按后端分组
    for idx, path in enumerate(paths):
        backend, stripped_path = self._get_backend_and_key(path)
        backend_batches[backend].append((idx, stripped_path))

    # 处理每个后端的批次
    for backend, batch in backend_batches.items():
        indices, stripped_paths = zip(*batch, strict=False)
        batch_responses = backend.download_files(list(stripped_paths))

        # 将响应放回原始位置
        for i, orig_idx in enumerate(indices):
            results[orig_idx] = FileDownloadResponse(
                path=paths[orig_idx],  # 原始路径
                content=batch_responses[i].content if i < len(batch_responses) else None,
                error=batch_responses[i].error if i < len(batch_responses) else None,
            )

    return results  # type: ignore[return-value]
```

## 9. 使用示例

### 9.1 混合存储策略

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents import create_deep_agent

# 创建复合后端
backend = CompositeBackend(
    default=StateBackend,  # 临时文件
    routes={
        "/memories/": StoreBackend,      # 持久化记忆
        "/skills/": StoreBackend,        # 技能定义
        "/conversation_history/": StoreBackend,  # 对话历史
    }
)

agent = create_deep_agent(backend=backend)
```

### 9.2 开发/生产混合

```python
# 开发环境：所有数据用 StateBackend
backend_dev = CompositeBackend(
    default=StateBackend
)

# 生产环境：关键数据持久化
backend_prod = CompositeBackend(
    default=StateBackend,
    routes={
        "/data/": StoreBackend,
    }
)
```

### 9.3 使用 LocalShellBackend 执行命令

```python
from deepagents.backends import CompositeBackend, LocalShellBackend, StoreBackend

backend = CompositeBackend(
    default=LocalShellBackend(root_dir="/workspace", virtual_mode=True),
    routes={
        "/memories/": StoreBackend,
    }
)

# 可以在 /workspace 下执行命令
result = backend.execute("ls -la")  # 通过 LocalShellBackend

# 在 /memories/ 下存储持久化数据
backend.write("/memories/note.txt", "persistent")
```

## 10. 优缺点分析

### 优点
- ✅ 灵活的路由配置
- ✅ 混合存储策略（State + Store）
- ✅ 支持执行能力的动态检测
- ✅ 批量操作优化
- ✅ 统一接口，调用方无感知

### 缺点
- ❌ 配置复杂度较高
- ❌ 路径前缀需要仔细设计
- ❌ 跨后端搜索需要合并结果
- ❌ 命令执行仅限于默认后端

## 11. 适用场景

1. **混合存储需求**：临时文件用 State，持久数据用 Store
2. **多租户隔离**：不同租户路由到不同 StoreBackend
3. **功能分离**：文件操作用 FilesystemBackend，执行用 LocalShellBackend
4. **渐进式迁移**：逐步将数据从 State 迁移到 Store

## 12. 注意事项

1. **路由顺序**：最长前缀优先匹配，设计时注意前缀关系
2. **路径前缀**：建议以 `/` 开头和结尾（如 `/data/`）
3. **State 同步**：StateBackend 的更新会同步到默认后端的 state
4. **错误处理**：批量操作时，每个文件的错误独立处理
5. **执行限制**：命令执行只通过默认后端，不支持跨后端执行

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
