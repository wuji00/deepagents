# FilesystemBackend 实现详解

## 1. 概述

`FilesystemBackend` 是 DeepAgents 的本地文件系统存储后端，将文件直接存储在本地磁盘上。它位于 `deepagents/backends/filesystem.py`，提供持久化存储和 ripgrep 加速的文件搜索能力。

## 2. 核心特性

| 特性 | 说明 |
|------|------|
| 存储位置 | 本地文件系统 |
| 持久化 | 是 |
| 大文件支持 | 是（支持分页读取） |
| 搜索加速 | 支持 ripgrep |
| 安全模式 | 支持 virtual_mode 路径隔离 |

## 3. 类定义与初始化

```python
class FilesystemBackend(BackendProtocol):
    """Backend that reads and writes files directly from the filesystem.
    
    !!! warning "Security Warning"
        This backend grants agents direct filesystem read/write access. Use with
        caution and only in appropriate environments.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool = False,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize filesystem backend.
        
        Args:
            root_dir: Optional root directory for file operations.
                - If not provided, defaults to the current working directory.
                - When `virtual_mode=False` (default): Only affects relative path
                    resolution. Provides **no security**.
                - When `virtual_mode=True`: All paths are restricted to this
                    directory with traversal protection enabled.
            
            virtual_mode: Enable path-based access restrictions.
                When `True`, all paths are treated as virtual paths anchored to
                `root_dir`. Path traversal (`..`, `~`) is blocked.
                When `False` (default), **no security is provided**.
            
            max_file_size_mb: Maximum file size in megabytes for operations like
                grep's Python fallback search.
        """
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
```

## 4. 路径解析与安全

### 4.1 _resolve_path 方法

```python
def _resolve_path(self, key: str) -> Path:
    """Resolve a file path with security checks.
    
    When `virtual_mode=True`:
    - Treat incoming paths as virtual absolute paths under `self.cwd`
    - Disallow traversal (`..`, `~`)
    - Ensure resolved path stays within root
    
    When `virtual_mode=False`:
    - Preserve legacy behavior: absolute paths are allowed as-is
    - Relative paths resolve under cwd
    
    Raises:
        ValueError: If path traversal is attempted in `virtual_mode` or if the
            resolved path escapes the root directory.
    """
    if self.virtual_mode:
        vpath = key if key.startswith("/") else "/" + key
        if ".." in vpath or vpath.startswith("~"):
            raise ValueError("Path traversal not allowed")
        full = (self.cwd / vpath.lstrip("/")).resolve()
        try:
            full.relative_to(self.cwd)
        except ValueError:
            raise ValueError(f"Path:{full} outside root directory: {self.cwd}") from None
        return full

    path = Path(key)
    if path.is_absolute():
        return path
    return (self.cwd / path).resolve()
```

### 4.2 路径解析模式对比

| 模式 | 绝对路径 | 相对路径 | `..` | `~` | 安全级别 |
|------|---------|---------|------|-----|---------|
| `virtual_mode=False` (默认) | 允许任意 | 相对于 `root_dir` | 允许 | 允许 | 无安全 |
| `virtual_mode=True` | 映射到 `root_dir` | 映射到 `root_dir` | 拒绝 | 拒绝 | 路径隔离 |

**重要警告：**
- 默认情况下 (`virtual_mode=False`)，即使设置了 `root_dir`，也无法阻止访问系统上的任意文件
- 只有启用 `virtual_mode=True` 才能获得路径级别的访问控制

## 5. 核心方法实现

### 5.1 ls_info - 列出目录

```python
def ls_info(self, path: str) -> list[FileInfo]:
    """List files and directories in the specified directory (non-recursive)."""
    dir_path = self._resolve_path(path)
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    results: list[FileInfo] = []
    cwd_str = str(self.cwd)
    if not cwd_str.endswith("/"):
        cwd_str += "/"

    try:
        for child_path in dir_path.iterdir():
            try:
                is_file = child_path.is_file()
                is_dir = child_path.is_dir()
            except OSError:
                continue

            abs_path = str(child_path)
            
            if not self.virtual_mode:
                # 非虚拟模式：使用绝对路径
                if is_file:
                    try:
                        st = child_path.stat()
                        results.append({
                            "path": abs_path,
                            "is_dir": False,
                            "size": int(st.st_size),
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                elif is_dir:
                    # 添加目录...
            else:
                # 虚拟模式：转换为虚拟路径
                if abs_path.startswith(cwd_str):
                    relative_path = abs_path[len(cwd_str):]
                else:
                    relative_path = abs_path
                virt_path = "/" + relative_path
                # 添加文件信息...
    except (OSError, PermissionError):
        pass

    results.sort(key=lambda x: x.get("path", ""))
    return results
```

### 5.2 read - 读取文件

```python
def read(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content with line numbers.
    
    Features:
    - Uses O_NOFOLLOW to avoid symlink traversal (where available)
    - Supports pagination (offset/limit)
    - Returns line-numbered output
    """
    resolved_path = self._resolve_path(file_path)

    if not resolved_path.exists() or not resolved_path.is_file():
        return f"Error: File '{file_path}' not found"

    try:
        # 使用 O_NOFOLLOW 避免跟随符号链接
        fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            content = f.read()

        empty_msg = check_empty_content(content)
        if empty_msg:
            return empty_msg

        lines = content.splitlines()
        start_idx = offset
        end_idx = min(start_idx + limit, len(lines))

        if start_idx >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

        selected_lines = lines[start_idx:end_idx]
        return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)
    except (OSError, UnicodeDecodeError) as e:
        return f"Error reading file '{file_path}': {e}"
```

**安全特性：**
- 使用 `O_NOFOLLOW` 标志防止符号链接遍历攻击
- 优雅处理各种错误（文件不存在、权限错误、编码错误）

### 5.3 write - 写入文件

```python
def write(
    self,
    file_path: str,
    content: str,
) -> WriteResult:
    """Create a new file with content.
    
    Security:
    - Uses O_NOFOLLOW to prevent writing through symlinks
    - Creates parent directories if needed
    """
    resolved_path = self._resolve_path(file_path)

    if resolved_path.exists():
        return WriteResult(
            error=f"Cannot write to {file_path} because it already exists. "
                  f"Read and then make an edit, or write to a new path."
        )

    try:
        # 创建父目录
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 O_NOFOLLOW 避免写入符号链接
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(resolved_path, flags, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)

        return WriteResult(path=file_path, files_update=None)
    except (OSError, UnicodeEncodeError) as e:
        return WriteResult(error=f"Error writing file '{file_path}': {e}")
```

### 5.4 edit - 编辑文件

```python
def edit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditResult:
    """Edit a file by replacing string occurrences."""
    resolved_path = self._resolve_path(file_path)

    if not resolved_path.exists() or not resolved_path.is_file():
        return EditResult(error=f"Error: File '{file_path}' not found")

    try:
        # 安全读取
        fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            content = f.read()

        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result

        # 安全写入
        flags = os.O_WRONLY | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(resolved_path, flags)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_content)

        return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
    except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
        return EditResult(error=f"Error editing file '{file_path}': {e}")
```

## 6. 搜索功能实现

### 6.1 grep_raw - 文件搜索

```python
def grep_raw(
    self,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """Search for a literal text pattern in files.
    
    Uses ripgrep if available, falling back to Python search.
    """
    try:
        base_full = self._resolve_path(path or ".")
    except ValueError:
        return []

    if not base_full.exists():
        return []

    # 优先尝试 ripgrep
    results = self._ripgrep_search(pattern, base_full, glob)
    if results is None:
        # 回退到 Python 搜索
        results = self._python_search(re.escape(pattern), base_full, glob)

    matches: list[GrepMatch] = []
    for fpath, items in results.items():
        for line_num, line_text in items:
            matches.append({"path": fpath, "line": int(line_num), "text": line_text})
    return matches
```

### 6.2 ripgrep 搜索

```python
def _ripgrep_search(
    self,
    pattern: str,
    base_full: Path,
    include_glob: str | None
) -> dict[str, list[tuple[int, str]]] | None:
    """Search using ripgrep with fixed-string (literal) mode.
    
    Returns None if ripgrep is unavailable or times out.
    """
    cmd = ["rg", "--json", "-F"]  # -F 启用固定字符串（字面量）模式
    if include_glob:
        cmd.extend(["--glob", include_glob])
    cmd.extend(["--", pattern, str(base_full)])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    results: dict[str, list[tuple[int, str]]] = {}
    for line in proc.stdout.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") != "match":
            continue
        pdata = data.get("data", {})
        ftext = pdata.get("path", {}).get("text")
        if not ftext:
            continue
        p = Path(ftext)
        if self.virtual_mode:
            try:
                virt = "/" + str(p.resolve().relative_to(self.cwd))
            except Exception:
                continue
        else:
            virt = str(p)
        ln = pdata.get("line_number")
        lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
        if ln is None:
            continue
        results.setdefault(virt, []).append((int(ln), lt))

    return results
```

**特点：**
- 使用 `-F` 标志进行字面量搜索（非正则）
- JSON 输出便于解析
- 30 秒超时保护
- 自动处理虚拟路径转换

### 6.3 Python 回退搜索

```python
def _python_search(
    self,
    pattern: str,
    base_full: Path,
    include_glob: str | None
) -> dict[str, list[tuple[int, str]]]:
    """Fallback search using Python when ripgrep is unavailable."""
    regex = re.compile(pattern)
    results: dict[str, list[tuple[int, str]]] = {}
    root = base_full if base_full.is_dir() else base_full.parent

    for fp in root.rglob("*"):
        try:
            if not fp.is_file():
                continue
        except (PermissionError, OSError):
            continue
        if include_glob and not wcglob.globmatch(fp.name, include_glob, flags=wcglob.BRACE):
            continue
        try:
            if fp.stat().st_size > self.max_file_size_bytes:
                continue
        except OSError:
            continue
        try:
            content = fp.read_text()
        except (UnicodeDecodeError, PermissionError, OSError):
            continue
        for line_num, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                # 处理虚拟路径...
                results.setdefault(virt_path, []).append((line_num, line))

    return results
```

**保护措施：**
- 跳过超大文件（默认 >10MB）
- 处理权限错误
- 处理编码错误
- 递归搜索所有子目录

### 6.4 glob_info - 文件匹配

```python
def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
    """Find files matching a glob pattern."""
    if pattern.startswith("/"):
        pattern = pattern.lstrip("/")

    search_path = self.cwd if path == "/" else self._resolve_path(path)
    if not search_path.exists() or not search_path.is_dir():
        return []

    results: list[FileInfo] = []
    try:
        for matched_path in search_path.rglob(pattern):
            try:
                is_file = matched_path.is_file()
            except (PermissionError, OSError):
                continue
            if not is_file:
                continue
            # 处理路径和元数据...
    except (OSError, ValueError):
        pass

    results.sort(key=lambda x: x.get("path", ""))
    return results
```

## 7. 批量文件操作

### 7.1 upload_files

```python
def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
    """Upload multiple files to the filesystem."""
    responses: list[FileUploadResponse] = []
    for path, content in files:
        try:
            resolved_path = self._resolve_path(path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "wb") as f:
                f.write(content)

            responses.append(FileUploadResponse(path=path, error=None))
        except FileNotFoundError:
            responses.append(FileUploadResponse(path=path, error="file_not_found"))
        except PermissionError:
            responses.append(FileUploadResponse(path=path, error="permission_denied"))
        except (ValueError, OSError):
            responses.append(FileUploadResponse(path=path, error="invalid_path"))

    return responses
```

### 7.2 download_files

```python
def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
    """Download multiple files from the filesystem."""
    responses: list[FileDownloadResponse] = []
    for path in paths:
        try:
            resolved_path = self._resolve_path(path)
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as f:
                content = f.read()
            responses.append(FileDownloadResponse(path=path, content=content, error=None))
        except FileNotFoundError:
            responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
        except PermissionError:
            responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
        except IsADirectoryError:
            responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
        except ValueError:
            responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))

    return responses
```

## 8. 使用示例

### 8.1 基本使用（不推荐用于生产）

```python
from deepagents.backends import FilesystemBackend

# 默认：无安全限制！
backend = FilesystemBackend()

# 读取文件
content = backend.read("/etc/passwd")  # 可以读取任意文件！
```

### 8.2 安全模式（推荐）

```python
from deepagents.backends import FilesystemBackend

# 启用 virtual_mode 进行路径隔离
backend = FilesystemBackend(
    root_dir="/home/user/project",
    virtual_mode=True,  # 必须启用！
)

# 虚拟路径自动映射到 root_dir
content = backend.read("/src/main.py")  # 实际读取 /home/user/project/src/main.py

# 路径遍历被拒绝
try:
    backend.read("/../etc/passwd")  # ValueError: Path traversal not allowed
except ValueError as e:
    print(e)
```

### 8.3 在 create_deep_agent 中使用

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(
        root_dir="/workspace",
        virtual_mode=True,
    )
)
```

## 9. 安全建议

### 9.1 开发环境
- 使用 `virtual_mode=True` 限制访问范围
- 设置适当的 `root_dir`
- 避免使用默认配置（无限制）

### 9.2 CI/CD 环境
- 确保构建环境隔离
- 排除敏感文件路径
- 考虑使用 `StateBackend` 或 `StoreBackend` 代替

### 9.3 生产环境
- **不推荐**直接使用 FilesystemBackend
- 使用 `SandboxBackend` 实现（Docker、VM 等）
- 配合 Human-in-the-Loop 中间件审核敏感操作

## 10. 优缺点分析

### 优点
- ✅ 数据持久化存储
- ✅ 支持大文件
- ✅ 可与其他工具共享文件
- ✅ grep 操作支持 ripgrep 加速
- ✅ 灵活的虚拟路径模式

### 缺点
- ❌ 需要文件系统权限
- ❌ 无内置命令执行（需配合 LocalShellBackend）
- ❌ 默认无安全限制
- ❌ 不适合多租户场景

## 11. 适用场景

1. **本地开发 CLI**：编码助手、开发工具
2. **个人项目**：单用户文件操作
3. **CI/CD 流水线**：有适当隔离的构建环境
4. **数据导入/导出**：批量文件处理

## 12. 注意事项

1. **安全性**：默认配置无安全限制，务必启用 `virtual_mode=True`
2. **并发**：不处理文件锁，并发写入可能导致数据损坏
3. **符号链接**：使用 `O_NOFOLLOW` 防止符号链接遍历
4. **编码**：假设 UTF-8 编码，其他编码可能导致错误
5. **路径格式**：内部使用 POSIX 路径（`/`），自动处理平台差异

---

*文档生成时间：2025年2月*
*基于 DeepAgents 源码分析*
