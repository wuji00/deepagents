# 20 - Backend 工具函数详解

## 1. 概述

`backends/utils.py` 提供了后端实现共享的工具函数集合，包括字符串格式化、内容处理和结构化辅助函数。这些函数支持各类后端（StateBackend、FilesystemBackend、StoreBackend）实现一致的行为。

## 2. 核心常量定义

```python
# 空内容警告
EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"

# 行长度限制（5000字符）
MAX_LINE_LENGTH = 5000

# 行号显示宽度
LINE_NUMBER_WIDTH = 6

# 工具结果token限制（约20000 tokens）
TOOL_RESULT_TOKEN_LIMIT = 20000

# 截断提示文本
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"
```

## 3. 内容格式化函数

### 3.1 format_content_with_line_numbers - 带行号的内容格式化

```python
def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """格式化文件内容，添加行号（类似 cat -n）。
    
    特点：
    - 支持超长行分割（MAX_LINE_LENGTH=5000）
    - 分割行使用小数标记（如 5.1, 5.2）
    - 统一的行号宽度（6字符）
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]  # 移除末尾空行
    else:
        lines = content

    result_lines = []
    for i, line in enumerate(lines):
        line_num = i + start_line

        if len(line) <= MAX_LINE_LENGTH:
            # 正常行：固定宽度 + 制表符 + 内容
            result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            # 超长行：分割为多个块
            num_chunks = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for chunk_idx in range(num_chunks):
                start = chunk_idx * MAX_LINE_LENGTH
                end = min(start + MAX_LINE_LENGTH, len(line))
                chunk = line[start:end]
                
                if chunk_idx == 0:
                    # 第一块：使用正常行号
                    result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{chunk}")
                else:
                    # 后续块：使用小数标记（5.1, 5.2）
                    continuation_marker = f"{line_num}.{chunk_idx}"
                    result_lines.append(
                        f"{continuation_marker:>{LINE_NUMBER_WIDTH}}\t{chunk}"
                    )

    return "\n".join(result_lines)
```

**示例输出：**
```
     1	import os
     2	def hello():
     3	    print("Hello, World!")
   5.1	[超长行的第一部分...]
   5.2	[超长行的第二部分...]
```

### 3.2 check_empty_content - 空内容检查

```python
def check_empty_content(content: str) -> str | None:
    """检查内容是否为空，返回警告消息。
    
    Returns:
        警告消息（如果为空），否则 None
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None
```

### 3.3 truncate_if_too_long - 智能截断

```python
def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """如果结果超过token限制则截断。
    
    估算：4字符 ≈ 1 token
    """
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            # 按比例截断并添加提示
            ratio = TOOL_RESULT_TOKEN_LIMIT * 4 / total_chars
            keep_count = int(len(result) * ratio)
            return result[:keep_count] + [TRUNCATION_GUIDANCE]
        return result
    
    # 字符串截断
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[:TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result
```

## 4. 文件数据处理

### 4.1 FileData 相关函数

```python
def file_data_to_string(file_data: dict[str, Any]) -> str:
    """将 FileData 转换为纯字符串内容。
    
    FileData 格式：
    {
        "content": ["line1", "line2", ...],  # 按行分割的内容
        "created_at": "2024-01-01T00:00:00",
        "modified_at": "2024-01-01T00:00:00"
    }
    """
    return "\n".join(file_data["content"])


def create_file_data(
    content: str, 
    created_at: str | None = None
) -> dict[str, Any]:
    """创建 FileData 对象。
    
    自动处理：
    - 字符串到行列表的转换
    - 时间戳生成（ISO格式）
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: dict[str, Any], content: str) -> dict[str, Any]:
    """更新 FileData，保留创建时间戳。
    
    关键：modified_at 更新为当前时间，created_at 保持不变
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": file_data["created_at"],  # 保留原始创建时间
        "modified_at": now,  # 更新修改时间
    }
```

### 4.2 format_read_response - 读取响应格式化

```python
def format_read_response(
    file_data: dict[str, Any],
    offset: int,
    limit: int,
) -> str:
    """格式化文件读取响应，支持偏移和限制。
    
    Args:
        file_data: FileData 字典
        offset: 行偏移（0索引）
        limit: 最大返回行数
    
    Returns:
        带行号的格式化内容或错误消息
    """
    content = file_data_to_string(file_data)
    
    # 空内容检查
    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    # 偏移超出范围检查
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    selected_lines = lines[start_idx:end_idx]
    return format_content_with_line_numbers(
        selected_lines, 
        start_line=start_idx + 1
    )
```

## 5. 字符串处理函数

### 5.1 perform_string_replacement - 字符串替换

```python
def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> tuple[str, int] | str:
    """执行字符串替换，带出现次数验证。
    
    Returns:
        成功: (new_content, occurrences)
        失败: 错误消息字符串
    """
    occurrences = content.count(old_string)

    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"

    if occurrences > 1 and not replace_all:
        return (
            f"Error: String '{old_string}' appears {occurrences} times in file. "
            "Use replace_all=True to replace all instances, or provide a more "
            "specific string with surrounding context."
        )

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences
```

### 5.2 sanitize_tool_call_id - 工具调用ID清理

```python
def sanitize_tool_call_id(tool_call_id: str) -> str:
    """清理 tool_call_id，防止路径遍历和分隔符问题。
    
    替换危险字符：. / \
    """
    sanitized = (
        tool_call_id
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return sanitized
```

**用途：**
- 将 tool_call_id 用作文件名时防止目录遍历攻击
- 确保跨平台兼容性

## 6. 路径处理函数

### 6.1 _normalize_path - 路径规范化

```python
def _normalize_path(path: str | None) -> str:
    """将路径规范化为标准形式。
    
    转换规则：
    - None -> "/"
    - 相对路径 -> 绝对路径（添加前导 /）
    - 移除尾部斜杠（根路径除外）
    
    Examples:
        None -> "/"
        "/dir/" -> "/dir"
        "dir" -> "/dir"
        "/" -> "/"
    """
    path = path or "/"
    if not path or path.strip() == "":
        raise ValueError("Path cannot be empty")

    # 确保前导 /
    normalized = path if path.startswith("/") else "/" + path

    # 移除非根路径的尾部斜杠
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    return normalized
```

### 6.2 _filter_files_by_path - 文件路径过滤

```python
def _filter_files_by_path(
    files: dict[str, Any], 
    normalized_path: str
) -> dict[str, Any]:
    """按规范化路径过滤文件字典。
    
    支持两种匹配模式：
    1. 精确文件匹配：/dir/file.txt
    2. 目录前缀匹配：/dir（匹配 /dir/*）
    """
    # 精确文件匹配
    if normalized_path in files:
        return {normalized_path: files[normalized_path]}

    # 目录前缀匹配
    if normalized_path == "/":
        # 根目录：返回所有以 / 开头的文件
        return {fp: fd for fp, fd in files.items() if fp.startswith("/")}
    
    # 非根目录：添加尾部斜杠进行前缀匹配
    dir_prefix = normalized_path + "/"
    return {fp: fd for fp, fd in files.items() if fp.startswith(dir_prefix)}
```

## 7. 搜索函数

### 7.1 _glob_search_files - Glob 模式搜索

```python
def _glob_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str = "/",
) -> str:
    """使用 glob 模式搜索文件。
    
    特点：
    - 支持 ** 递归匹配
    - 支持 {a,b} 扩展
    - 结果按修改时间排序（最新的在前）
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No files found"

    filtered = _filter_files_by_path(files, normalized_path)

    matches = []
    for file_path, file_data in filtered.items():
        # 计算相对路径用于 glob 匹配
        if normalized_path == "/":
            relative = file_path[1:]  # 移除前导 /
        elif file_path == normalized_path:
            relative = file_path.split("/")[-1]  # 精确文件匹配
        else:
            relative = file_path[len(normalized_path) + 1:]  # 目录内相对路径

        # 使用 wcmatch 进行 glob 匹配
        if wcglob.globmatch(
            relative, 
            pattern, 
            flags=wcglob.BRACE | wcglob.GLOBSTAR
        ):
            matches.append((file_path, file_data["modified_at"]))

    # 按修改时间排序（最新的在前）
    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)
```

### 7.2 _grep_search_files - 内容搜索

```python
def _grep_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """使用正则表达式搜索文件内容。
    
    Args:
        pattern: 正则表达式
        path: 搜索路径
        glob: 可选的文件名过滤模式
        output_mode: 输出格式
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    normalized_path = _normalize_path(path)
    filtered = _filter_files_by_path(files, normalized_path)

    # 应用 glob 过滤
    if glob:
        filtered = {
            fp: fd for fp, fd in filtered.items()
            if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)
        }

    # 执行搜索
    results: dict[str, list[tuple[int, str]]] = {}
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
                results.setdefault(file_path, []).append((line_num, line))

    if not results:
        return "No matches found"
    
    return _format_grep_results(results, output_mode)
```

### 7.3 _format_grep_results - 搜索结果格式化

```python
def _format_grep_results(
    results: dict[str, list[tuple[int, str]]],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """根据输出模式格式化 grep 结果。
    
    Modes:
    - files_with_matches: 仅返回文件路径列表
    - count: 返回每个文件的匹配数
    - content: 返回完整的文件路径、行号和内容
    """
    if output_mode == "files_with_matches":
        return "\n".join(sorted(results.keys()))
    
    if output_mode == "count":
        lines = []
        for file_path in sorted(results.keys()):
            count = len(results[file_path])
            lines.append(f"{file_path}: {count}")
        return "\n".join(lines)
    
    # content 模式
    lines = []
    for file_path in sorted(results.keys()):
        lines.append(f"{file_path}:")
        for line_num, line in results[file_path]:
            lines.append(f"  {line_num}: {line}")
    return "\n".join(lines)
```

## 8. 结构化辅助函数

### 8.1 grep_matches_from_files - 结构化匹配

```python
def grep_matches_from_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """返回结构化的 grep 匹配结果。
    
    与 _grep_search_files 不同：
    - 使用简单子串搜索（非正则）
    - 返回结构化数据而非格式化字符串
    - 便于后端组合使用
    
    Returns:
        GrepMatch 列表或错误消息字符串
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return []

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        filtered = {
            fp: fd for fp, fd in filtered.items()
            if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)
        }

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if pattern in line:  # 简单子串搜索
                matches.append({
                    "path": file_path,
                    "line": int(line_num),
                    "text": line
                })
    return matches
```

### 8.2 build_grep_results_dict - 结果分组

```python
def build_grep_results_dict(
    matches: list[GrepMatch]
) -> dict[str, list[tuple[int, str]]]:
    """将结构化匹配分组为传统字典格式。
    
    用于与现有格式化逻辑兼容。
    """
    grouped: dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped
```

### 8.3 format_grep_matches - 结构化结果格式化

```python
def format_grep_matches(
    matches: list[GrepMatch],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """格式化结构化的 grep 匹配。
    
    组合 build_grep_results_dict 和 _format_grep_results 的便捷函数。
    """
    if not matches:
        return "No matches found"
    return _format_grep_results(build_grep_results_dict(matches), output_mode)
```

## 9. 设计原则

### 9.1 统一错误处理

- 使用字符串返回错误消息（非抛出异常）
- 保持后端实现的一致性
- 便于工具上下文中的错误传播

### 9.2 类型导出

```python
# 向后兼容的类型重导出
FileInfo = _FileInfo
GrepMatch = _GrepMatch
```

### 9.3 结构化 vs 格式化分离

- `_grep_search_files`: 完整格式化流程
- `grep_matches_from_files`: 仅结构化搜索
- `format_grep_matches`: 仅格式化

这种分离支持：
- 后端组合使用（如 CompositeBackend）
- 不同输出需求的灵活处理
- 测试和调试的便利性

## 10. 总结

Backend 工具函数提供了：

1. **内容格式化**：行号、截断、空内容处理
2. **文件数据管理**：创建、更新、转换
3. **路径处理**：规范化、过滤、验证
4. **搜索功能**：glob 和 grep 实现
5. **安全辅助**：tool_call_id 清理
6. **结构化支持**：组合友好的辅助函数

这些函数确保了所有后端实现的一致行为，同时保持代码的DRY（Don't Repeat Yourself）原则。
