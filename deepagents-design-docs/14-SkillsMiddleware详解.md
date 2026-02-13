# 14 - SkillsMiddleware 详解

## 1. 概述

`SkillsMiddleware` 实现了 Anthropic 的 Agent Skills 规范，支持从后端存储加载技能（Skill）并在系统提示中展示。采用**渐进式披露（Progressive Disclosure）**模式，让代理先看到技能元数据，需要时再读取完整技能文档。

## 2. 核心概念

### 2.1 Skill 结构

```
/skills/user/web-research/
├── SKILL.md          # 必需：YAML frontmatter + markdown 说明
└── helper.py         # 可选：辅助文件
```

### 2.2 SKILL.md 格式

```markdown
---
name: web-research
description: Structured approach to conducting thorough web research
license: MIT
---

# Web Research Skill

## When to Use
- User asks you to research a topic
...
```

### 2.3 核心数据类型

```python
class SkillMetadata(TypedDict):
    """技能元数据（符合 Agent Skills 规范）。"""
    
    path: str           # SKILL.md 文件路径
    name: str           # 技能标识符（1-64字符，小写+连字符）
    description: str    # 技能描述（1-1024字符）
    license: str | None # 许可证
    compatibility: str | None  # 环境要求
    metadata: dict[str, str]   # 额外元数据
    allowed_tools: list[str]   # 推荐使用的工具列表
```

## 3. 元数据验证

### 3.1 技能名称验证

```python
def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """验证技能名称是否符合 Agent Skills 规范。
    
    约束条件：
    - 1-64 字符
    - Unicode 小写字母、数字和连字符
    - 不能以连字符开头或结尾
    - 不能包含连续连字符 "--"
    - 必须与目录名匹配
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:  # 64
        return False, "name exceeds 64 characters"
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False, "name must be lowercase alphanumeric with single hyphens only"
    
    # 检查每个字符
    for c in name:
        if c == "-":
            continue
        if (c.isalpha() and c.islower()) or c.isdigit():
            continue
        return False, "name must be lowercase alphanumeric with single hyphens only"
    
    # 必须匹配目录名
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    
    return True, ""
```

### 3.2 YAML Frontmatter 解析

```python
def _parse_skill_metadata(
    content: str,
    skill_path: str,
    directory_name: str,
) -> SkillMetadata | None:
    """从 SKILL.md 内容解析元数据。
    
    1. 检查文件大小限制（10MB）
    2. 提取 --- 之间的 YAML frontmatter
    3. 解析 YAML 并验证必需字段
    4. 构建 SkillMetadata
    """
    MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024  # 10MB 安全限制
    
    if len(content) > MAX_SKILL_FILE_SIZE:
        logger.warning("Skipping %s: content too large", skill_path)
        return None
    
    # 匹配 YAML frontmatter
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    
    if not match:
        logger.warning("Skipping %s: no valid YAML frontmatter", skill_path)
        return None
    
    frontmatter_str = match.group(1)
    
    try:
        frontmatter_data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in %s: %s", skill_path, e)
        return None
    
    # 提取必需字段
    name = str(frontmatter_data.get("name", "")).strip()
    description = str(frontmatter_data.get("description", "")).strip()
    
    if not name or not description:
        logger.warning("Skipping %s: missing required fields", skill_path)
        return None
    
    # 验证名称格式（警告但继续加载）
    is_valid, error = _validate_skill_name(name, directory_name)
    if not is_valid:
        logger.warning("Skill '%s' does not follow spec: %s", name, error)
    
    return SkillMetadata(...)
```

## 4. 技能加载机制

### 4.1 列出源目录中的技能

```python
def _list_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """从单个源目录加载所有技能。
    
    流程：
    1. 列出源目录下的所有项目
    2. 筛选出包含 SKILL.md 的子目录
    3. 批量下载所有 SKILL.md 文件
    4. 解析每个文件的元数据
    """
    items = backend.ls_info(source_path)
    
    # 找出所有技能目录（包含 SKILL.md 的目录）
    skill_dirs = [item["path"] for item in items if item.get("is_dir")]
    
    # 构建 SKILL.md 路径列表
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))
    
    # 批量下载（优化性能）
    paths_to_download = [md_path for _, md_path in skill_md_paths]
    responses = backend.download_files(paths_to_download)
    
    # 解析每个 SKILL.md
    skills = []
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses):
        if response.error or response.content is None:
            continue
        
        content = response.content.decode("utf-8")
        directory_name = PurePosixPath(skill_dir_path).name
        
        skill_metadata = _parse_skill_metadata(content, skill_md_path, directory_name)
        if skill_metadata:
            skills.append(skill_metadata)
    
    return skills
```

### 4.2 多源合并策略

```python
def before_agent(self, state, runtime, config) -> SkillsStateUpdate | None:
    """加载所有源的技能，后加载的覆盖先加载的（同名技能）。"""
    
    # 跳过已加载的情况
    if "skills_metadata" in state:
        return None
    
    backend = self._get_backend(state, runtime, config)
    all_skills: dict[str, SkillMetadata] = {}
    
    # 按顺序加载所有源，后覆盖先
    for source_path in self.sources:
        source_skills = _list_skills(backend, source_path)
        for skill in source_skills:
            all_skills[skill["name"]] = skill  # 同名覆盖
    
    return SkillsStateUpdate(skills_metadata=list(all_skills.values()))
```

## 5. 渐进式披露模式

### 5.1 系统提示模板

```python
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you see their name and description above, but only read full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Read the skill's full instructions**: Use the path shown in the skill list above
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include helper scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- User's request matches a skill's domain (e.g., "research X" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills -> See "web-research" skill with its path
2. Read the skill using the path shown
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""
```

### 5.2 格式化技能列表

```python
def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
    """将技能元数据格式化为系统提示中的列表。"""
    if not skills:
        paths = [f"{source_path}" for source_path in self.sources]
        return f"(No skills available yet. You can create skills in {' or '.join(paths)})"
    
    lines = []
    for skill in skills:
        # 添加注解（许可证、兼容性）
        annotations = _format_skill_annotations(skill)
        desc_line = f"- **{skill['name']}**: {skill['description']}"
        if annotations:
            desc_line += f" ({annotations})"
        lines.append(desc_line)
        
        # 显示推荐工具
        if skill["allowed_tools"]:
            lines.append(f"  -> Allowed tools: {', '.join(skill['allowed_tools'])}")
        
        # 显示读取路径（关键：渐进式披露）
        lines.append(f"  -> Read `{skill['path']}` for full instructions")
    
    return "\n".join(lines)

def _format_skill_annotations(skill: SkillMetadata) -> str:
    """构建注解字符串。"""
    parts = []
    if skill.get("license"):
        parts.append(f"License: {skill['license']}")
    if skill.get("compatibility"):
        parts.append(f"Compatibility: {skill['compatibility']}")
    return ", ".join(parts)
```

### 5.3 格式化技能位置

```python
def _format_skills_locations(self) -> str:
    """格式化技能源位置信息。"""
    locations = []
    
    for i, source_path in enumerate(self.sources):
        # 从路径提取名称（如 "/skills/user/" -> "User"）
        name = PurePosixPath(source_path.rstrip("/")).name.capitalize()
        
        # 最后一个源标记为高优先级
        suffix = " (higher priority)" if i == len(self.sources) - 1 else ""
        locations.append(f"**{name} Skills**: `{source_path}`{suffix}")
    
    return "\n".join(locations)
```

## 6. 请求修改流程

```python
def modify_request(self, request: ModelRequest) -> ModelRequest:
    """将技能文档注入到模型请求的系统消息中。"""
    
    # 从状态获取已加载的技能元数据
    skills_metadata = request.state.get("skills_metadata", [])
    
    # 格式化位置和列表
    skills_locations = self._format_skills_locations()
    skills_list = self._format_skills_list(skills_metadata)
    
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

## 7. 状态管理

### 7.1 SkillsState 定义

```python
class SkillsState(AgentState):
    """SkillsMiddleware 的状态模式。"""
    
    skills_metadata: NotRequired[
        Annotated[list[SkillMetadata], PrivateStateAttr]
    ]
    """已加载的技能元数据列表，不传播给父代理。"""


class SkillsStateUpdate(TypedDict):
    """SkillsMiddleware 的状态更新。"""
    
    skills_metadata: list[SkillMetadata]
```

### 7.2 后端解析

```python
def _get_backend(
    self, 
    state: SkillsState, 
    runtime: Runtime, 
    config: RunnableConfig
) -> BackendProtocol:
    """解析后端实例或工厂函数。"""
    
    if callable(self._backend):
        # 构建 ToolRuntime 以解析工厂函数
        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            stream_writer=runtime.stream_writer,
            store=runtime.store,
            config=config,
            tool_call_id=None,
        )
        backend = self._backend(tool_runtime)
        if backend is None:
            raise AssertionError("SkillsMiddleware requires a valid backend")
        return backend
    
    return self._backend
```

## 8. 使用示例

```python
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware

# 使用文件系统后端
backend = FilesystemBackend(root_dir="/path/to/skills")

middleware = SkillsMiddleware(
    backend=backend,
    sources=[
        "/skills/base/",      # 基础技能（低优先级）
        "/skills/user/",      # 用户技能
        "/skills/project/",   # 项目技能（高优先级）
    ],
)
```

### 使用 StateBackend（工厂函数）

```python
from deepagents.backends.state import StateBackend

middleware = SkillsMiddleware(
    backend=lambda rt: StateBackend(rt),  # 工厂函数
    sources=["/skills/"],
)
```

## 9. 关键设计决策

### 9.1 为什么使用渐进式披露？

1. **减少上下文占用**：只向LLM展示元数据，而非完整技能文档
2. **按需加载**：代理根据任务需要决定读取哪些技能
3. **动态发现**：代理可以在对话过程中学习和使用新技能

### 9.2 为什么使用后覆盖策略？

```python
# 后加载的覆盖先加载的（同名技能）
for source_path in self.sources:
    source_skills = _list_skills(backend, source_path)
    for skill in source_skills:
        all_skills[skill["name"]] = skill  # 覆盖
```

这实现了技能分层：基础 → 用户 → 项目，允许逐级定制。

### 9.3 安全考虑

```python
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024  # 10MB 限制防止 DoS

if len(content) > MAX_SKILL_FILE_SIZE:
    logger.warning("Skipping %s: content too large", skill_path)
    return None
```

## 10. 总结

SkillsMiddleware 实现了完整的技能管理系统：

1. **规范兼容**：符合 Anthropic Agent Skills 规范
2. **渐进披露**：元数据展示 + 按需读取完整文档
3. **多源支持**：支持从多个源加载，后覆盖前
4. **状态隔离**：使用 PrivateStateAttr 避免状态污染
5. **平台无关**：纯后端API，不直接访问文件系统
