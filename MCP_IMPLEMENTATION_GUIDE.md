# Model Context Protocol (MCP) Implementation Guide

## Overview

Your system implements **Model Context Protocol (MCP)** as a standardized interface for tools, resources, and prompts. MCP acts as a bridge between LLM agents, frontend clients, and backend services, providing:

- **Centralized Tool Registry**: Single source of truth for all available tools
- **Structured Contracts**: Strong parameter validation and type safety
- **Unified Async Execution**: Handles sync/async handlers transparently
- **LLM Integration**: Supports LLM proposals and natural language synthesis

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
│  (Frontend / LLM Agent / External Service)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP REST API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Routes Layer                            │
│  (/api/mcp/tools, /api/mcp/resources, /api/mcp/prompts)    │
│                                                              │
│  - List tools/resources/prompts                            │
│  - Call tool (sync or async)                               │
│  - Propose tool (LLM-driven)                               │
│  - Render prompts with variables                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           MCPServer (Core Orchestrator)                     │
│                                                              │
│  - Tool registry: Dict[name → ToolDefinition + Handler]    │
│  - Resource registry: Dict[name → Resource]                │
│  - Prompt registry: Dict[name → PromptTemplate]            │
│                                                              │
│  Methods:                                                   │
│  - register_tool()          - Tool registration            │
│  - register_resource()      - Resource registration        │
│  - register_prompt()        - Prompt registration          │
│  - call_tool()              - Execute tool (async-safe)    │
│  - call_proposed_tool()     - Execute LLM proposal         │
│  - render_prompt()          - Template rendering           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Backend Service Layer                             │
│                                                              │
│  - Database Operations (Tasks)                             │
│  - Vector Search (Embeddings)                              │
│  - RAG Tools (semantic search)                             │
│  - Agent Execution                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. **Tool Definition & Registration**

#### ToolParameter
```python
class ToolParameter(BaseModel):
    name: str                      # Parameter name
    type: str                      # "string", "number", "boolean", "object", "array"
    description: str               # Human-readable description
    required: bool = False         # Whether parameter is required
    enum: Optional[List[Any]]      # Allowed values (optional)
```

#### ToolDefinition
```python
class ToolDefinition(BaseModel):
    name: str                      # Unique tool name (e.g., "create_task")
    description: str               # What the tool does
    tool_type: ToolType            # FUNCTION, RESOURCE, or PROMPT
    parameters: List[ToolParameter]  # Parameter schema
    metadata: Dict[str, Any]       # Version, owner, tags, etc.
```

#### Tool Registration Example
```python
mcp_server.register_tool(
    name="create_task",
    description="Create a new task",
    parameters=[
        {
            "name": "title",
            "type": "string",
            "description": "Task title",
            "required": True
        },
        {
            "name": "description",
            "type": "string",
            "description": "Task description",
            "required": False
        }
    ],
    handler=async_create_task_handler,
    metadata={"version": "1.0", "owner": "TaskAgent"}
)
```

---

## Registered Tools

### Task Management Tools

#### 1. **create_task**
- **Description**: Create a new task
- **Parameters**:
  - `title` (string, required): Task title
  - `description` (string, optional): Task description
  - `priority` (string, optional): "low", "medium", "high"
- **Returns**: `{task_id, title, status: "created"}`

```bash
POST /api/mcp/tools/call
{
  "tool_name": "create_task",
  "arguments": {
    "title": "Fix authentication bug",
    "description": "JWT tokens expire prematurely",
    "priority": "high"
  }
}
```

#### 2. **search_tasks**
- **Description**: Search for tasks by query
- **Parameters**:
  - `query` (string, required): Search query
  - `limit` (number, optional): Maximum results (default: 10)
- **Returns**: `{results: [], count: int}`

```bash
POST /api/mcp/tools/call
{
  "tool_name": "search_tasks",
  "arguments": {
    "query": "authentication",
    "limit": 5
  }
}
```

#### 3. **list_tasks**
- **Description**: List tasks with optional filtering
- **Parameters**:
  - `limit` (number, optional): Max tasks (default: 100)
  - `offset` (number, optional): Pagination offset (default: 0)
  - `completed` (boolean, optional): Filter by completion status
- **Returns**: `{results: [], count: int}`

#### 4. **complete_task**
- **Description**: Mark a task as completed
- **Parameters**:
  - `task_id` (number, required): Task ID
- **Returns**: `{status: "completed", task_id: int}`

#### 5. **reopen_task**
- **Description**: Reopen a completed task
- **Parameters**:
  - `task_id` (number, required): Task ID
- **Returns**: `{status: "reopened", task_id: int}`

#### 6. **delete_task**
- **Description**: Delete a task
- **Parameters**:
  - `task_id` (number, required): Task ID
- **Returns**: `{status: "deleted", task_id: int}`
- **Note**: Requires confirmation via `call_proposed_tool()` unless `ALLOW_DESTRUCTIVE_PROPOSALS=true`

#### 7. **update_task**
- **Description**: Update task details
- **Parameters**:
  - `task_id` (number, required): Task ID
  - `title` (string, optional): New title
  - `description` (string, optional): New description
- **Returns**: `{status: "updated", task_id: int}`

#### 8. **vector_search**
- **Description**: Semantic search using embeddings
- **Parameters**:
  - `query` (string, required): Search query
  - `limit` (number, optional): Max results (default: 5)
  - `threshold` (number, optional): Similarity threshold (default: 0.08)
- **Returns**: `{results: [{id, content, similarity_score}], count: int}`

---

## MCPServer Class

### Core Methods

#### `register_tool()`
```python
async def register_tool(
    name: str,
    description: str,
    parameters: List[Dict[str, Any]],
    handler: Callable,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a new tool.
    
    Args:
        name: Tool identifier
        description: Human-readable description
        parameters: Parameter definitions
        handler: Async/sync callable to execute
        metadata: Additional info (version, owner, etc.)
    """
```

#### `call_tool()`
```python
async def call_tool(request: ToolCallRequest) -> ToolCallResult:
    """
    Execute a registered tool safely.
    
    - Handles both async and sync handlers
    - Validates tool exists and has handler
    - Wraps exceptions in ToolCallResult
    - Returns structured result: {tool_name, success, result, error}
    
    Args:
        request: ToolCallRequest with tool_name and arguments
        
    Returns:
        ToolCallResult with execution outcome
    """
```

**Example**:
```python
request = ToolCallRequest(
    tool_name="create_task",
    arguments={"title": "New task", "priority": "high"}
)
result = await mcp_server.call_tool(request)
# result.success = True
# result.result = {"task_id": 42, "title": "New task", ...}
```

#### `call_proposed_tool()`
```python
async def call_proposed_tool(
    proposal: Proposal,
    user_context: Optional[str] = None
) -> ToolCallResult:
    """
    Execute a tool proposal from an LLM with validation.
    
    - Validates proposal against tool schema
    - Auto-populates missing args from user_context if applicable
    - Requires explicit confirmation for destructive ops
    - Returns same structure as call_tool()
    
    Args:
        proposal: LLM-generated Proposal with tool, args, confirm flag
        user_context: Optional context to auto-fill arguments
        
    Returns:
        ToolCallResult with execution outcome
    """
```

**Example**:
```python
proposal = Proposal(
    tool="delete_task",
    args={"task_id": 5},
    confirm=True,  # Required for destructive operations
    user_context="task #5"
)
result = await mcp_server.call_proposed_tool(proposal)
```

#### `get_tool()`
```python
def get_tool(name: str) -> Optional[ToolDefinition]:
    """Get tool definition by name."""
```

#### `list_tools()`
```python
def list_tools() -> List[ToolDefinition]:
    """List all registered tools."""
```

#### `list_resources()`
```python
def list_resources() -> List[Dict[str, Any]]:
    """List all registered resources."""
```

#### `list_prompts()`
```python
def list_prompts() -> List[Dict[str, Any]]:
    """List all registered prompt templates."""
```

#### `render_prompt()`
```python
def render_prompt(
    name: str,
    variables: Dict[str, str]
) -> Optional[str]:
    """
    Render a prompt template with variables.
    
    Example:
        template = "Create task: {title}\nDesc: {description}"
        rendered = mcp_server.render_prompt(
            "task_creation",
            {"title": "Bug fix", "description": "Authentication"}
        )
        # "Create task: Bug fix\nDesc: Authentication"
    """
```

---

## Request/Response Models

### ToolCallRequest
```python
class ToolCallRequest(BaseModel):
    tool_name: str                     # Tool to execute
    arguments: Dict[str, Any] = {}     # Tool arguments
```

### ToolCallResult
```python
class ToolCallResult(BaseModel):
    tool_name: str                     # Which tool was executed
    success: bool                      # Execution success
    result: Any                        # Tool result (if success)
    error: Optional[str] = None        # Error message (if failed)
```

### Proposal (LLM-Generated)
```python
class Proposal(BaseModel):
    agent: Optional[str] = None        # Agent that generated proposal
    tool: str                          # Tool name to execute
    args: Dict[str, Any] = {}          # Arguments for tool
    intent: Optional[str] = None       # Why the tool was proposed
    confirm: bool = False              # User confirmation flag
    user_context: Optional[str] = None # Additional context
```

---

## API Endpoints

### Tool Discovery & Listing

#### `GET /api/mcp/tools`
List all available MCP tools.

**Response**:
```json
{
  "tools": [
    {
      "name": "create_task",
      "description": "Create a new task",
      "tool_type": "function",
      "parameters": [
        {
          "name": "title",
          "type": "string",
          "description": "Task title",
          "required": true
        }
      ],
      "metadata": {}
    }
  ],
  "count": 8
}
```

#### `GET /api/mcp/tools/{tool_name}`
Get a specific tool definition.

**Response**:
```json
{
  "name": "search_tasks",
  "description": "Search for tasks by query",
  "parameters": [...]
}
```

#### `GET /api/mcp/resources`
List all registered resources.

#### `GET /api/mcp/resources/{resource_name}`
Get a specific resource.

#### `GET /api/mcp/prompts`
List all prompt templates.

#### `GET /api/mcp/prompts/{prompt_name}`
Get a specific prompt template.

### Tool Execution

#### `POST /api/mcp/tools/call`
Execute a tool synchronously.

**Request**:
```json
{
  "tool_name": "create_task",
  "arguments": {
    "title": "Fix bug",
    "description": "Authentication issue"
  }
}
```

**Response** (success):
```json
{
  "tool_name": "create_task",
  "success": true,
  "result": {
    "task_id": 42,
    "title": "Fix bug",
    "status": "created"
  },
  "error": null
}
```

**Response** (failure):
```json
{
  "tool_name": "create_task",
  "success": false,
  "result": null,
  "error": "Tool validation failed: missing required parameter 'title'"
}
```

#### `POST /api/mcp/tools/call?natural=true`
Execute tool and synthesize natural language response.

**Query Parameters**:
- `natural=true` - Request natural language synthesis
- `user_message=<text>` - Guide for LLM response generation

**Response**:
```json
{
  "tool_name": "search_tasks",
  "success": true,
  "result": {...},
  "assistant_response": "I found 3 tasks related to authentication..."
}
```

#### `POST /api/mcp/tools/propose`
Execute an LLM-proposed tool (with validation & confirmation checks).

**Request**:
```json
{
  "tool": "delete_task",
  "args": {"task_id": 5},
  "confirm": true,
  "intent": "Remove completed task",
  "user_context": "task #5"
}
```

**Response** (destructive op without confirmation):
```json
{
  "tool_name": "delete_task",
  "success": false,
  "result": null,
  "error": "Destructive operation requires explicit confirmation. Resubmit proposal with confirm=true to proceed."
}
```

#### `POST /api/mcp/tools/call_natural`
Call tool and generate natural language response (alternative endpoint).

**Request**:
```json
{
  "tool_name": "search_tasks",
  "arguments": {"query": "authentication", "limit": 5},
  "user_message": "Summarize the results"
}
```

#### `POST /api/mcp/prompts/{prompt_name}/render`
Render a prompt template with variables.

**Request**:
```json
{
  "variables": {
    "title": "Fix bug",
    "description": "Auth issue",
    "priority": "high"
  }
}
```

**Response**:
```json
{
  "prompt": "task_creation",
  "rendered": "Create task: Fix bug\nDescription: Auth issue\nPriority: high"
}
```

#### `GET /api/mcp/status`
Get MCP server status.

**Response**:
```json
{
  "status": "running",
  "tools_count": 8,
  "resources_count": 2,
  "prompts_count": 2
}
```

---

## Async Handler Pattern

MCP supports both synchronous and asynchronous handlers seamlessly.

### Async Handler Example
```python
async def create_task_handler(title: str, description: str = ""):
    db = SessionLocal()
    try:
        task = TaskModel(title=title, description=description, completed=False)
        db.add(task)
        db.commit()
        db.refresh(task)
        return {"task_id": task.id, "title": title, "status": "created"}
    finally:
        db.close()

mcp_server.register_tool(
    name="create_task",
    description="Create a new task",
    parameters=[
        {"name": "title", "type": "string", "required": True},
        {"name": "description", "type": "string", "required": False}
    ],
    handler=create_task_handler
)
```

### Sync Handler Example
```python
def simple_handler(value: int) -> Dict[str, Any]:
    return {"result": value * 2}

mcp_server.register_tool(
    name="double_number",
    description="Double a number",
    parameters=[{"name": "value", "type": "number", "required": True}],
    handler=simple_handler  # Will be awaited automatically if needed
)
```

### MCPServer.call_tool() Handling
```python
async def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
    handler = self.tool_handlers.get(request.tool_name)
    
    # Check if handler is async function
    if asyncio.iscoroutinefunction(handler):
        result = await handler(**request.arguments)
    else:
        # Call sync function and check if result is coroutine
        result = handler(**request.arguments)
        if asyncio.iscoroutine(result):
            result = await result
    
    return ToolCallResult(
        tool_name=request.tool_name,
        success=True,
        result=result
    )
```

---

## Initialization & Setup

### Startup Flow (startup.py)

1. **Create MCPServer Instance**
   ```python
   mcp_server = MCPServer()
   ```

2. **Define Handlers**
   ```python
   async def create_task_handler(title: str, description: str = ""):
       # Implementation...
       pass
   ```

3. **Get Tool Definitions from Registry**
   ```python
   create_task_def = MCPToolRegistry.create_task_tool()
   # Returns: {"name": "create_task", "description": "...", "parameters": [...]}
   ```

4. **Register Tools**
   ```python
   mcp_server.register_tool(
       name=create_task_def["name"],
       description=create_task_def["description"],
       parameters=create_task_def["parameters"],
       handler=create_task_handler
   )
   ```

5. **Register Prompts** (optional)
   ```python
   mcp_server.register_prompt(
       name="task_creation",
       template="Create a task: {title}\nDescription: {description}\nPriority: {priority}",
       description="Template for creating tasks",
       variables=["title", "description", "priority"]
   )
   ```

6. **Set Global Instance**
   ```python
   set_mcp_server(mcp_server)
   state.mcp_server = mcp_server
   ```

### Global Singleton Pattern

```python
# mcp_server.py
_mcp_server: Optional[MCPServer] = None

def get_mcp_server() -> MCPServer:
    """Get the global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server

def set_mcp_server(server: MCPServer) -> None:
    """Set the global MCP server instance."""
    global _mcp_server
    _mcp_server = server
```

**Usage in Routes**:
```python
@router.get("/tools")
async def list_tools(mcp_server: MCPServer = Depends(get_mcp_server)):
    return {"tools": mcp_server.list_tools()}
```

---

## Integration with Agents & Workflows

### Agent Using MCP Tools

```python
class TaskAgent(Agent):
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        operation = task.get("operation")
        mcp_server = task.get("mcp_server")
        
        if operation == "create_task":
            request = ToolCallRequest(
                tool_name="create_task",
                arguments={"title": task["title"], "description": task.get("description")}
            )
            result = await mcp_server.call_tool(request)
            return result.dict()
```

### LangGraph Workflow Using MCP

```python
class LangGraphWorkflow:
    async def _task_stage_node(self, state: WorkflowState) -> WorkflowState:
        # Use MCP tools to execute task operations
        request = ToolCallRequest(
            tool_name="search_tasks",
            arguments={"query": state.user_input}
        )
        result = await self.mcp_server.call_tool(request)
        state.stage_context = result.result
        return state
```

### LLM-Proposed Tool Call Flow

```
1. LLM generates Proposal JSON
   {
     "tool": "create_task",
     "args": {"title": "...", "description": "..."},
     "intent": "User wants to create a task"
   }

2. Frontend sends to /api/mcp/tools/propose
   
3. MCPServer.call_proposed_tool()
   - Validates proposal schema
   - Auto-populates missing args from context
   - Checks confirmation for destructive ops
   - Executes via call_tool()
   
4. Returns ToolCallResult to frontend
```

---

## Security & Safety Features

### Confirmation for Destructive Operations

```python
# Environment-controlled safety
allow_destructive = os.getenv("ALLOW_DESTRUCTIVE_PROPOSALS", "true")

# In call_proposed_tool()
if proposal.tool in destructive_tools and not proposal.confirm:
    if not allow_destructive:
        return ToolCallResult(
            success=False,
            error="Destructive operation requires explicit confirmation"
        )
```

**Destructive Tools** (require confirmation):
- `delete_task`
- `reopen_task` (potentially)
- Any custom tool marked as destructive

### Parameter Validation

Each tool's parameters are validated against its schema:
```python
# Tool expects: title (string, required), description (string, optional)
# Request with missing required param -> Error
{
  "tool_name": "create_task",
  "arguments": {}  # Missing 'title'
}

# Response:
{
  "success": false,
  "error": "Tool validation failed: missing required parameter 'title'"
}
```

### Error Handling

All exceptions are caught and wrapped safely:
```python
try:
    result = handler(**request.arguments)
except Exception as e:
    return ToolCallResult(
        tool_name=request.tool_name,
        success=False,
        error=str(e)  # Exception details returned safely
    )
```

---

## Example Workflows

### Workflow 1: Simple Task Creation

```
Frontend          API                 MCP              Backend
   │              │                   │                  │
   ├─ POST ──────→│                   │                  │
   │ /api/mcp/    │                   │                  │
   │ tools/call   │                   │                  │
   │              │                   │                  │
   │              ├─ ToolCallRequest ─→│                  │
   │              │                   │                  │
   │              │                   ├─ Handler ──────→ │
   │              │                   │                  │
   │              │                   │ ← TaskModel       │
   │              │                   │                  │
   │              │ ← ToolCallResult ─│                  │
   │              │                   │                  │
   │ ← Response ──┤                   │                  │
   │   (success)  │                   │                  │
   │              │                   │                  │
```

**Frontend Call**:
```typescript
const result = await aiService.callMCPTool('create_task', {
  title: 'Fix authentication bug',
  description: 'JWT tokens expire too quickly',
  priority: 'high'
});
```

**Response**:
```json
{
  "success": true,
  "result": {
    "task_id": 123,
    "title": "Fix authentication bug",
    "status": "created"
  }
}
```

### Workflow 2: LLM-Proposed Tool with Confirmation

```
LLM              Frontend          API              MCP            Backend
 │                  │              │                │                │
 ├─ Generates ──────→│              │                │                │
 │ Proposal JSON    │              │                │                │
 │                  │              │                │                │
 │                  ├─ POST ──────→│                │                │
 │                  │ /mcp/tools/  │                │                │
 │                  │ propose      │                │                │
 │                  │              │                │                │
 │                  │              ├─ Validate ────→│                │
 │                  │              │                │                │
 │                  │              │ (Check confirm)│                │
 │                  │              │                │                │
 │                  │              ├─ Call tool ───→│ ─ Handler ───→│
 │                  │              │                │                │
 │                  │              │←─ Result ──────│ ← Result       │
 │                  │              │                │                │
 │                  │ ← Response ──┤                │                │
 │                  │              │                │                │
```

### Workflow 3: Natural Language Synthesis

```
Backend Tool       MCP               LLM              Frontend
    │              │                 │                  │
    ├─ Tool ──────→│                 │                  │
    │ executes     │                 │                  │
    │              │                 │                  │
    │              ├─ Format ──────→ │                  │
    │              │ result          │                  │
    │              │                 │                  │
    │              │                 ├─ Generate ─────→│
    │              │                 │ natural language│
    │              │←─────────────────│ response        │
    │              │                 │                  │
    │              └─ Return response with text ──────→│
    │                                                   │
```

**Call with Natural Language**:
```bash
POST /api/mcp/tools/call?natural=true
{
  "tool_name": "search_tasks",
  "arguments": {"query": "authentication", "limit": 5}
}

Response:
{
  "tool_name": "search_tasks",
  "success": true,
  "result": [...],
  "assistant_response": "I found 3 tasks related to authentication issues. The most urgent is the JWT token expiration problem..."
}
```

---

## Comparison: MCP vs Direct Function Calls

| Aspect | MCP | Direct Function |
|--------|-----|-----------------|
| **Discovery** | Centralized list via API | Scattered across codebase |
| **Contract** | Strong schema validation | Implicit parameters |
| **Async** | Handled transparently | Caller must know |
| **Error Handling** | Consistent ToolCallResult | Varies by function |
| **Observability** | Full logging & tracing | Manual logging needed |
| **Composition** | LLM proposals, workflows | Manual orchestration |
| **Testability** | Mock MCPServer easily | Mock individual functions |
| **Complexity** | Medium overhead | Minimal overhead |
| **Use Case** | Production, multi-agent | Prototypes, local scripts |

---

## Best Practices

### 1. Tool Design
- **One responsibility**: Each tool does one thing well
- **Clear parameters**: Use descriptive names and types
- **Required fields**: Mark truly required parameters
- **Error messages**: Return actionable error strings

### 2. Handler Implementation
- **Async preferred**: Use async handlers for I/O operations
- **Resource cleanup**: Use try/finally for DB/resource cleanup
- **Validation**: Validate inputs before processing
- **Consistent returns**: Always return same structure for success/failure

### 3. Tool Registration
- **Startup initialization**: Register all tools at server startup
- **Registry pattern**: Use MCPToolRegistry for centralized definitions
- **Metadata**: Include version and owner for tracking
- **Documentation**: Clear descriptions for LLM consumption

### 4. Frontend Usage
- **Cache tool list**: Don't call `/api/mcp/tools` on every operation
- **Progressive disclosure**: Show available tools based on context
- **Error display**: Show tool errors in UI appropriately
- **Confirmation**: Require user confirmation for destructive ops

### 5. Workflow Integration
- **Consistent naming**: Tool names match operation names
- **Context passing**: Use user_context for auto-population
- **Logging**: Log all tool calls for debugging
- **Fallbacks**: Handle tool failures gracefully

---

## Debugging & Monitoring

### Logging Tool Calls

```python
logger.info(f"Calling MCP tool: {request.tool_name} with args: {request.arguments}")
logger.error(f"Error calling MCP tool {request.tool_name}: {e}")
```

### MCP Status Endpoint

```bash
GET /api/mcp/status

{
  "status": "running",
  "tools_count": 8,
  "resources_count": 2,
  "prompts_count": 2
}
```

### Tool Introspection

```bash
GET /api/mcp/tools

# Returns full tool definitions including:
# - Parameter names, types, descriptions
# - Required vs optional fields
# - Enums and allowed values
# - Metadata (version, owner)
```

---

## Summary

**MCP provides:**
1. ✅ Centralized tool registry accessible to all components
2. ✅ Strong contract enforcement via parameter schema
3. ✅ Transparent async/sync handler support
4. ✅ Structured error handling and results
5. ✅ LLM integration for proposals and natural language
6. ✅ Full observability and auditability

**Use MCP when:**
- Tools are shared by multiple agents/workflows
- You need LLM integration (proposals, natural language)
- Strong contracts are important (production systems)
- Observability and auditing are required

**Use direct functions when:**
- Quick prototypes or local scripts
- Simple utilities without external sharing
- Low-complexity systems where overhead isn't justified
