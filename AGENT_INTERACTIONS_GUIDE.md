# Agent Interactions Guide

## System Overview

Your system implements a **multi-agent orchestration framework** with:
- **Agent Base System**: Core agent classes and message passing
- **Agent Manager**: Central coordinator for routing and message delivery
- **LangGraph Workflow**: Sophisticated multi-step orchestration engine
- **4 Specialized Agents**: Task, Chat, RAG, and Analysis agents

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Request                              │
│                    (API Endpoint / Chat)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │    LangGraph Workflow          │
        │  (Orchestration Engine)        │
        │                                │
        │  - Router Node (Intent Detect) │
        │  - Task Stage Node             │
        │  - RAG Stage Node              │
        │  - Analysis Stage Node         │
        │  - Chat Final Node             │
        │  - Finalize Node               │
        └────────┬───────────────────────┘
                 │
                 │ Routes to appropriate stages
                 │
        ┌────────┴──────────┬──────────────┬──────────────┐
        │                   │              │              │
        ▼                   ▼              ▼              ▼
   ┌─────────┐         ┌────────┐    ┌─────────┐   ┌──────────┐
   │  Task   │         │  RAG   │    │Analysis │   │  Chat    │
   │ Manager │         │ Agent  │    │ Agent   │   │  Agent   │
   │ Agent   │         │        │    │         │   │          │
   └────┬────┘         └────┬───┘    └────┬────┘   └──────────┘
        │                   │             │
        │ Message Bus       │             │
        └───────────┬───────┴─────────────┘
                    │
        ┌───────────▼──────────────┐
        │   AgentManager           │
        │ (Message Routing &       │
        │  Message History)        │
        └──────────────────────────┘
```

---

## Agent Types & Roles

### 1. **Task Manager Agent** 
- **Agent ID**: `task_manager_001`
- **Role**: `TASK_MANAGER`
- **Responsibilities**:
  - Create/update/delete tasks
  - Track task status
  - Complete/reopen tasks
  - Task search and mutations
  - Generate task analytics

**Operations**:
- `list_tasks` - List all tasks
- `get_task` - Retrieve specific task
- `create_task` - Create new task
- `complete_task` - Mark task as complete
- `reopen_task` - Reopen completed task
- `delete_task` - Delete task
- `update_task` - Update task details
- `search_and_create` - Search then create
- `search_and_complete` - Search then complete
- `search_and_delete` - Search then delete

### 2. **RAG Agent**
- **Agent ID**: `rag_agent_001`
- **Role**: `RAG_AGENT`
- **Responsibilities**:
  - Semantic search on documents
  - Retrieve relevant context
  - Generate embeddings
  - Index and manage knowledge base

**Operations**:
- `search` - Semantic search with query
- `retrieve_context` - Get relevant documents
- `index_document` - Add to knowledge base

### 3. **Analysis Agent**
- **Agent ID**: `analyzer_001`
- **Role**: `ANALYZER`
- **Responsibilities**:
  - Analyze task data
  - Generate insights and reports
  - Identify patterns and trends
  - Provide recommendations
  - Summarize complex data

**Operations**:
- `analyze_data` - Analyze datasets
- `generate_report` - Create reports
- `identify_patterns` - Find trends
- `summarize` - Summarize data

### 4. **Chat Agent**
- **Agent ID**: `chat_agent_001`
- **Role**: `CHAT_AGENT`
- **Responsibilities**:
  - Process conversational messages
  - Maintain chat context
  - Generate natural language responses
  - Manage conversation history

**Operations**:
- `send_message` - Send chat message
- `get_response` - Get conversational response

---

## Message Flow Architecture

### Message Structure

```python
AgentMessage {
    sender_id: str              # Who sent it (e.g., "task_manager_001")
    recipient_id: str           # Who receives it (e.g., "rag_agent_001")
    message_type: str           # Type: "task_request", "chat_message", "search_query"
    content: Dict[str, Any]     # Payload with operation details
    timestamp: datetime         # When message was created
}
```

### Message Routing Flow

```
┌─────────────────────────────────────────┐
│  Agent A creates AgentMessage           │
└────────────────┬────────────────────────┘
                 │
                 ▼
         ┌───────────────────┐
         │  AgentManager     │
         │  send_message()   │
         └────────┬──────────┘
                  │
         ┌────────┴───────────┐
         │                    │
         ▼                    ▼
    ┌─────────────┐    ┌─────────────────┐
    │  Store in   │    │  Get Recipient  │
    │ Message     │    │ Agent by ID     │
    │ History     │    └────────┬────────┘
    └─────────────┘             │
                                ▼
                        ┌──────────────────┐
                        │  Check Agent     │
                        │  Found?          │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴──────────────┐
                    │                           │
             ┌──────▼─────┐           ┌────────▼────────┐
             │    NO      │           │      YES        │
             │  Return    │           │                 │
             │   None     │           ▼                 │
             └────────────┘    ┌──────────────────┐    │
                               │ Receive Message  │    │
                               │ (Add to queue)   │    │
                               └────────┬─────────┘    │
                                        │              │
                                        ▼              │
                               ┌──────────────────┐    │
                               │ Process Message  │    │
                               │ (Async)          │    │
                               └────────┬─────────┘    │
                                        │              │
                                        ▼              │
                               ┌──────────────────┐    │
                               │ Return Response  │    │
                               │ (AgentMessage    │    │
                               │  or None)        │    │
                               └──────────────────┘    │
                                                       │
```

---

## LangGraph Workflow Orchestration

### Intent-Based Routing (Router Node)

The **Router Node** analyzes user input and determines which agents to invoke:

```
User Input: "Find tasks about design and analyze the results"
                          │
                          ▼
            ┌─────────────────────────────┐
            │  Keyword Detection          │
            │  task_count = 2             │
            │  rag_count = 1              │
            │  analysis_count = 1         │
            └──────────┬──────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Routing Decision Logic          │
        │  has_task = true                 │
        │  has_rag = true                  │
        │  has_analysis = true             │
        │  wants_full_flow = true          │
        └──────────┬───────────────────────┘
                   │
                   ▼
         ┌─────────────────────────┐
         │  Route: task_rag_analysis│
         │  Execute all 3 agents   │
         └─────────────────────────┘
```

### Routing Modes

```python
# 1. Rule-Based Routing (High Confidence)
if has_task and has_rag and has_analysis:
    route = "task_rag_analysis"  # All three agents execute

# 2. Compound Detection
if has_task and has_rag:
    route = "task_rag"  # Task + RAG agents

# 3. LLM Classification (Ambiguous Cases)
if ambiguous_prompt and compound_prompt:
    route = llm_classifier(user_input)  # Use LLM to decide

# 4. Follow-Up Confirmation
if is_follow_up_confirmation and has_pending_action:
    route = "task_only"  # Confirm previously selected action
```

### Workflow Graph Edges

```
Entry Point
    │
    ▼
 ROUTER ──┬─────────────────────┬──────────────────┬─────────┬──────────┐
          │                     │                  │         │          │
          ▼                     ▼                  ▼         ▼          ▼
    task_rag_analysis    task_rag_analysis    task_only  rag_only  analysis_only  chat_only
          │                     │                  │         │          │          │
          ▼                     ▼                  ▼         ▼          ▼          │
    TASK_STAGE ─────► RAG_STAGE ──────► ANALYSIS_STAGE    ANALYSIS    CHAT_FINAL  │
          │               │                   │            STAGE         │         │
          └───────────────┴───────────────────┴─────────────┴────────────┘
                                              │
                                              ▼
                                        CHAT_FINAL (Synthesize)
                                              │
                                              ▼
                                        FINALIZE (Complete)
                                              │
                                              ▼
                                            END
```

### Stage Transitions

**Task Stage → Next Stage**:
```
task_stage → {
    "rag_analysis": Go to RAG stage if results need retrieval,
    "rag": Go to RAG for document retrieval,
    "analysis": Go to analysis for insights,
    "chat_final": Finalize response,
    "finalize": End workflow
}
```

**RAG Stage → Next Stage**:
```
rag_stage → {
    "analysis": Go to analysis for deeper insights,
    "chat_final": Finalize response,
    "finalize": End workflow
}
```

**Analysis Stage → Always Chat Final**:
```
analysis_stage → chat_final
```

---

## Interaction Patterns

### Pattern 1: Simple Task Creation
```
User: "Create a task for debugging the API"
                │
                ▼
        ┌──────────────────┐
        │ Router (analyze) │  has_task=true, others=false
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Task Stage       │  TaskManager executes create_task
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Chat Final       │  Craft response
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Response to User │
        └──────────────────┘
```

### Pattern 2: Task Search + Analysis
```
User: "Show me high-priority tasks and analyze trends"
                │
                ▼
        ┌──────────────────┐
        │ Router (analyze) │  has_task=true, has_analysis=true
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Task Stage       │  TaskManager lists high-priority tasks
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Analysis Stage   │  AnalysisAgent analyzes trends
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Chat Final       │  Synthesize results
        └────────┬─────────┘
```

### Pattern 3: Complex Multi-Agent Flow
```
User: "Find docs about API design and create tasks for anything critical"
                │
                ▼
        ┌──────────────────────┐
        │ Router (analyze)     │  has_task=true, has_rag=true
        └────────┬─────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ Task Stage           │  TaskManager prepares context
        └────────┬─────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ RAG Stage            │  RAGAgent retrieves docs
        └────────┬─────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ Analysis Stage       │  AnalysisAgent identifies critical items
        └────────┬─────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ Chat Final           │  Synthesize + prepare task creation
        └────────┬─────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ Response to User     │
        └──────────────────────┘
```

---

## Agent Manager Operations

### System Status Retrieval
```
GET /api/agents/system/status

Returns:
{
  "total_agents": 4,
  "agents_by_role": {
    "task_manager": 1,
    "chat_agent": 1,
    "rag_agent": 1,
    "analyzer": 1
  },
  "agents": [
    {"agent_id": "task_manager_001", "status": "idle", ...},
    {"agent_id": "rag_agent_001", "status": "idle", ...},
    {"agent_id": "analyzer_001", "status": "processing", ...},
    {"agent_id": "chat_agent_001", "status": "idle", ...}
  ],
  "message_history_size": 42
}
```

### Message History Tracking
```python
# AgentManager maintains circular buffer of messages
message_history: List[AgentMessage]  # Last 1000 messages
max_history: int = 1000

# Access via:
GET /api/agents/message-history?limit=50
```

### Agent Status Transitions

```
IDLE ──┐
  ▲   │
  │   ▼ (receive message / task)
  │ PROCESSING
  │   │
  │   ├─→ ERROR (exception occurs)
  │   │     │
  │   │     ▼
  │   │   ERROR (must reset)
  │   │
  │   └─→ IDLE (execution complete)
  │
  └─────── (normal completion)

OFFLINE (agent not initialized/connected)
```

---

## Workflow State Management

The **WorkflowState** persists context across agents:

```python
class WorkflowState:
    # Core execution state
    task_id: str                           # Unique workflow ID
    user_input: str                        # Original user query
    agent_messages: list[Dict]             # Messages between agents
    workflow_log: list[Dict]               # All agent calls
    workflow_status: str                   # "started", "routing_complete", etc.
    
    # Routing decisions
    current_agent: str                     # Which agents active
    routing_decision: Dict                 # Route analysis results
    
    # Multi-turn conversation context
    dialog_history: list[Dict]             # Chat history
    last_user_input: str                   # Prior user message
    last_assistant_response: str           # Prior response
    
    # Task-related persistence
    pending_task_creation: Optional[Dict]  # Awaiting confirmation
    pending_action: Optional[Dict]         # Mutation awaiting confirmation
    last_created_task: Optional[Dict]      # Most recent created task
    last_searched_tasks: list[Dict]        # Most recent search results
    last_selected_task: Optional[Dict]     # User-selected task from search
    
    # Stage communication
    stage_context: str                     # Context passed between stages
    stage_tool_results: str                # Accumulated tool results
    
    # Proposals and decisions
    last_proposal: Optional[Dict]          # Generated proposal
    error_message: Optional[str]           # Error tracking
    result: Optional[Dict]                 # Final result
```

---

## Confirmation Flow for Mutations

For destructive operations, agents implement a confirmation pattern:

```
User Input: "Delete task #5"
       │
       ▼
   TASK_STAGE (Preview Mutation)
       │
       ├─► _preview_mutation()
       │   - Find target tasks
       │   - Build mutation preview
       │   - Set pending_action
       │
       ▼
   CHAT_FINAL (Ask confirmation)
       │
       ▼
   Response: "Found 1 task. Should I delete it? Reply 'confirm' or 'yes'"
       │
       ▼
   User: "confirm"
       │
       ▼
   ROUTER (Detect confirmation)
       │
       ├─► _is_follow_up_confirmation()
       │   returns True
       │
       ▼
   TASK_STAGE (Execute mutation)
       │
       └─► _search_and_delete()
           - Executes the deletion
           - Returns confirmation
```

---

## Key Interaction Points

### 1. **Router → Agents**
- Router analyzes intent
- Determines agent combination
- Passes control to appropriate stage

### 2. **Agents → Message Bus**
- Agents can send AgentMessages to each other
- Manager routes and queues messages
- Maintains message history

### 3. **Stage → Stage**
- Task stage produces context
- RAG stage enhances with documents
- Analysis stage generates insights
- Chat final synthesizes all results

### 4. **Agent Manager → Agents**
- Receives task execution requests
- Routes to appropriate agent
- Tracks status and results
- Manages message queues

---

## Example: Complete Multi-Agent Execution

```
User: "Find open tasks related to authentication and summarize their status"

┌─────────────────────────────────────────────────────────────────┐
│ 1. ROUTER NODE                                                  │
│    Input: "Find open tasks related to authentication and       │
│            summarize their status"                             │
│    Detected: has_task=true, has_rag=true, has_analysis=true   │
│    Decision: route = "task_rag_analysis"                      │
│    Output: WorkflowState with routing_decision                │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 2. TASK STAGE                                                   │
│    Agent: TaskManager                                           │
│    Operation: list_tasks                                       │
│    Params: {filter: "open", search: "authentication"}         │
│    Output: [Task1, Task2, Task3, ...]                         │
│    Context stored in: state.stage_context                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 3. RAG STAGE                                                    │
│    Agent: RAGAgent                                              │
│    Operation: retrieve_context                                │
│    Input: List of tasks from Task Stage                       │
│    Purpose: Get related documentation about authentication     │
│    Output: Relevant documents + context                       │
│    Stored in: state.stage_tool_results                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 4. ANALYSIS STAGE                                               │
│    Agent: AnalysisAgent                                         │
│    Input:                                                       │
│      - Task list from stage 2                                  │
│      - Document context from stage 3                           │
│    Operation: Summarize task status + analyze patterns         │
│    Output: {summary, status_breakdown, insights}              │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 5. CHAT_FINAL NODE                                              │
│    Agent: ChatAgent                                             │
│    Input: All accumulated stage results                        │
│    Operation: Generate natural language response               │
│    Output: Formatted response for user                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 6. FINALIZE NODE                                                │
│    - Package final result                                      │
│    - Update workflow status                                    │
│    - Return to API caller                                      │
└────────────────────────────────────────────────────────────────┘
```

---

## Communication Channels

### 1. **Synchronous Message Bus** (AgentManager)
- Direct agent-to-agent messages
- `send_message()` - routes with response
- Used for inter-agent queries within a workflow

### 2. **Async Task Execution** (AgentManager)
- `execute_task()` - runs agent operation asynchronously
- Returns result dictionary
- Primary execution path from workflow stages

### 3. **Workflow State Passing** (LangGraph)
- State flows through nodes
- Each node updates shared WorkflowState
- Enables cross-agent context sharing

### 4. **Message History Log**
- AgentManager maintains history (last 1000 messages)
- Accessible via API: `GET /api/agents/message-history`
- Useful for debugging and auditing

---

## API Endpoints for Agent Interaction

```
# System Information
GET  /api/agents/system/status              → Whole system state
GET  /api/agents/agents                     → List all agents
GET  /api/agents/agents/{agent_id}          → Single agent status
GET  /api/agents/agents/role/{role}         → Agents by role

# Task Execution
POST /api/agents/task/execute               → Execute task on agent
POST /api/agents/task/send-message          → Send message between agents

# Message History
GET  /api/agents/message-history            → Message history log

# Workflow Execution
POST /api/workflow/execute                  → Run full workflow
```

---

## Summary

**Agents interact through three primary mechanisms:**

1. **Intent-Based Routing** (Router Node)
   - Analyzes user input keywords
   - Determines agent participation
   - Routes to optimal execution path

2. **Message-Driven Communication** (AgentManager)
   - Direct async message passing
   - Message queuing and history
   - Status tracking per agent

3. **Workflow Orchestration** (LangGraph)
   - Multi-step stage execution
   - State sharing between agents
   - Conditional transitions
   - Confirmation flows for mutations

**The result:** A flexible, composable system where agents collaborate to deliver sophisticated multi-step results while maintaining clean separation of concerns.
