"""
IT Projects Training Data Generator and Examples

Generate Mistral fine-tuning data specifically for IT project management tasks.
Includes common development, DevOps, and infrastructure scenarios.
"""

import json
from typing import List, Dict, Any

# ============================================================================
# IT PROJECT TASK CATEGORIES
# ============================================================================

IT_COMPONENTS = [
    "frontend",
    "backend", 
    "database",
    "devops",
    "mobile",
    "api",
    "security",
    "performance",
    "testing",
    "documentation",
    "infrastructure",
]

IT_TASK_TYPES = [
    "feature",
    "bug",
    "refactor",
    "technical-debt",
    "chore",
    "documentation",
    "infrastructure",
    "security",
]

IT_PRIORITIES = ["critical", "high", "medium", "low"]

# ============================================================================
# IT PROJECT EXAMPLES
# ============================================================================

IT_PROJECT_EXAMPLES = [
    # Feature Development
    {
        "messages": [
            {
                "role": "user",
                "content": "Create a task to implement user authentication with OAuth2"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Implement OAuth2 authentication", "description": "Integrate OAuth2 with Google and GitHub providers. Setup JWT token refresh mechanism.", "priority": "high", "component": "backend", "estimate": "8 hours", "skills": ["OAuth2", "JWT", "Node.js"]}, "intent": "create"}'
            }
        ]
    },
    
    # Bug Fix
    {
        "messages": [
            {
                "role": "user",
                "content": "Log a critical bug: Database queries are timing out under load"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Fix database query timeouts", "description": "Investigation: Slow queries during peak load. Add database indexes and optimize N+1 query issues. Expected impact: 50% performance improvement.", "type": "bug", "priority": "critical", "component": "database"}, "intent": "create"}'
            }
        ]
    },
    
    # DevOps/Infrastructure
    {
        "messages": [
            {
                "role": "user",
                "content": "Setup continuous deployment pipeline with GitHub Actions"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Setup CI/CD with GitHub Actions", "description": "Configure automated testing, linting, and deployment to staging/production. Include security scanning and performance checks.", "priority": "high", "component": "devops", "estimate": "12 hours"}, "intent": "create"}'
            }
        ]
    },
    
    # Testing
    {
        "messages": [
            {
                "role": "user",
                "content": "Increase test coverage for payment processing module to 90%"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Improve payment module test coverage", "description": "Add unit tests for payment flows, error handling, and edge cases. Target: 90% coverage. Tools: Jest, Supertest.", "priority": "high", "component": "testing", "estimate": "6 hours"}, "intent": "create"}'
            }
        ]
    },
    
    # Security
    {
        "messages": [
            {
                "role": "user",
                "content": "Implement rate limiting and DDoS protection on API"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Add rate limiting and DDoS protection", "description": "Implement Redis-based rate limiting. Configure Cloudflare DDoS protection. Setup request throttling per user/IP.", "priority": "high", "component": "security", "estimate": "6 hours", "compliance": ["OWASP", "CWE-770"]}, "intent": "create"}'
            }
        ]
    },
    
    # Documentation
    {
        "messages": [
            {
                "role": "user",
                "content": "Document all REST API endpoints with OpenAPI spec"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Create OpenAPI/Swagger documentation", "description": "Document all endpoints with request/response examples. Generate interactive Swagger UI. Include error codes and authentication.", "priority": "medium", "component": "documentation", "estimate": "5 hours"}, "intent": "create"}'
            }
        ]
    },
    
    # Performance Optimization
    {
        "messages": [
            {
                "role": "user",
                "content": "Optimize frontend performance and reduce bundle size"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Frontend performance optimization", "description": "Code splitting, lazy loading, and tree-shaking. Target: 40% reduction in bundle size. Goal: <2s initial load time.", "priority": "medium", "component": "frontend", "estimate": "10 hours", "metrics": ["bundle_size", "load_time", "lighthouse_score"]}, "intent": "create"}'
            }
        ]
    },
    
    # Refactoring
    {
        "messages": [
            {
                "role": "user",
                "content": "Refactor authentication module to follow SOLID principles"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Refactor auth module for SOLID compliance", "description": "Decouple dependencies, improve testability. Replace magic strings with enums. Add proper error handling.", "priority": "medium", "type": "refactor", "component": "backend", "estimate": "8 hours"}, "intent": "create"}'
            }
        ]
    },
    
    # Technical Debt
    {
        "messages": [
            {
                "role": "user",
                "content": "Upgrade Node.js version and dependencies"
            },
            {
                "role": "assistant",
                "content": '{"tool": "create_task", "args": {"title": "Upgrade Node.js and dependencies", "description": "Update Node.js from v16 to v20. Update all npm packages. Test for breaking changes. Update CI/CD config.", "priority": "high", "type": "technical-debt", "component": "backend", "estimate": "6 hours", "risk": "medium"}, "intent": "create"}'
            }
        ]
    },
    
    # Search Tasks
    {
        "messages": [
            {
                "role": "user",
                "content": "Show me all high-priority backend tasks for this sprint"
            },
            {
                "role": "assistant",
                "content": '{"tool": "search_tasks", "args": {"component": "backend", "priority": "high", "sprint": "current", "status": "pending"}, "intent": "search"}'
            }
        ]
    },
    
    # List Tasks
    {
        "messages": [
            {
                "role": "user",
                "content": "What are all the critical bugs we need to fix?"
            },
            {
                "role": "assistant",
                "content": '{"tool": "search_tasks", "args": {"type": "bug", "priority": "critical", "status": "open"}, "intent": "search"}'
            }
        ]
    },
    
    # Update Task
    {
        "messages": [
            {
                "role": "user",
                "content": "Mark the authentication implementation as done"
            },
            {
                "role": "assistant",
                "content": '{"tool": "update_task", "args": {"query": "OAuth2 authentication", "status": "completed"}, "intent": "update"}'
            }
        ]
    },
    
    # Get Statistics
    {
        "messages": [
            {
                "role": "user",
                "content": "How many tasks do we have completed this sprint?"
            },
            {
                "role": "assistant",
                "content": '{"tool": "get_task_stats", "args": {"sprint": "current", "status": "completed"}, "intent": "stats"}'
            }
        ]
    },
]

# ============================================================================
# SPECIALIZED CATEGORIES
# ============================================================================

FRONTEND_TASKS = [
    {
        "messages": [
            {"role": "user", "content": "Create a task for responsive UI redesign"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Responsive UI redesign", "description": "Update components for mobile, tablet, and desktop. Use CSS Grid and Flexbox. Test on iOS and Android.", "priority": "high", "component": "frontend", "estimate": "12 hours", "technologies": ["React", "CSS3", "Tailwind"]}, "intent": "create"}'}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Implement dark mode support"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Add dark mode support", "description": "Implement theme switcher using CSS variables. Persist preference in localStorage. Test contrast ratios for accessibility.", "priority": "medium", "component": "frontend", "estimate": "4 hours"}, "intent": "create"}'}
        ]
    },
]

BACKEND_TASKS = [
    {
        "messages": [
            {"role": "user", "content": "Implement GraphQL API for task management"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Implement GraphQL API", "description": "Create GraphQL schema with queries, mutations, and subscriptions. Replace REST endpoints. Setup Apollo Server.", "priority": "high", "component": "backend", "estimate": "16 hours", "technologies": ["GraphQL", "Apollo", "Node.js"]}, "intent": "create"}'}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Add caching layer with Redis"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Implement Redis caching", "description": "Add Redis for session storage and query caching. Setup cache invalidation strategy. Monitor cache hit rates.", "priority": "medium", "component": "backend", "estimate": "8 hours", "tools": ["Redis", "Node-Redis"]}, "intent": "create"}'}
        ]
    },
]

DEVOPS_TASKS = [
    {
        "messages": [
            {"role": "user", "content": "Setup Kubernetes deployment for microservices"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Setup Kubernetes deployment", "description": "Containerize services. Create K8s manifests. Setup ingress, services, and persistent volumes. Configure scaling policies.", "priority": "high", "component": "devops", "estimate": "20 hours", "tools": ["Kubernetes", "Docker", "Helm"]}, "intent": "create"}'}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Setup monitoring with Prometheus and Grafana"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Setup Prometheus monitoring", "description": "Configure metrics collection. Create Grafana dashboards. Setup alerting rules. Monitor CPU, memory, and custom metrics.", "priority": "high", "component": "devops", "estimate": "12 hours", "tools": ["Prometheus", "Grafana", "AlertManager"]}, "intent": "create"}'}
        ]
    },
]

SECURITY_TASKS = [
    {
        "messages": [
            {"role": "user", "content": "Implement end-to-end encryption for sensitive data"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Add E2E encryption", "description": "Implement TLS 1.3 for transport. Add database encryption at rest. Setup key rotation. Comply with HIPAA/GDPR.", "priority": "critical", "component": "security", "estimate": "12 hours", "compliance": ["HIPAA", "GDPR", "SOC2"]}, "intent": "create"}'}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Conduct security audit and fix vulnerabilities"},
            {"role": "assistant", "content": '{"tool": "create_task", "args": {"title": "Security audit and remediation", "description": "Run SAST/DAST tools. Perform code review. Fix identified vulnerabilities. Update dependencies. Generate compliance report.", "priority": "high", "component": "security", "estimate": "10 hours", "tools": ["Snyk", "SonarQube", "OWASP ZAP"]}, "intent": "create"}'}
        ]
    },
]

# ============================================================================
# TRAINING DATA GENERATION FUNCTIONS
# ============================================================================

def create_it_task_example(
    user_message: str,
    title: str,
    description: str,
    component: str,
    priority: str = "medium",
    task_type: str = "feature",
    estimate: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Create IT project task training example."""
    args = {
        "title": title,
        "description": description,
        "component": component,
        "priority": priority,
        "type": task_type,
    }
    if estimate:
        args["estimate"] = estimate
    
    args.update(kwargs)
    
    return {
        "messages": [
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": json.dumps({
                    "tool": "create_task",
                    "args": args,
                    "intent": "create"
                })
            }
        ]
    }


def generate_sprint_tasks(
    sprint_name: str,
    tasks: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Generate tasks for a sprint."""
    examples = []
    for task in tasks:
        example = create_it_task_example(
            user_message=f"Add to {sprint_name}: {task['title']}",
            title=task['title'],
            description=task['description'],
            component=task.get('component', 'backend'),
            priority=task.get('priority', 'medium'),
            task_type=task.get('type', 'feature'),
            estimate=task.get('estimate'),
            sprint=sprint_name
        )
        examples.append(example)
    
    return examples


def generate_search_examples() -> List[Dict[str, Any]]:
    """Generate common search task queries for IT projects."""
    return [
        {
            "messages": [
                {"role": "user", "content": "Show all high-priority backend tasks"},
                {"role": "assistant", "content": '{"tool": "search_tasks", "args": {"component": "backend", "priority": "high"}, "intent": "search"}'}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "List all critical bugs"},
                {"role": "assistant", "content": '{"tool": "search_tasks", "args": {"type": "bug", "priority": "critical"}, "intent": "search"}'}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What tasks are blocked by database migration?"},
                {"role": "assistant", "content": '{"tool": "search_tasks", "args": {"blocked_by": "database migration"}, "intent": "search"}'}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Show DevOps tasks for current sprint"},
                {"role": "assistant", "content": '{"tool": "search_tasks", "args": {"component": "devops", "sprint": "current"}, "intent": "search"}'}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "List all tasks awaiting code review"},
                {"role": "assistant", "content": '{"tool": "search_tasks", "args": {"status": "awaiting_review", "component": ["frontend", "backend"]}, "intent": "search"}'}
            ]
        },
    ]


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_jsonl(examples: List[Dict[str, Any]], filename: str):
    """Export examples to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    print(f"✓ Exported {len(examples)} examples to {filename}")


if __name__ == "__main__":
    print("IT Projects Training Data Examples\n")
    
    # Example 1: Export all IT project examples
    print("1. Exporting IT project examples...")
    export_to_jsonl(IT_PROJECT_EXAMPLES, "it_projects_full.jsonl")
    
    # Example 2: Export by category
    print("\n2. Exporting by category...")
    export_to_jsonl(FRONTEND_TASKS, "frontend_tasks.jsonl")
    export_to_jsonl(BACKEND_TASKS, "backend_tasks.jsonl")
    export_to_jsonl(DEVOPS_TASKS, "devops_tasks.jsonl")
    export_to_jsonl(SECURITY_TASKS, "security_tasks.jsonl")
    
    # Example 3: Generate search examples
    print("\n3. Exporting search examples...")
    search_examples = generate_search_examples()
    export_to_jsonl(search_examples, "search_examples.jsonl")
    
    # Example 4: Create custom sprint
    print("\n4. Generating sprint-specific tasks...")
    sprint_tasks = [
        {
            "title": "Implement API rate limiting",
            "description": "Add rate limiting to prevent abuse",
            "component": "backend",
            "priority": "high",
            "estimate": "4 hours"
        },
        {
            "title": "Fix memory leak in worker process",
            "description": "Debug and fix memory leak in background job worker",
            "component": "backend",
            "priority": "critical",
            "type": "bug",
            "estimate": "6 hours"
        },
    ]
    sprint_examples = generate_sprint_tasks("Sprint-12", sprint_tasks)
    export_to_jsonl(sprint_examples, "sprint_12_tasks.jsonl")
    
    print("\n✓ All training data generated successfully!")
