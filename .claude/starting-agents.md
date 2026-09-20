# Starting agents in Claude Code

Notes on how to start subagents. Written from the tool descriptions available in a
Claude Code session, not from the official docs, so details marked *(unverified)*
should be checked. To get current details, ask Claude to "use the claude-code-guide
agent".

## 1. Ask in plain language

Claude spawns subagents through its Agent tool. Naming an agent type, or saying
"use a subagent", triggers it:

- "Use a subagent to search the codebase for where auth tokens are refreshed"
- "Use the Explore agent to map how the notebooks are organized"
- "Use the Plan agent to design the AlexNet implementation"

Claude does not spawn them on its own for ordinary tasks. Each agent starts with no
context and costs extra.

## 2. Built-in agent types

| Agent | Use for |
|---|---|
| `Explore` | Read-only search across many files |
| `Plan` | Designing an implementation plan |
| `general-purpose` | Multi-step research and tasks |
| `claude-code-guide` | Questions about Claude Code itself |
| `fork` | A copy of Claude that inherits the current conversation |

## 3. Define your own

Create a markdown file with frontmatter in either location:

- `.claude/agents/<name>.md` for this project
- `~/.claude/agents/<name>.md` for all your projects

The frontmatter sets the agent's name, description, tools and model. The body is its
system prompt.

The `/agents` command should provide an interface for creating and managing them
*(unverified)*.

Agents can also be defined programmatically through the Claude Agent SDK.

## Useful details

- Subagents run in the background. You are notified when they finish.
- Continue a running agent with `SendMessage` rather than starting a new one.
- `isolation: "worktree"` gives an agent its own git worktree, so its edits do not
  touch your working tree.
