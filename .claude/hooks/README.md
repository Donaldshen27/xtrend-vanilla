# Hooks System

Production-tested hooks for automating workflows, ensuring code quality, and providing intelligent assistance in Claude Code.

## Available Hooks

### 1. **Skill Activation (UserPromptSubmit)** ⭐ THE KEY INNOVATION

**File:** `skill-activation-prompt.ts` / `skill-activation-prompt.sh`

Automatically suggests relevant skills based on keyword and intent matching defined in `.claude/skills/skill-rules.json`:
- Keyword matching (e.g., "test" → test-driven-development)
- Intent pattern matching via regex
- Configurable through skill-rules.json

**Configuration:**
```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/skill-activation-prompt.sh"
      }]
    }]
  }
}
```

### 2. **Codex Code Review (PreToolUse)** 🆕 QUALITY GATE

**File:** `codex-review-hook.py`

Uses OpenAI Codex to review all code changes before they're applied:
- Catches bugs before they're introduced
- Enforces best practices
- Suggests improvements
- Validates logic and safety

**Supports:** Edit, Write, PatchPackage tools

**Requirements:**
- `codex` CLI tool installed and configured
- OpenAI API access

**Configuration:**
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write|PatchPackage",
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/codex-review-hook.py"
      }]
    }]
  }
}
```

**See:** [CODEX_REVIEW_SETUP.md](./CODEX_REVIEW_SETUP.md) for detailed setup

### 3. **Post-Tool Tracker (PostToolUse)**

**File:** `post-tool-use-tracker.sh`

Tracks file modifications for TypeScript validation:
- Detects monorepo structure
- Identifies affected packages
- Queues TypeScript compilation checks (`tsc`)
- Manages cache for Stop hooks

**Note:** This hook queues TypeScript checks only. Build commands are logged but not executed.

**Configuration:**
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|MultiEdit|Write",
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/post-tool-use-tracker.sh"
      }]
    }]
  }
}
```

### 4. **Build Error Diagnostic (Stop)**

**File:** `trigger-build-resolver.sh`

Checks for git changes in service directories and launches the `build-error-resolver` agent when changes are detected:
- Monitors hardcoded service directories for changes
- Triggers diagnostic agent for affected services
- Provides debug logging

**Note:** This is a diagnostic/agent trigger hook, not an automated build runner.

### 5. **Error Handling Reminder (Stop)**

**File:** `error-handling-reminder.ts` / `error-handling-reminder.sh`

Reminds about error handling best practices:
- **Backend:** try/catch usage, input validation
- **Frontend:** error boundaries, user-friendly error messages
- **Database:** schema verification, migration testing

**Disable with:** `export SKIP_ERROR_REMINDER=1`

### 6. **TypeScript Validation (Stop)**

**File:** `stop-build-check-enhanced.sh`

Runs TypeScript compilation checks using cached data from PostToolUse hooks:
- Executes accumulated `tsc` checks
- Aggregates TypeScript errors from cache
- Recommends auto-error-resolver agent for 5+ errors
- Cleans up cache on success

**Note:** Only TypeScript checks are run. Build commands are not executed.

### 7. **Session Start (SessionStart)**

**File:** `session-start.sh`

Initializes session state when Claude Code starts.

## Hook Events Reference

| Event | When It Fires | Common Uses |
|-------|--------------|-------------|
| **SessionStart** | Claude Code starts | Initialize state, load project context |
| **UserPromptSubmit** | User sends a message | Skill activation via keyword matching |
| **PreToolUse** | Before tool executes | Code review, permission checks, validation |
| **PostToolUse** | After tool completes | Track changes, queue TypeScript checks |
| **Stop** | Conversation pauses/ends | Run TypeScript checks, show summaries |

## Quick Start

### 1. Install Dependencies

```bash
cd .claude/hooks
npm install
```

### 2. Set Permissions

```bash
chmod +x .claude/hooks/*.sh .claude/hooks/*.py
```

### 3. Configure Hooks

Edit `.claude/settings.json` with your desired hooks (see examples below).

### 4. Optional: Install Codex (for code review)

See [CODEX_REVIEW_SETUP.md](./CODEX_REVIEW_SETUP.md) for installation and configuration.

## Configuration Examples

### Minimal (Skill Activation Only)

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/skill-activation-prompt.sh"
      }]
    }]
  }
}
```

### Recommended (Auto-activation + TypeScript Checks)

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/skill-activation-prompt.sh"
      }]
    }],
    "PostToolUse": [{
      "matcher": "Edit|MultiEdit|Write",
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/post-tool-use-tracker.sh"
      }]
    }],
    "Stop": [{
      "hooks": [
        {
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/stop-build-check-enhanced.sh"
        },
        {
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/error-handling-reminder.sh"
        }
      ]
    }]
  }
}
```

### Maximum (All Quality Gates)

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/skill-activation-prompt.sh"
      }]
    }],
    "PreToolUse": [{
      "matcher": "Edit|Write|PatchPackage",
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/codex-review-hook.py"
      }]
    }],
    "PostToolUse": [{
      "matcher": "Edit|MultiEdit|Write",
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/post-tool-use-tracker.sh"
      }]
    }],
    "Stop": [{
      "hooks": [
        {
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/stop-build-check-enhanced.sh"
        },
        {
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/error-handling-reminder.sh"
        },
        {
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/trigger-build-resolver.sh"
        }
      ]
    }]
  }
}
```

## Hook Workflow

Here's how hooks work together in a typical development session:

```
1. SessionStart
   └─> Initialize session state

2. User types: "Add authentication to the login endpoint"
   └─> UserPromptSubmit
       └─> Keyword matching suggests: backend-dev-guidelines, error-tracking

3. Claude writes code
   └─> PreToolUse (if enabled)
       └─> Codex reviews the change
       └─> APPROVE or REJECT with feedback

   └─> PostToolUse
       └─> Track modified file
       └─> Queue TypeScript check for affected package

4. Conversation pauses
   └─> Stop
       ├─> Run cached TypeScript checks (stop-build-check-enhanced.sh)
       ├─> Check for service changes (trigger-build-resolver.sh)
       └─> Show error handling reminders

5. Session ends (cleanup happens automatically)
```

## Customization

### Skip Conditions

Add skip logic to any hook:

```bash
# Skip generated files
if [[ "$file_path" =~ /generated/ ]] || [[ "$file_path" =~ \.gen\. ]]; then
    exit 0
fi

# Skip test files for certain hooks
if [[ "$file_path" =~ \.test\. ]] || [[ "$file_path" =~ /__tests__/ ]]; then
    exit 0
fi
```

### Custom Matchers

Hooks can be targeted to specific tools:

```json
{
  "PostToolUse": [{
    "matcher": "Edit",  // Only Edit tool
    "hooks": [...]
  }, {
    "matcher": "Write|MultiEdit",  // Write OR MultiEdit
    "hooks": [...]
  }]
}
```

### Environment Variables

Control hook behavior:

```bash
# Disable error reminders
export SKIP_ERROR_REMINDER=1

# Custom project directory
export CLAUDE_PROJECT_DIR=/path/to/project
```

## Troubleshooting

### Hooks Not Running

1. **Check permissions:**
   ```bash
   chmod +x .claude/hooks/*.sh .claude/hooks/*.py
   ```

2. **Check configuration:**
   Verify hooks are registered in `.claude/settings.json`

3. **Check TypeScript compilation:**
   ```bash
   cd .claude/hooks
   npx tsc
   ```

### Performance Issues

1. **Add skip conditions** for large/generated files
2. **Use selective hooks** - don't enable all at once
3. **Optimize matchers** - target specific tools

### Debugging

Enable debug output in any shell hook:

```bash
# At the top of the hook file
set -x  # Prints each command before executing

# Or add specific debug lines
echo "DEBUG: Processing $file_path" >&2
```

## Best Practices

1. **Start minimal** - Enable UserPromptSubmit first
2. **Add incrementally** - One hook type at a time
3. **Test thoroughly** - Make sure hooks work for your project
4. **Document customizations** - Comment your changes
5. **Version control** - Commit .claude/ to git
6. **Share with team** - Keep configurations consistent

## Files Reference

| File | Type | Event | Purpose |
|------|------|-------|---------|
| skill-activation-prompt.ts/sh | TypeScript/Shell | UserPromptSubmit | Keyword-based skill suggestions |
| codex-review-hook.py | Python | PreToolUse | AI code review |
| post-tool-use-tracker.sh | Shell | PostToolUse | Track files, queue TypeScript checks |
| trigger-build-resolver.sh | Shell | Stop | Check services, trigger agent |
| error-handling-reminder.ts/sh | TypeScript/Shell | Stop | Error handling reminders |
| stop-build-check-enhanced.sh | Shell | Stop | Run TypeScript validations |
| session-start.sh | Shell | SessionStart | Initialize session |

## Further Reading

- [CONFIG.md](./CONFIG.md) - Detailed configuration guide
- [CODEX_REVIEW_SETUP.md](./CODEX_REVIEW_SETUP.md) - Codex integration guide
- [Claude Code Hooks Documentation](https://docs.claude.com/en/docs/claude-code/hooks)

## Contributing

Have a useful hook? Contributions welcome!

1. Add your hook to `.claude/hooks/`
2. Document it in this README
3. Add configuration example
4. Test thoroughly
5. Submit PR

---

**Questions?** Check CONFIG.md for advanced customization options.
