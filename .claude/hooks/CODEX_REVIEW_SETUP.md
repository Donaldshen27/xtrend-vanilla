# Codex Review Hook Setup Guide

> **💡 Efficiency Tip:** Consider using the **codex-review skill** (`.claude/skills/codex-review/`) instead of the PreToolUse hook for non-blocking, context-aware code reviews. The skill approach avoids the 2-15 second latency per Edit/Write operation. See `.claude/skills/codex-review/SKILL.md` for details.

This guide explains how to set up and use the Codex Review Hook for automatic code review in Claude Code.

## What is the Codex Review Hook?

The Codex Review Hook uses OpenAI's Codex to automatically review all code changes before they're applied. It acts as a quality gate, catching bugs, enforcing best practices, and suggesting improvements before code is written.

### Benefits

- **Catch bugs early** - Issues are caught before they're introduced
- **Enforce standards** - Consistent code quality across all changes
- **Learn best practices** - Get feedback on your code in real-time
- **Reduce technical debt** - Problems are fixed immediately, not later

## Prerequisites

### 1. Install the `codex` CLI Tool

The hook requires the `codex` CLI tool to be installed and configured.

**Installation:**

```bash
# Install codex CLI (requires Node.js)
npm install -g @anthropic-ai/codex-cli

# Or with yarn
yarn global add @anthropic-ai/codex-cli
```

**Note:** The `codex` CLI is a separate tool from Claude Code. Check the official documentation for installation instructions.

### 2. Configure OpenAI API Access

The `codex` tool needs access to OpenAI's API:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key-here

# Add to your shell profile for persistence
echo 'export OPENAI_API_KEY=your-api-key-here' >> ~/.bashrc  # or ~/.zshrc
```

### 3. Verify Installation

Test that codex is working:

```bash
codex exec "What is 2+2?"
```

You should get a response from Codex.

## Hook Installation

### 1. The Hook is Already Included

The `codex-review-hook.py` file is already in `.claude/hooks/` with this repository.

### 2. Make it Executable

```bash
chmod +x .claude/hooks/codex-review-hook.py
```

### 3. Configure in settings.json

Add the hook to your `.claude/settings.json`:

**Minimal Configuration (Edit and Write only):**

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

**Note:** The matcher includes `Edit|Write|PatchPackage` to cover all code modification tools.

## How It Works

### Review Flow

```
1. Claude wants to write code
   ↓
2. PreToolUse hook triggers
   ↓
3. Hook sends code to Codex for review
   ↓
4. Codex analyzes the change
   ↓
5a. APPROVE → Code is written
5b. REJECT → Claude sees feedback and revises
```

### Supported Tools

The hook reviews these Claude Code tools:

- **Edit** - Code modifications
- **Write** - New file creation
- **PatchPackage** - Multi-file patches

### Review Criteria

Codex evaluates:

- **Correctness** - Logic errors, bugs, edge cases
- **Safety** - Security vulnerabilities, injection risks
- **Best practices** - Idiomatic code, patterns
- **Performance** - Inefficient algorithms, memory leaks
- **Maintainability** - Code clarity, documentation needs

## Configuration Options

### Adjust Timeout

The hook has a 1490-second timeout (24+ minutes). To adjust:

Edit `.claude/hooks/codex-review-hook.py`:

```python
result = subprocess.run(
    ["codex", "exec", prompt],
    capture_output=True,
    text=True,
    timeout=600  # 10 minutes instead
)
```

### Skip Certain Files

Add skip logic at the top of the hook:

```python
def main():
    hook_input = json.load(sys.stdin)
    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    # Skip generated files
    if "/generated/" in file_path or file_path.endswith(".gen.ts"):
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": "Skipping generated file"
            }
        }))
        sys.exit(0)

    # ... rest of hook
```

### Change Review Model

The hook uses `codex exec` which uses the default model. To specify a different model:

Edit the subprocess call:

```python
result = subprocess.run(
    ["codex", "exec", "--model", "gpt-4", prompt],
    # ...
)
```

## Usage Examples

### Example 1: Bug Caught

**Claude attempts:**
```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

**Codex rejects:**
```
REJECT: Division by zero vulnerability

If `numbers` is an empty list, this will raise ZeroDivisionError.

Fixed version:
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
```

**Claude revises based on feedback.**

### Example 2: Security Issue

**Claude attempts:**
```python
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
```

**Codex rejects:**
```
REJECT: SQL injection vulnerability

This code is vulnerable to SQL injection. Use parameterized queries:

def get_user(user_id):
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()
```

### Example 3: Approval

**Claude attempts:**
```python
def is_palindrome(s):
    """Check if string is a palindrome."""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```

**Codex approves:**
```
APPROVE: Clean implementation of palindrome check

- Handles mixed case correctly
- Strips non-alphanumeric characters
- Clear and concise logic
```

## Troubleshooting

### Hook Not Running

**Symptom:** Code changes go through without review

**Solutions:**

1. **Check codex installation:**
   ```bash
   which codex
   codex --version
   ```

2. **Check hook permissions:**
   ```bash
   ls -l .claude/hooks/codex-review-hook.py
   # Should show -rwxr-xr-x (executable)
   ```

3. **Check settings.json:**
   ```bash
   cat .claude/settings.json | jq '.hooks.PreToolUse'
   ```

4. **Test hook manually:**
   ```bash
   echo '{"tool_name":"Write","tool_input":{"file_path":"test.py","content":"print(1)"}}' | \
   .claude/hooks/codex-review-hook.py
   ```

### "Codex command not found"

**Symptom:** Hook fails with "Codex command not found"

**Solutions:**

1. Install codex CLI (see Prerequisites)
2. Ensure codex is in PATH:
   ```bash
   echo $PATH
   which codex
   ```

### Timeout Errors

**Symptom:** Hook times out for large changes

**Solutions:**

1. Reduce timeout in hook (see Configuration)
2. Break large changes into smaller pieces
3. Skip review for very large generated files

### API Key Issues

**Symptom:** "OpenAI API key not found" or authentication errors

**Solutions:**

1. Set OPENAI_API_KEY environment variable
2. Check API key is valid
3. Verify API quotas/limits

### False Rejections

**Symptom:** Codex rejects valid code

**Solutions:**

1. Review the feedback - may be catching real issues
2. Add context in code comments to help Codex understand
3. If consistently wrong, disable for specific file patterns

## Performance Considerations

### Review Time

- **Small changes:** 2-5 seconds
- **Medium changes:** 5-15 seconds
- **Large changes:** 15-30 seconds

### Cost

Each review makes an API call to OpenAI:
- ~$0.01-0.05 per review (typical)
- Costs vary by change size and model used

### When to Disable

Consider disabling for:
- **Large refactors** - Too slow for big changes
- **Generated code** - Automated output doesn't need review
- **Rapid prototyping** - Speed over quality
- **Non-critical files** - Tests, documentation, configs

## Best Practices

### 1. Start with Selective Enabling

Begin with reviews only on critical files:

```json
{
  "PreToolUse": [{
    "matcher": "Edit|Write|PatchPackage",
    "hooks": [{
      "type": "command",
      "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/codex-review-hook.py"
    }]
  }]
}
```

Then add skip logic in the hook for non-critical paths.

### 2. Review the Feedback

When Codex rejects code:
- **Read carefully** - Often catches real issues
- **Ask Claude to fix** - Paste Codex feedback to Claude
- **Learn patterns** - Common issues indicate areas to improve

### 3. Keep Changes Focused

Smaller, focused changes get better reviews:
- ✅ One function at a time
- ✅ Single responsibility changes
- ❌ Massive refactors
- ❌ Multiple unrelated changes

### 4. Add Context

Help Codex understand your code:
```python
# This uses a custom cache to avoid database calls
# The cache is populated in the middleware
def get_user_from_cache(user_id):
    return cache.get(f"user_{user_id}")
```

### 5. Monitor Costs

Track your OpenAI API usage:
- Set up billing alerts
- Review usage monthly
- Adjust hook usage if costs are high

## Advanced Configuration

### Multiple Hook Configurations

Run different configurations for different file types:

```json
{
  "PreToolUse": [
    {
      "matcher": "Edit",
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/codex-review-hook.py"
      }]
    },
    {
      "matcher": "Write",
      "hooks": [{
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/codex-strict-review.py"
      }]
    }
  ]
}
```

### Integration with CI/CD

The hook can be run in CI for automated reviews:

```bash
# In your CI pipeline
cat changes.json | .claude/hooks/codex-review-hook.py
```

### Custom Review Prompts

Modify the prompts in `codex-review-hook.py` to focus on specific concerns:

```python
# Add your custom review criteria
prompt = f"""Review this code change with focus on:
1. Security vulnerabilities
2. Performance bottlenecks
3. Memory leaks
4. Thread safety

{standard_review_prompt}
"""
```

## FAQs

**Q: Does this slow down Claude Code?**
A: Yes, reviews add 2-30 seconds per change. Use selectively for critical code.

**Q: Can I use a different model?**
A: Yes, pass `--model` to the codex exec command (see Configuration).

**Q: Will this work offline?**
A: No, requires internet connection for OpenAI API calls.

**Q: Can I disable it temporarily?**
A: Yes, remove from settings.json or set `SKIP_CODEX_REVIEW=1` environment variable (add check to hook).

**Q: What if Codex is wrong?**
A: You can override by modifying the change based on feedback or adjusting skip logic.

## Resources

- [OpenAI Codex Documentation](https://platform.openai.com/docs/guides/code)
- [Claude Code Hooks Documentation](https://docs.claude.com/en/docs/claude-code/hooks)
- [Hook source code](./codex-review-hook.py)
- [General hooks configuration](./CONFIG.md)

## Support

Issues with the hook?

1. Check this guide's Troubleshooting section
2. Review [CONFIG.md](./CONFIG.md) for general hook help
3. Open an issue on GitHub

---

**Ready to enable code review?** Add the hook to your settings.json and start writing better code!
