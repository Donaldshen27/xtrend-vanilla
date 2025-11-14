#!/usr/bin/env python3
"""
Codex Review Hook for Claude Code

!!! DEPRECATED - DO NOT USE AS PreToolUse HOOK !!!

This approach blocks every Edit/Write operation (2-15 seconds each),
creating significant latency during coding with no way to batch reviews.

RECOMMENDED APPROACH: Use the 'codex-review' skill instead
- Location: .claude/skills/codex-review/
- Benefit: Non-blocking suggestions, context-aware triggering
- Activation: Configured in skill-rules.json
- Usage: Claude suggests review at appropriate times, you decide when to run it

This file is kept for reference and alternative workflows (e.g., git pre-commit),
but should NOT be registered as a PreToolUse hook in .claude/settings.json.

If you still want to use this script:
- Consider git pre-commit hook (batch review at commit time)
- Consider PostToolUse + Stop hook (batch review at end of conversation)
- DO NOT use as PreToolUse (too much latency)
"""

import json
import sys
import subprocess
import re


def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Hook error: Failed to parse input JSON: {e}"
            }
        }))
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # Build prompt based on tool type
    if tool_name == "Edit":
        file_path = tool_input.get("file_path", "unknown")
        old_string = tool_input.get("old_string", "")
        new_string = tool_input.get("new_string", "")

        prompt = f"""Review this code change before it's applied:

File: {file_path}

Problem/Context: Code modification requested by user

Proposed Solution:
OLD CODE:
```
{old_string}
```

NEW CODE:
```
{new_string}
```

Please assess this change and respond with one of the following:

1. If the change is correct, safe, and ready to apply, start your response with 'APPROVE:' followed by a brief confirmation.

2. If the change has issues, start your response with 'REJECT:' followed by:
   - What's wrong with the proposed change
   - Specific suggestions for how to fix it
   - What the corrected version should look like

Be critical and thorough. Only approve if you're 100% satisfied."""

    elif tool_name == "Write":
        file_path = tool_input.get("file_path", "unknown")
        content = tool_input.get("content", "")

        prompt = f"""Review this new file before it's created:

File: {file_path}

Problem/Context: New file creation requested by user

Proposed Solution:
```
{content}
```

Please assess this new file and respond with one of the following:

1. If the code is correct, safe, and ready to create, start your response with 'APPROVE:' followed by a brief confirmation.

2. If the code has issues, start your response with 'REJECT:' followed by:
   - What's wrong with the proposed code
   - Specific suggestions for how to fix it
   - What the corrected version should look like

Be critical and thorough. Only approve if you're 100% satisfied."""
    elif tool_name == "PatchPackage":
        ticket_id = tool_input.get("ticket", "unknown ticket")
        diff = tool_input.get("diff", "")

        prompt = f"""Review this unified diff before it's applied:

Ticket: {ticket_id}

Proposed change (unified diff):
```diff
{diff}
```

Please assess this entire patch set and respond with one of the following:

1. If every change is correct, safe, and ready to merge, start your response with 'APPROVE:' followed by a brief confirmation (feel free to mention any key points you checked).

2. If you spot any issue—incorrect logic, missing context, regressions, tests needed, etc.—start your response with 'REJECT:' followed by:
   - What is wrong
   - Specific suggestions for how to fix it
   - What the corrected code should look like or what additional work is required

Evaluate the diff holistically; approving partial fixes is not allowed. Only approve if you're completely satisfied with the combined changes."""

    else:
        # Unsupported tool - just allow it
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": f"Tool {tool_name} not configured for review"
            }
        }))
        sys.exit(0)

    # Execute codex exec with the prompt
    try:
        result = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True,
            text=True,
            timeout=1490  # Leave 10 seconds buffer before the 1500s hook timeout
        )
        review_output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Codex review timed out after 1490 seconds. Please try again."
            }
        }))
        sys.exit(0)
    except FileNotFoundError:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Codex command not found. Please ensure codex is installed and in PATH."
            }
        }))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Error running codex: {e}"
            }
        }))
        sys.exit(0)

    # Parse codex response for APPROVE: or REJECT:
    if re.search(r'^APPROVE:', review_output, re.MULTILINE):
        # Approved - allow the change
        approval_msg = re.sub(r'^APPROVE:\s*', '', review_output, flags=re.MULTILINE).strip()
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": f"✅ Codex approved: {approval_msg}"
            }
        }))
        sys.exit(0)
    elif re.search(r'^REJECT:', review_output, re.MULTILINE):
        # Rejected - deny with feedback
        rejection_msg = re.sub(r'^REJECT:\s*', '', review_output, flags=re.MULTILINE).strip()
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"❌ Codex rejected this change:\n\n{rejection_msg}\n\nPlease revise based on the feedback above."
            }
        }))
        sys.exit(0)
    else:
        # Unexpected format - deny for safety
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"⚠️  Codex response format unexpected. Expected APPROVE: or REJECT: prefix.\n\nCodex output:\n{review_output}"
            }
        }))
        sys.exit(0)


if __name__ == "__main__":
    main()
