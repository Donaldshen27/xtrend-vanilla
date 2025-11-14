# Codex Review Skill

Get a second opinion from OpenAI Codex on your code changes - non-blocking, context-aware quality checks.

## Overview

This skill provides intelligent suggestions to run Codex reviews at appropriate times during development, without blocking your workflow.

### Key Benefits

- **Zero latency** - Code changes happen immediately, review is optional
- **Context-aware** - Triggers for security-sensitive code (auth, JWT, crypto, SQL)
- **Session tracking** - Won't nag repeatedly in the same session
- **You control** - Decide when to run reviews, skip when inappropriate

## How It Works

1. **Claude writes code** - Edit/Write operations complete immediately
2. **Skill triggers** - Suggests review for security-sensitive or complex code
3. **You decide** - Run `codex exec` when suggested, or skip
4. **Get feedback** - Codex reviews and provides suggestions
5. **Address issues** - Fix any problems found

## Prerequisites

### Install Codex CLI

```bash
# Install codex CLI
npm install -g @anthropic-ai/codex-cli

# Configure OpenAI API key
export OPENAI_API_KEY=your-api-key-here

# Test installation
codex exec "What is 2+2?"
```

## Usage

### When Claude Suggests Review

After writing security-sensitive code, Claude will suggest running Codex review:

```
Claude: I've implemented the JWT authentication middleware.
Given this is security-sensitive code, I suggest getting a
second opinion from Codex. Would you like me to prepare a
review prompt?
```

### Running Codex Review

Use the Bash tool to execute codex:

```bash
codex exec "Review this authentication code for security issues:

[Your code here]

Focus on:
- JWT token validation
- Timing attacks
- Error handling
- Security best practices"
```

### Interpreting Results

Codex will provide:
- Security vulnerabilities found
- Logic issues or edge cases
- Performance concerns
- Best practice suggestions

## Configuration

### Trigger Conditions

The skill is configured in `.claude/skills/skill-rules.json` to trigger when:

**Content Patterns:**
- `authenticate`, `authorize`
- `jwt.verify`, `jwt.sign`
- `bcrypt`, `crypto.`, `password`
- `sql.*query`, `SELECT.*FROM`
- `exec(`, `eval(`

**File Types:**
- Python (`.py`)
- TypeScript (`.ts`, `.tsx`)
- JavaScript (`.js`, `.jsx`)

**Exclusions:**
- Test files (`*.test.*`, `test_*`, `*_test`)
- Markdown documentation
- `node_modules/`, `.git/`

### Skip Conditions

The skill won't trigger if:

1. **Session tracking** - Already used codex-review in this session
2. **File marker** - File contains `@skip-codex-review` comment
3. **Environment variable** - `SKIP_CODEX_REVIEW=1` is set

### Skip Example

```python
# @skip-codex-review
def legacy_auth_helper():
    # This is legacy code being phased out
    # No need for codex review
    pass
```

## Examples

### Example 1: Authentication Review

**Code written:**
```typescript
export const authenticateUser = async (req, res, next) => {
  const token = req.cookies.jwt;
  if (!token) return res.status(401).json({ error: 'Unauthorized' });

  const decoded = jwt.verify(token, process.env.JWT_SECRET);
  req.user = await User.findById(decoded.userId);
  next();
};
```

**Codex review command:**
```bash
codex exec "Review this JWT authentication middleware:

[code above]

Focus on: Security vulnerabilities, error handling, edge cases"
```

**Codex findings:**
- Missing try/catch for jwt.verify
- No check for null user after database query
- Should validate JWT_SECRET exists on startup

### Example 2: SQL Query Review

**Code written:**
```python
def search_users(query):
    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
    return db.execute(sql)
```

**Codex review:**
- **CRITICAL:** SQL injection vulnerability
- **Suggestion:** Use parameterized queries
- **Fixed version provided**

### Example 3: Approval

**Code written:**
```typescript
function validateEmail(email: string): boolean {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
}
```

**Codex review:**
- Clean implementation
- Appropriate regex pattern
- No security concerns
- Good type safety

## Skill vs Hook Comparison

### Codex Review Skill (This Approach)

✅ Zero blocking - code changes happen immediately
✅ Context-aware - triggers for relevant code
✅ You control when to review
✅ Session tracking prevents nagging
✅ Natural workflow integration

### PreToolUse Hook (Alternative)

❌ Blocks every Edit/Write (2-15 seconds)
❌ No batching - reviews each operation separately
❌ Workflow disruption
❌ Higher API costs (more calls)

✅ Mandatory enforcement (compliance)
✅ Can't forget to review

**Recommendation:** Use skill for normal development, consider hook only for compliance-mandated workflows.

## Best Practices

### 1. Focus Reviews

Be specific about what you want Codex to check:

```bash
# Good - Focused
codex exec "Review for SQL injection vulnerabilities in this query builder"

# Less effective - Too broad
codex exec "Review this code"
```

### 2. Provide Context

Help Codex understand your code:

```bash
codex exec "Review this caching layer:

Context: Uses Redis for distributed caching
Timeout: 5 minutes
Invalidation: Manual via /api/cache/clear

[code]

Focus on: Race conditions, cache stampede, memory leaks"
```

### 3. Batch Related Changes

Review multiple related files together:

```bash
codex exec "Review this authentication system:

File 1 (middleware.ts):
[code]

File 2 (service.ts):
[code]

File 3 (validation.ts):
[code]

Focus on: Security, integration between files"
```

### 4. Act on Feedback

- **Critical issues:** Fix immediately
- **Important suggestions:** Address before commit
- **Minor improvements:** Note for future refactoring
- **Disagree:** Document reasoning in comments

## Troubleshooting

### Skill Not Triggering

**Check:**
1. File type matches trigger patterns
2. Not excluded by `pathExclusions`
3. Content contains trigger patterns
4. Skill not already used in session

**Debug:**
```bash
# Check skill configuration
cat .claude/skills/skill-rules.json | jq '.skills."codex-review"'
```

### Codex Command Not Found

```bash
# Verify installation
which codex
codex --version

# Reinstall if needed
npm install -g @anthropic-ai/codex-cli
```

### API Key Issues

```bash
# Check environment variable
echo $OPENAI_API_KEY

# Set if missing
export OPENAI_API_KEY=your-api-key-here

# Add to shell profile for persistence
echo 'export OPENAI_API_KEY=your-key' >> ~/.bashrc
```

## Alternative Approaches

If the skill approach doesn't fit your workflow:

1. **Git pre-commit hook** - Review at commit time (batch all changes)
2. **PostToolUse + Stop hook** - Batch review at end of conversation
3. **Manual reviews** - Run codex exec whenever you want

See `.claude/hooks/CODEX_REVIEW_SETUP.md` for hook-based approaches.

## Related Documentation

- **Skill implementation:** [SKILL.md](./SKILL.md)
- **Hook alternative:** [.claude/hooks/CODEX_REVIEW_SETUP.md](../../hooks/CODEX_REVIEW_SETUP.md)
- **Skill configuration:** [skill-rules.json](../skill-rules.json)

## Support

Questions or issues?

1. Check `SKILL.md` for detailed usage instructions
2. Review `CODEX_REVIEW_SETUP.md` for hook alternatives
3. Check codex CLI documentation
4. Open an issue on the repository

---

**Ready to use Codex reviews?** The skill is already enabled. Just write code and follow Claude's suggestions to run reviews when appropriate!
