---
name: codex-review
description: AUTOMATICALLY run Codex review after writing security-sensitive code (auth, crypto, SQL) - get a second opinion from OpenAI Codex
---

# Codex Code Review - AUTO-EXECUTE MODE

**IMPORTANT:** When this skill activates, you MUST automatically run a Codex review using the Bash tool. Do not ask for permission - just execute it.

**Core principle:** Two LLMs are better than one.

## When to Auto-Execute Codex Review

**MUST run review after writing:**
- Authentication/authorization code (jwt.verify, bcrypt, crypto)
- SQL queries or database operations
- Password handling or cryptographic operations
- Input validation or sanitization
- exec(), eval(), or dynamic code execution

**Skip when:**
- Trivial changes (documentation, comments, formatting)
- Changes are < 10 lines
- User explicitly says "skip review" or has `@skip-codex-review` marker

## How to Request Codex Review

### 1. Prepare the Context

Summarize what you changed and why:

```
CHANGES:
- Added user authentication middleware
- Implemented JWT token validation
- Added role-based access control

CONTEXT:
- Using Express.js with TypeScript
- JWT stored in httpOnly cookies
- Three roles: admin, user, guest
```

### 2. Run Codex Exec

Use the Bash tool to execute codex:

```bash
codex exec "Review this code implementation:

PROBLEM: Need secure authentication for Express API

SOLUTION:
[Paste the relevant code here]

QUESTIONS:
- Are there any security vulnerabilities?
- Is the error handling robust?
- Any performance concerns?
- Suggestions for improvement?"
```

### 3. Analyze Codex Response

Codex will provide:
- Security assessment
- Logic validation
- Performance considerations
- Best practice suggestions

### 4. Act on Feedback

- **Critical issues**: Fix immediately
- **Important suggestions**: Implement before commit
- **Minor improvements**: Note for future refactoring
- **Disagreements**: Trust your judgment but document reasoning

## Example Session

```
[Just completed: JWT authentication middleware]

You: Let me get Codex's opinion on this authentication implementation.

codex exec "Review this authentication code:

CONTEXT: Express.js API with JWT cookie-based auth

CODE:
export const authenticateJWT = async (req, res, next) => {
  const token = req.cookies.jwt;
  if (!token) return res.status(401).json({ error: 'Unauthorized' });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = await User.findById(decoded.userId);
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

QUESTIONS:
- Security vulnerabilities?
- Error handling sufficient?
- Performance implications?
"

[Codex responds]:
SECURITY:
- ✅ Good: httpOnly cookies, JWT verification
- ⚠️ Issue: No JWT_SECRET validation on startup
- ⚠️ Issue: User query could fail, should handle null

PERFORMANCE:
- ⚠️ DB query on every request - consider caching decoded token

SUGGESTIONS:
- Add JWT_SECRET existence check at startup
- Handle null user case
- Consider Redis for token caching
- Add rate limiting

You: [Fix critical issues: JWT_SECRET check, null user handling]
You: [Note for later: token caching, rate limiting]
```

## Integration with Existing Workflows

**With code-reviewer subagent:**
- Use code-reviewer for architectural review
- Use codex-review for security/implementation details
- Complementary perspectives

**With TDD:**
- Write tests first
- Implement code
- Run codex-review before commit
- Fix issues, ensure tests still pass

**With git workflow:**
- Make changes
- Run codex-review
- Fix issues
- Commit with confidence

## Skip Conditions

**Automatic skip:**
- File contains `@skip-codex-review` comment
- Environment variable: `SKIP_CODEX_REVIEW=1`
- Session tracking: Won't nag repeatedly

**Manual skip:**
- Simply proceed without running codex
- Skill is suggestive, not blocking

## Tips for Effective Codex Reviews

**1. Be specific in prompts:**
```bash
# Good
codex exec "Review for SQL injection vulnerabilities in this query builder"

# Less effective
codex exec "Review this code"
```

**2. Provide context:**
- What problem are you solving?
- What constraints exist?
- What specific concerns do you have?

**3. Focus reviews:**
- Security review
- Performance review
- Logic review
- Best practices review

**4. Batch related changes:**
- Review 3-5 related files together
- Show how they interact
- Get holistic feedback

## Common Patterns

### Security Review
```bash
codex exec "Security review for authentication code:
[code]
Focus on: injection attacks, token handling, timing attacks"
```

### Performance Review
```bash
codex exec "Performance review for data processing:
[code]
Focus on: algorithmic complexity, database queries, memory usage"
```

### Logic Review
```bash
codex exec "Logic review for workflow engine:
[code]
Focus on: edge cases, state transitions, error paths"
```

## Red Flags (When to ALWAYS use Codex)

- Authentication/authorization code
- Data validation logic
- SQL query construction
- File system operations
- Cryptographic operations
- Regular expressions
- Parsing user input
- Rate limiting/throttling

## Benefits

✅ **Catch blind spots**: Different LLM, different perspective
✅ **No blocking**: Suggestion only, no workflow disruption
✅ **Fast**: 2-10 seconds for review
✅ **Cost-effective**: Single API call for entire changeset
✅ **Learning**: See alternative approaches and patterns

---

**Remember**: Codex review is advisory. Use your engineering judgment for final decisions.
