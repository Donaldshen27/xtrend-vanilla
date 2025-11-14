# Claude Code Autopilot 🚀

> **The Ultimate Claude Code Toolkit**: Auto-activating skills + Production-tested infrastructure + Specialized agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is This?

**Claude Code Autopilot** is a comprehensive integration of two powerful Claude Code ecosystems:

- **[obra/superpowers](https://github.com/obra/superpowers)** - 20+ proven skills and workflows
- **[diet103/claude-code-infrastructure-showcase](https://github.com/diet103/claude-code-infrastructure-showcase)** - Auto-activation infrastructure and specialized agents

This integration gives you the **best of both worlds**: ready-to-use skills that **automatically activate** when needed, powered by production-tested infrastructure from enterprise-scale projects.

## 🎯 Key Features

### Auto-Activation (The Game Changer)
- **UserPromptSubmit hooks** that analyze your prompts and automatically suggest relevant skills
- No more remembering to invoke skills manually
- Context-aware suggestions based on what you're trying to do

### 26 Production Skills
Combined from both repositories:
- **Testing**: TDD, async testing, anti-patterns
- **Debugging**: Systematic debugging, root cause analysis
- **Development**: Git worktrees, branch management, execution plans
- **Backend**: Express/TypeScript best practices
- **Frontend**: React/MUI v7 patterns
- **Collaboration**: Brainstorming, planning, code review
- **Error Tracking**: Production error handling

### 11 Specialized Agents
Ready-to-use agents for complex tasks:
- **Auth Route Debugger** - Debug authentication routes
- **Auth Route Tester** - Test authentication flows
- **Auto Error Resolver** - Automatically fix errors
- **Code Architecture Reviewer** - Review codebase architecture
- **Code Refactor Master** - Expert refactoring assistance
- **Code Reviewer** - Comprehensive code review
- **Documentation Architect** - Generate comprehensive docs
- **Frontend Error Fixer** - Fix frontend issues
- **Plan Reviewer** - Review implementation plans
- **Refactor Planner** - Plan refactoring efforts
- **Web Research Specialist** - Research web technologies

### Production-Tested Hooks
- **Skill Activation**: Auto-suggest relevant skills based on context
- **Error Handling**: Automatic error detection and guidance
- **Build Checks**: TypeScript compilation validation
- **Tool Tracking**: Monitor and optimize tool usage
- And more automation hooks

### Dev Docs Pattern
Three-file knowledge persistence system:
- `overview.md` - High-level project understanding
- `architecture.md` - System design and patterns
- `runbook.md` - Common operations and procedures

## 📦 Installation

### Quick Start (One Command!)

**Install to any project with a single command:**

```bash
curl -sL https://raw.githubusercontent.com/donaldshen27/claude-code-autopilot/main/install.sh | bash -s -- -y /path/to/your/project
```

> **Note:** The `-y` flag skips confirmation prompts. Omit it if you want to be prompted before installation.

That's it! The installer will:
- ✓ Download the latest template from GitHub
- ✓ Backup any existing files (`.backup.timestamp`)
- ✓ Copy all skills, agents, hooks, and commands
- ✓ Install hook dependencies automatically
- ✓ Set correct permissions
- ✓ Verify installation

Then just:
```bash
cd /path/to/your/project
claude
```

The hooks will automatically start suggesting relevant skills!

### Alternative: Manual Installation

If you prefer to install manually or want to inspect first:

1. **Clone this repository:**
```bash
git clone https://github.com/donaldshen27/claude-code-autopilot.git
cd claude-code-autopilot
```

2. **Run the installer locally:**
```bash
./install.sh /path/to/your/project
```

3. **Or copy manually:**
```bash
cp -r .claude /path/to/your/project/
cp -r dev /path/to/your/project/
cd /path/to/your/project/.claude/hooks
npm install
```

## 🚀 Usage

### Auto-Activated Skills

Just start working naturally! The hooks will detect what you're doing and suggest skills:

```bash
# You type: "I need to debug this authentication issue"
# Autopilot suggests: systematic-debugging, error-tracking skills

# You type: "Let's refactor this component"
# Autopilot suggests: code-refactor-master agent, refactoring skills

# You type: "Write tests for the API"
# Autopilot suggests: test-driven-development, route-tester
```

### Manual Skill Invocation

You can still manually invoke skills when needed:

```bash
# In your Claude Code conversation:
/skill systematic-debugging
/skill test-driven-development
```

### Using Agents

Invoke specialized agents for complex tasks:

```bash
@code-architecture-reviewer  # Review your codebase architecture
@auto-error-resolver         # Automatically fix errors
@documentation-architect     # Generate comprehensive docs
```

### Slash Commands

Built-in commands for planning and execution:

- `/brainstorm` - Generate ideas and approaches
- `/write-plan` - Create detailed implementation plans
- `/execute-plan` - Execute planned tasks
- `/dev-docs` - View/update development documentation
- `/dev-docs-update` - Update dev docs with new learnings

## 📂 Repository Structure

```
.claude/
├── agents/               # 11 specialized agents
│   ├── code-architecture-reviewer.md
│   ├── auto-error-resolver.md
│   ├── documentation-architect.md
│   └── ...
│
├── hooks/               # Auto-activation and automation
│   ├── skill-activation-prompt.ts    # THE KEY INNOVATION
│   ├── error-handling-reminder.ts
│   ├── post-tool-use-tracker.sh
│   └── ...
│
├── skills/              # 26 production skills
│   ├── systematic-debugging/
│   ├── test-driven-development/
│   ├── backend-dev-guidelines/
│   ├── frontend-dev-guidelines/
│   └── ...
│
├── commands/            # Slash commands
│   ├── brainstorm.md
│   ├── write-plan.md
│   ├── execute-plan.md
│   └── dev-docs.md
│
└── settings.json        # Configuration

dev/                     # Dev docs pattern
├── README.md           # Overview of dev docs system
└── (your project-specific docs)
```

## 🎓 Learning Path

### For Beginners
1. Start with auto-activation - just use Claude Code naturally
2. Explore suggested skills when they appear
3. Read `.claude/skills/using-superpowers/SKILL.md`
4. Try manual skill invocation

### For Advanced Users
1. Customize `.claude/settings.json`
2. Modify hooks in `.claude/hooks/` for your workflow
3. Create custom agents based on templates
4. Extend skills with your own patterns

### For Teams
1. Set up dev docs pattern in `dev/` directory
2. Configure shared skill activation rules
3. Customize agents for your tech stack
4. Add project-specific skills

## 🔧 Configuration

### Settings File (`.claude/settings.json`)

Key configuration options:

```json
{
  "hooks": {
    "UserPromptSubmit": ".claude/hooks/skill-activation-prompt.ts"
  },
  "permissions": {
    // Configure tool permissions
  }
}
```

### Hook Customization

Edit `.claude/hooks/skill-activation-prompt.ts` to customize when skills are suggested:

- Add keywords that trigger specific skills
- Adjust confidence thresholds
- Customize suggestion messages

### Skill Rules

Edit `.claude/skills/skill-rules.json` to define skill activation patterns.

## 🏗️ Architecture Highlights

### Modular Design (500-line Rule)
Skills are broken into digestible chunks to stay within context limits with progressive disclosure.

### Auto-Activation Intelligence
Hooks analyze your prompt for:
- Keywords indicating task type
- Code patterns in your project
- Error messages and stack traces
- Development phase (planning, coding, debugging)

### Context Management
- Progressive skill loading
- Smart context preservation across sessions
- Dev docs for knowledge persistence

## 📊 Stats

- **26 Skills** from both superpowers and showcase
- **11 Specialized Agents** for complex tasks
- **Production Hooks** for automation
- **6 Slash Commands** for workflows
- **Dev Docs Pattern** for knowledge management

## 🤝 Contributing

This is an integration project combining:
- [obra/superpowers](https://github.com/obra/superpowers) (MIT License)
- [diet103/claude-code-infrastructure-showcase](https://github.com/diet103/claude-code-infrastructure-showcase) (MIT License)

To contribute:
1. Fork this repository
2. Create a feature branch
3. Add your skills/agents/hooks
4. Test thoroughly
5. Submit a pull request

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

This project integrates two MIT-licensed projects:
- obra/superpowers © Jesse Vincent
- diet103/claude-code-infrastructure-showcase © Claude Code Infrastructure Contributors

## 🙏 Acknowledgments

Huge thanks to:
- **[obra](https://github.com/obra)** for creating the comprehensive superpowers skill library
- **[diet103](https://github.com/diet103)** for developing the auto-activation infrastructure and sharing 6 months of production learnings
- The entire Claude Code community for continuous innovation

## 🔗 Links

- [Claude Code Documentation](https://docs.claude.com/en/docs/claude-code)
- [obra/superpowers](https://github.com/obra/superpowers)
- [diet103/claude-code-infrastructure-showcase](https://github.com/diet103/claude-code-infrastructure-showcase)
- [Anthropic Blog: Claude Code Plugins](https://www.anthropic.com/news/claude-code-plugins)

## 🚀 What's Next?

- [ ] Add more skills from community contributions
- [ ] Create video tutorials
- [ ] Build skill marketplace integration
- [ ] Add more specialized agents
- [ ] Improve auto-activation intelligence

---

**Ready to supercharge your Claude Code experience?** Get started in 15 minutes! 🚀

For questions, issues, or suggestions, please open an issue on GitHub.
