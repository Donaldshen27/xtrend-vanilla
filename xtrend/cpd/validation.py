"""Validation methods and types for GP-CPD."""
from dataclasses import dataclass
from typing import Any, List


@dataclass
class ValidationCheck:
    """Single validation check result.

    Attributes:
        name: Check name
        expected: Expected value or range
        actual: Actual value
        passed: Whether check passed
    """
    name: str
    expected: Any
    actual: Any
    passed: bool


@dataclass
class ValidationReport:
    """Statistical validation report.

    Attributes:
        checks: List of validation checks
    """
    checks: List[ValidationCheck]

    def __str__(self) -> str:
        """Format report as string."""
        lines = ["Validation Report", "=" * 50]

        passed_count = sum(1 for c in self.checks if c.passed)
        lines.append(f"Passed: {passed_count}/{len(self.checks)}\n")

        for check in self.checks:
            status = "✓" if check.passed else "✗"
            lines.append(f"{status} {check.name}")
            lines.append(f"  Expected: {check.expected}")
            lines.append(f"  Actual: {check.actual}")
            lines.append("")

        return "\n".join(lines)
