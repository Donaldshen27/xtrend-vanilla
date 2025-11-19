"""Integration test for training script."""
import subprocess
import sys
from pathlib import Path


def test_training_script_smoke_test():
    """Test that training script runs without errors (quick smoke test)."""
    # This is a smoke test - just verify imports work and argparse is correct
    train_script = Path(__file__).parent.parent / "scripts" / "train_xtrend.py"

    # Run with --help to verify argparse setup
    result = subprocess.run(
        [sys.executable, str(train_script), "--help"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Training script failed: {result.stderr}"

    # Verify key arguments are present
    assert "--model" in result.stdout
    assert "--dropout" in result.stdout
    assert "--context-method" in result.stdout

    # Verify dropout default is documented
    assert "0.3" in result.stdout, "Dropout default should be 0.3"
