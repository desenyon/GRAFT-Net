"""Figure CLI plots recorded metrics and labels synthetic demonstrations explicitly."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_figures_accept_actual_history(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"train/task_loss": [1.4, 1.3], "val/task_loss": [1.5, 1.4]}))
    output = tmp_path / "figures"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts/generate_figures.py"),
            "--history",
            str(history),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "training_curves.png").exists()
    assert not (output / "gradient_utility.png").exists()
