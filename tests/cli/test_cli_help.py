import subprocess


def test_cli_help_runs():
    # Ensure the CLI help prints without error (requires editable install in practice)
    result = subprocess.run(["python", "-m", "alpr_jetson", "-h"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "ALPR Jetson CLI" in result.stdout

