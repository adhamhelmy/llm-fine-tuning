"""Runs the offline smoke test (synthetic bars, no Alpaca / GPU) under pytest."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(REPO, '.pi', 'skills', 'trading-rl-smoke-test', 'scripts', 'smoke.py')


def test_smoke():
    env = dict(os.environ, MPLBACKEND='Agg', TRADING_RL_CACHE='')
    proc = subprocess.run([sys.executable, SMOKE, '--repo', REPO], capture_output=True, text=True,
                          env=env, timeout=900)
    failed = [line for line in proc.stdout.splitlines() if line.startswith('FAIL')]
    assert proc.returncode == 0, '\n'.join(failed) or (proc.stdout[-3000:] + proc.stderr[-3000:])
