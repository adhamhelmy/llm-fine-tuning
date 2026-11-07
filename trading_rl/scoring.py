"""Backtests of generated code in worker subprocesses: a hard timeout and parallel scoring.

    with ScoringPool(workers=4, timeout=10) as pool:
        outcomes = pool.run_many(backtester, [dict(code=..., symbol='AAPL', start=..., end=...), ...])

Why subprocesses: an in-process SIGALRM cannot interrupt a loop stuck in C code, and the
thread fallback leaves the runaway thread running. A worker that overruns `timeout + grace`
is killed and replaced. Workers also run several backtests at once (GRPO scores a whole
generation batch per reward call), and a crash/segfault/memory blow-up in generated code
only takes down the worker.

Workers are started with `python -c "...scoring._worker_main(...)"` (not multiprocessing), so this works
from scripts without a __main__ guard and from notebooks. Bars are sent from the parent's
cache once per worker, so workers never call Alpaca. This is isolation, not a security
sandbox: generated code still runs with the user's permissions.
"""

import os
import pickle
import select
import subprocess
import sys
import time

import trading_rl

_GRACE = 5           # extra seconds before the parent kills a worker whose SIGALRM didn't fire
_WORKER_CMD = 'import sys; from trading_rl.scoring import _worker_main; _worker_main(int(sys.argv[1]))'


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 1))


class _Worker:
    def __init__(self, memory_mb):
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(trading_rl.__file__)))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [pkg_root] + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]))
        self.proc = subprocess.Popen(
            [sys.executable, '-c', _WORKER_CMD, str(memory_mb or 0)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self.known = set()      # bar cache keys already sent to this worker
        self.task = None        # (index, deadline) of the running task

    def send(self, msg):
        pickle.dump(msg, self.proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        self.proc.stdin.flush()

    def recv(self):
        return pickle.load(self.proc.stdout)

    def kill(self):
        try:
            self.proc.kill()
            self.proc.wait(5)
        except Exception:
            pass

    def close(self):
        try:
            self.send(None)
            self.proc.wait(5)
        except Exception:
            self.kill()


class ScoringPool:
    """A pool of backtest worker processes with a hard per-backtest time limit."""

    def __init__(self, workers=None, timeout=10, grace=_GRACE, memory_mb=None):
        """
        workers:   number of processes (default: cpu_count - 1, at most 8).
        timeout:   per-backtest limit in seconds (SIGALRM inside the worker; the worker is
                   killed after timeout + grace if that doesn't fire).
        memory_mb: address-space limit per worker (Linux only; ignored on macOS).
        """
        self.n_workers = workers or default_workers()
        self.timeout, self.grace, self.memory_mb = timeout, grace, memory_mb
        self._workers = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        for w in self._workers:
            w.close()
        self._workers = []

    def _ensure_workers(self, n):
        while len(self._workers) < min(n, self.n_workers):
            self._workers.append(_Worker(self.memory_mb))

    def _replace(self, w):
        w.kill()
        self._workers[self._workers.index(w)] = _Worker(self.memory_mb)

    def run(self, backtester, **job):
        return self.run_many(backtester, [job])[0]

    def run_many(self, backtester, jobs, timeout=None):
        """Backtest each job = dict(code, symbol, start, end, timeframe='day', cash=10_000,
        commission=0.0, slippage=0.0) in parallel. Returns, per job, one of
        ('ok', BacktestResult), ('timeout', None) or ('exception', message)."""
        from .backtest import BacktestResult
        timeout = timeout or self.timeout
        jobs = [{'timeframe': 'day', 'cash': 10_000, 'commission': 0.0, 'slippage': 0.0, **j} for j in jobs]
        # bars come from the parent's cache (fetched here if needed, never in a worker)
        for j in jobs:
            syms = j['symbol'] if isinstance(j['symbol'], (list, tuple)) else [j['symbol']]
            j['_bars'] = {backtester.cache_key(s, j['start'], j['end'], j['timeframe']):
                          backtester.get_bars(s, j['start'], j['end'], j['timeframe']) for s in syms}

        self._ensure_workers(len(jobs))
        results = [None] * len(jobs)
        pending = list(range(len(jobs)))
        busy = {}
        while pending or busy:
            for w in self._workers:
                if not pending:
                    break
                if w in busy.values():
                    continue
                i = pending.pop(0)
                j = jobs[i]
                try:
                    for key, bars in j['_bars'].items():
                        if key not in w.known:
                            w.send(('bars', key, bars))
                            w.known.add(key)
                    w.send(('run', j['code'], j['symbol'], j['start'], j['end'], j['timeframe'],
                            j['cash'], j['commission'], j['slippage'], timeout))
                except (BrokenPipeError, OSError):
                    self._replace(w)
                    pending.insert(0, i)
                    continue
                w.task = (i, time.monotonic() + timeout + self.grace)
                busy[w.proc.stdout.fileno()] = w

            if not busy:
                continue
            next_deadline = min(w.task[1] for w in busy.values())
            ready, _, _ = select.select(list(busy), [], [], max(0.0, next_deadline - time.monotonic()))
            for fd in ready:
                w = busy.pop(fd)
                i = w.task[0]
                try:
                    status, payload = w.recv()
                    results[i] = (status, BacktestResult(*payload) if status == 'ok' else payload)
                except Exception:   # worker died (segfault, memory limit, os._exit in code...)
                    results[i] = ('exception', 'WorkerCrashed: the backtest process died')
                    self._replace(w)
            now = time.monotonic()
            for fd, w in list(busy.items()):
                if w.task[1] <= now:
                    results[w.task[0]] = ('timeout', None)
                    del busy[fd]
                    self._replace(w)
        return results


# ── worker process ─────────────────────────────────────────────────────────────

def _worker_main(memory_mb):
    proto_in = sys.stdin.buffer
    proto_out = os.fdopen(os.dup(1), 'wb')
    # generated code may print: keep it away from the protocol pipe
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = open(os.devnull, 'w')
    if memory_mb and sys.platform.startswith('linux'):
        import resource
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    from .backtest import Backtester, call_with_timeout
    from .codegen import describe_exception, extract_strategy
    backtester = Backtester(verbose=False, cache_dir=None)

    while True:
        try:
            msg = pickle.load(proto_in)
        except EOFError:
            return
        if msg is None:
            return
        if msg[0] == 'bars':
            Backtester._cache[msg[1]] = msg[2]
            continue
        _, code, symbol, start, end, timeframe, cash, commission, slippage, timeout = msg
        try:
            strategy = extract_strategy(code)
            res = call_with_timeout(backtester.run, timeout, strategy, symbol, start, end,
                                    timeframe=timeframe, cash=cash,
                                    commission=commission, slippage=slippage)
            reply = ('ok', tuple(res))
        except TimeoutError:
            reply = ('timeout', None)
        except BaseException as e:
            reply = ('exception', describe_exception(e))
        pickle.dump(reply, proto_out, protocol=pickle.HIGHEST_PROTOCOL)
        proto_out.flush()


if __name__ == '__main__':
    _worker_main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
