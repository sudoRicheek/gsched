"""Tests for the single-file GPU scheduler (experiments/gsched.py).

The scheduling core is `_step` (one poll iteration) with an injectable GPU-memory
source, so we can drive the whole reap/evict/dispatch logic deterministically
with trivial shell jobs (true/false/sleep) and a fake `mem_fn` -- no real GPUs.
"""
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gsched  # noqa: E402


# ---------- fixtures ----------
@pytest.fixture
def sched(tmp_path, monkeypatch):
    """Point gsched at a throwaway DB/logdir and return its module."""
    monkeypatch.setattr(gsched, "GDIR", str(tmp_path))
    monkeypatch.setattr(gsched, "DB", str(tmp_path / "gsched.db"))
    monkeypatch.setattr(gsched, "LOGDIR", str(tmp_path / "logs"))
    return gsched


def _submit(g, cmd, name=None):
    ns = types.SimpleNamespace(cmd=cmd, name=name)
    g.cmd_submit(ns)


def _rows(c, status=None):
    q = "SELECT * FROM jobs" + (" WHERE status=?" if status else "") + " ORDER BY id"
    return c.execute(q, (status,) if status else ()).fetchall()


def _free(*gpus):
    """mem_fn: listed GPUs free (0 MiB), everything else busy (99999 MiB)."""
    return lambda: {g: (0 if g in gpus else 99999) for g in range(8)}


def _wait_exit(procs, timeout=5):
    """Block until every launched Popen has exited (for reap assertions)."""
    end = time.time() + timeout
    while time.time() < end and any(p.poll() is None for p in procs.values()):
        time.sleep(0.02)


# ---------- submit / submitf / evict-queued / rm ----------
def test_submit_creates_queued_row(sched):
    c = sched.db()
    _submit(sched, "true", name="job-a")
    (r,) = _rows(c)
    assert r["status"] == "queued" and r["cmd"] == "true" and r["name"] == "job-a"
    assert r["gpu"] is None and r["pid"] is None


def test_submitf_reads_lines_and_tab_names(sched, tmp_path):
    f = tmp_path / "jobs.txt"
    f.write_text("# a comment\n\nplain-cmd\nnamed\techo hi\n")
    sched.cmd_submitf(types.SimpleNamespace(file=str(f), name=None))
    rows = _rows(sched.db())
    assert [r["cmd"] for r in rows] == ["plain-cmd", "echo hi"]
    assert rows[1]["name"] == "named"          # name<TAB>cmd parsed
    assert rows[0]["name"] is None             # comments/blank lines skipped


def test_evict_queued_is_immediate(sched):
    c = sched.db()
    _submit(sched, "true")
    sched.cmd_evict(types.SimpleNamespace(id=1))
    assert _rows(c)[0]["status"] == "evicted"   # never ran, no daemon needed


def test_rm_done_clears_finished_only(sched):
    c = sched.db()
    c.execute("INSERT INTO jobs(cmd,status) VALUES('x','done')")
    c.execute("INSERT INTO jobs(cmd,status) VALUES('y','queued')")
    c.commit()
    sched.cmd_rm(types.SimpleNamespace(id=None, done=True, all=False))
    assert [r["status"] for r in _rows(c)] == ["queued"]


# ---------- dispatch logic ----------
def test_dispatch_fills_free_gpus_fifo(sched):
    c = sched.db()
    for i in range(3):
        _submit(sched, "sleep 30", name=f"j{i}")
    procs = {}
    n = sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(0, 1, 2, 3))
    assert n == 3                                       # 3 jobs, 4 free gpus
    run = _rows(c, "running")
    assert [r["id"] for r in run] == [1, 2, 3]          # FIFO by id
    assert sorted(r["gpu"] for r in run) == [0, 1, 2]   # distinct, lowest gpus
    for p in procs.values():
        p.terminate()


def test_dispatch_skips_busy_and_respects_allowed(sched):
    c = sched.db()
    for _ in range(4):
        _submit(sched, "sleep 30")
    procs = {}
    # allowed = {0,1,2,3}; only gpu2 reported free -> exactly one launch, on gpu2
    n = sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(2))
    assert n == 1
    (r,) = _rows(c, "running")
    assert r["gpu"] == 2
    assert len(_rows(c, "queued")) == 3
    for p in procs.values():
        p.terminate()


def test_never_uses_gpu_outside_allowed_list(sched):
    c = sched.db()
    _submit(sched, "sleep 30")
    procs = {}
    # gpu5 is free but NOT allowed (allowed = 0..3) -> nothing launches
    n = sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(5))
    assert n == 0 and len(_rows(c, "queued")) == 1


def test_no_double_launch_on_occupied_gpu(sched):
    c = sched.db()
    _submit(sched, "sleep 30")
    _submit(sched, "sleep 30")
    procs = {}
    # step 1: gpu0 free -> launches job1 on gpu0
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    # step 2: gpu0 now has a running job of ours -> job2 must stay queued
    #         (even though the fake mem still says gpu0 is free)
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    assert len(_rows(c, "running")) == 1
    assert len(_rows(c, "queued")) == 1
    for p in procs.values():
        p.terminate()


# ---------- reaping ----------
def test_reap_marks_done_and_failed_with_rc(sched):
    c = sched.db()
    _submit(sched, "true")     # exits 0
    _submit(sched, "false")    # exits 1
    procs = {}
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    assert len(_rows(c, "running")) == 2
    _wait_exit(procs)
    # next step reaps both
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    done = {r["cmd"]: r for r in _rows(c) if r["status"] in ("done", "failed")}
    assert done["true"]["status"] == "done" and done["true"]["rc"] == 0
    assert done["false"]["status"] == "failed" and done["false"]["rc"] == 1


def test_finished_gpu_is_reused_for_next_queued(sched):
    c = sched.db()
    _submit(sched, "true")      # finishes fast, frees gpu0
    _submit(sched, "sleep 30")  # should then take gpu0
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))   # launch job1 on gpu0
    _wait_exit(procs)                                   # job1 exits
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))   # reap job1, launch job2
    assert _rows(c)[0]["status"] == "done"
    r2 = _rows(c)[1]
    assert r2["status"] == "running" and r2["gpu"] == 0
    for p in procs.values():
        p.terminate()


# ---------- eviction of a running job ----------
def test_evict_running_kills_process(sched):
    c = sched.db()
    _submit(sched, "sleep 60")
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    (r,) = _rows(c, "running")
    pid = r["pid"]
    assert gsched.alive(pid)
    # request eviction, then let a step carry it out
    sched.cmd_evict(types.SimpleNamespace(id=r["id"]))
    assert _rows(c)[0]["status"] == "evicting"
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    assert _rows(c)[0]["status"] == "evicted"
    # process is actually gone, and NOT re-marked failed by a later reap
    for _ in range(50):
        if not gsched.alive(pid):
            break
        time.sleep(0.02)
    assert not gsched.alive(pid)
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    assert _rows(c)[0]["status"] == "evicted"          # stays evicted


def test_evicted_frees_gpu_for_next_job(sched):
    c = sched.db()
    _submit(sched, "sleep 60")
    _submit(sched, "sleep 60")
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))   # job1 running on gpu0
    sched.cmd_evict(types.SimpleNamespace(id=1))
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))   # evict job1, launch job2
    statuses = {r["id"]: r["status"] for r in _rows(c)}
    assert statuses[1] == "evicted" and statuses[2] == "running"
    for p in procs.values():
        p.terminate()


# ---------- live GPU-list change ----------
def test_gpus_updates_daemon_row_live(sched):
    c = sched.db()
    c.execute("INSERT INTO daemon(id,pid,gpus,poll,mem,heartbeat) VALUES(1,1,'0,1',7,2000,0)")
    c.commit()
    sched.cmd_gpus(types.SimpleNamespace(gpus="0,1,2,3,4,5"))
    assert c.execute("SELECT gpus FROM daemon WHERE id=1").fetchone()["gpus"] == "0,1,2,3,4,5"


def test_gpus_without_daemon_is_noop(sched):
    c = sched.db()
    sched.cmd_gpus(types.SimpleNamespace(gpus="0,1"))
    assert c.execute("SELECT * FROM daemon").fetchone() is None


# ---------- restart resilience ----------
def test_reaps_orphaned_running_row_after_restart(sched):
    # simulate a running row whose process is already dead and NOT in our procs
    # (as if the daemon restarted): _step should mark it done.
    c = sched.db()
    c.execute("INSERT INTO jobs(cmd,status,gpu,pid,started) VALUES('x','running',0,999999,0)")
    c.commit()
    sched._step(c, [0], {}, 2000, mem_fn=_free(0))
    assert _rows(c)[0]["status"] == "done"
