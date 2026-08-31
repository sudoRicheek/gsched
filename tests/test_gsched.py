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


def _submit(g, cmd, name=None, notify=None, no_tty=True, ngpu=1):
    ns = types.SimpleNamespace(cmd=cmd, name=name, notify=notify, no_tty=no_tty, ngpu=ngpu)
    g.cmd_submit(ns)


def _gpus_of(row):
    """Physical GPUs a row was given, as a sorted list."""
    return sorted(gsched.gpulist(row["gpus"]))


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
    assert r["gpus"] is None and r["pid"] is None and r["ngpu"] == 1


def test_submitf_reads_lines_and_tab_names(sched, tmp_path):
    f = tmp_path / "jobs.txt"
    f.write_text("# a comment\n\nplain-cmd\nnamed\techo hi\n")
    sched.cmd_submitf(types.SimpleNamespace(file=str(f), name=None, notify=None,
                                            no_tty=True, ngpu=1))
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
    assert sorted(g for r in run for g in _gpus_of(r)) == [0, 1, 2]   # distinct, lowest
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
    assert _gpus_of(r) == [2]
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
    assert r2["status"] == "running" and _gpus_of(r2) == [0]
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


# ---------- multi-GPU jobs ----------
def test_multi_gpu_job_gets_all_its_gpus_at_once(sched):
    c = sched.db()
    _submit(sched, "sleep 30", name="ddp", ngpu=4)
    procs = {}
    n = sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(0, 1, 2, 3))
    assert n == 1
    (r,) = _rows(c, "running")
    assert _gpus_of(r) == [0, 1, 2, 3] and r["ngpu"] == 4
    for p in procs.values():
        p.terminate()


def test_job_sees_exactly_its_gpus_in_cuda_visible_devices(sched, tmp_path):
    c = sched.db()
    out = tmp_path / "cvd.txt"
    _submit(sched, f"echo $CUDA_VISIBLE_DEVICES > {out}", ngpu=2)
    procs = {}
    # gpus 1 and 3 are the free ones -> the job must be pinned to those two
    sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(1, 3))
    _wait_exit(procs)
    assert out.read_text().strip() == "1,3"


def test_multi_gpu_job_waits_until_enough_are_free(sched):
    c = sched.db()
    _submit(sched, "sleep 30", ngpu=3)
    procs = {}
    assert sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(0, 1)) == 0   # only 2 free
    assert len(_rows(c, "queued")) == 1
    assert sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(0, 1, 2)) == 1
    assert _gpus_of(_rows(c, "running")[0]) == [0, 1, 2]
    for p in procs.values():
        p.terminate()


def test_big_job_holds_the_queue_by_default(sched):
    """Strict FIFO: a 4-GPU job at the head is not overtaken, so it can't starve."""
    c = sched.db()
    _submit(sched, "sleep 30", name="big", ngpu=4)
    _submit(sched, "sleep 30", name="small")
    procs = {}
    n = sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(0, 1))   # 2 of 4 free
    assert n == 0 and len(_rows(c, "queued")) == 2


def test_backfill_lets_small_jobs_past_a_blocked_big_one(sched):
    c = sched.db()
    _submit(sched, "sleep 30", name="big", ngpu=4)
    _submit(sched, "sleep 30", name="small")
    procs = {}
    n = sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(0, 1), backfill=True)
    assert n == 1
    (r,) = _rows(c, "running")
    assert r["name"] == "small" and len(_gpus_of(r)) == 1
    assert [x["name"] for x in _rows(c, "queued")] == ["big"]
    for p in procs.values():
        p.terminate()


def test_impossible_request_is_skipped_not_queue_blocking(sched):
    """8 GPUs on a 2-GPU daemon can never run - it must not stall everyone else."""
    c = sched.db()
    _submit(sched, "sleep 30", name="impossible", ngpu=8)
    _submit(sched, "sleep 30", name="fine")
    procs = {}
    n = sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    assert n == 1
    assert _rows(c, "running")[0]["name"] == "fine"
    assert [r["name"] for r in _rows(c, "queued")] == ["impossible"]
    for p in procs.values():
        p.terminate()


def test_gpus_of_a_multi_gpu_job_are_all_released_on_finish(sched):
    c = sched.db()
    _submit(sched, "true", ngpu=2)          # takes 0,1 then exits
    _submit(sched, "sleep 30", ngpu=2)      # must be able to take them back
    procs = {}
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    _wait_exit(procs)
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    assert _rows(c)[0]["status"] == "done"
    assert _gpus_of(_rows(c)[1]) == [0, 1]
    for p in procs.values():
        p.terminate()


def test_running_multi_gpu_job_blocks_every_gpu_it_holds(sched):
    c = sched.db()
    _submit(sched, "sleep 30", ngpu=2)
    _submit(sched, "sleep 30")
    procs = {}
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))   # job1 takes both
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))   # nothing left for job2
    assert len(_rows(c, "running")) == 1 and len(_rows(c, "queued")) == 1
    for p in procs.values():
        p.terminate()


def test_mixed_sizes_pack_onto_the_free_set(sched):
    c = sched.db()
    _submit(sched, "sleep 30", name="a", ngpu=2)
    _submit(sched, "sleep 30", name="b")
    _submit(sched, "sleep 30", name="c")
    procs = {}
    n = sched._step(c, [0, 1, 2, 3], procs, 2000, mem_fn=_free(0, 1, 2, 3))
    assert n == 3
    got = {r["name"]: _gpus_of(r) for r in _rows(c, "running")}
    assert got == {"a": [0, 1], "b": [2], "c": [3]}
    for p in procs.values():
        p.terminate()


def test_submitf_accepts_a_per_line_gpu_count(sched, tmp_path):
    f = tmp_path / "jobs.txt"
    f.write_text("solo\techo one\nddp\t4\ttorchrun train.py\nbare-cmd\n")
    sched.cmd_submitf(types.SimpleNamespace(file=str(f), name=None, notify=None,
                                            no_tty=True, ngpu=2))
    rows = _rows(sched.db())
    assert [(r["name"], r["ngpu"], r["cmd"]) for r in rows] == [
        ("solo", 2, "echo one"),               # no count on the line -> file default
        ("ddp", 4, "torchrun train.py"),       # name<TAB>ngpu<TAB>cmd
        (None, 2, "bare-cmd")]


def test_evicting_multi_gpu_job_frees_all_of_its_gpus(sched):
    c = sched.db()
    _submit(sched, "sleep 60", ngpu=2)
    _submit(sched, "sleep 60", ngpu=2)
    procs = {}
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    sched.cmd_evict(types.SimpleNamespace(id=1))
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    statuses = {r["id"]: r["status"] for r in _rows(c)}
    assert statuses[1] == "evicted" and statuses[2] == "running"
    assert _gpus_of(_rows(c)[1]) == [0, 1]
    for p in procs.values():
        p.terminate()


def test_multi_gpu_notification_names_every_gpu(sched, tmp_path):
    c = sched.db()
    out = tmp_path / "n.txt"
    _submit(sched, "true", name="ddp", ngpu=2,
            notify=f"echo \"$GSCHED_GPUS|$GSCHED_GPU|$GSCHED_NGPU\" > {out}")
    procs = {}
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    _wait_exit(procs)
    sched._step(c, [0, 1], procs, 2000, mem_fn=_free(0, 1))
    _wait_hooks(sched)
    assert out.read_text().strip() == "0,1|0|2"
    assert "on GPUs 0,1" in sched._msg(_rows(c)[0])


def test_old_db_rows_migrate_to_the_gpu_list(sched, tmp_path):
    """A v2 DB (one gpu per job, no ngpu) opens and keeps its history readable."""
    import sqlite3
    path = str(tmp_path / "gsched.db")
    old = sqlite3.connect(path)
    old.executescript("""CREATE TABLE jobs(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
      cmd TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', gpu INTEGER, pid INTEGER,
      rc INTEGER, log TEXT, created REAL, started REAL, finished REAL);
      CREATE TABLE daemon(id INTEGER PRIMARY KEY CHECK (id=1), pid INTEGER, gpus TEXT,
      poll REAL, mem INTEGER, heartbeat REAL);
      INSERT INTO jobs(name,cmd,status,gpu,rc) VALUES('old','true','done',3,0);
      INSERT INTO jobs(name,cmd,status) VALUES('waiting','true','queued');""")
    old.commit()
    old.close()
    c = sched.db()                       # migrates in place
    rows = _rows(c)
    assert _gpus_of(rows[0]) == [3]      # v2 gpu -> v3 gpus list
    assert rows[0]["ngpu"] == 1 and rows[1]["ngpu"] == 1
    _submit(sched, "sleep 30", ngpu=2)   # and the migrated DB schedules normally
    procs = {}                           # the legacy queued row takes gpu0, ours takes 1,2
    assert sched._step(c, [0, 1, 2], procs, 2000, mem_fn=_free(0, 1, 2)) == 2
    assert _gpus_of(_rows(c)[2]) == [1, 2]
    for p in procs.values():
        p.terminate()


# ---------- notification ----------
def _wait_hooks(g, timeout=5):
    """Block until every fired notify hook has exited."""
    end = time.time() + timeout
    while time.time() < end and any(h.poll() is None for h in g._HOOKS):
        time.sleep(0.02)


def test_job_hook_fires_on_completion_with_env(sched, tmp_path):
    c = sched.db()
    out = tmp_path / "notified.txt"
    _submit(sched, "true", name="run-a",
            notify=f"echo \"$GSCHED_ID $GSCHED_NAME $GSCHED_STATUS $GSCHED_RC\" > {out}")
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_exit(procs)
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))   # reap -> done -> notify
    _wait_hooks(sched)
    assert out.read_text().split() == ["1", "run-a", "done", "0"]


def test_hook_reports_failure_rc(sched, tmp_path):
    c = sched.db()
    out = tmp_path / "n.txt"
    _submit(sched, "false", notify=f"echo $GSCHED_STATUS $GSCHED_RC > {out}")
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_exit(procs)
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_hooks(sched)
    assert out.read_text().split() == ["failed", "1"]


def test_daemon_wide_hook_applies_to_jobs_without_one(sched, tmp_path):
    c = sched.db()
    out = tmp_path / "d.txt"
    c.execute("INSERT INTO daemon(id,pid,gpus,poll,mem,heartbeat,notify) "
              "VALUES(1,1,'0',7,2000,0,?)", (f"echo $GSCHED_ID >> {out}",))
    c.commit()
    _submit(sched, "true")
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_exit(procs)
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_hooks(sched)
    assert out.read_text().split() == ["1"]


def test_job_hook_overrides_daemon_hook(sched, tmp_path):
    c = sched.db()
    dhook, jhook = tmp_path / "d.txt", tmp_path / "j.txt"
    c.execute("INSERT INTO daemon(id,pid,gpus,poll,mem,heartbeat,notify) "
              "VALUES(1,1,'0',7,2000,0,?)", (f"touch {dhook}",))
    c.commit()
    _submit(sched, "true", notify=f"touch {jhook}")
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_exit(procs)
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_hooks(sched)
    assert jhook.exists() and not dhook.exists()


def test_eviction_also_notifies(sched, tmp_path):
    c = sched.db()
    out = tmp_path / "e.txt"
    _submit(sched, "sleep 60", notify=f"echo $GSCHED_STATUS > {out}")
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    sched.cmd_evict(types.SimpleNamespace(id=1))
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_hooks(sched)
    assert out.read_text().strip() == "evicted"


def test_submit_records_tty_and_writes_to_it(sched, tmp_path, monkeypatch):
    """The submitting terminal is recorded, and the finish line is written to it.
    A regular file stands in for the tty - _write_tty just opens and writes."""
    c = sched.db()
    fake_tty = tmp_path / "tty"
    fake_tty.write_text("")
    monkeypatch.setattr(sched, "cur_tty", lambda: str(fake_tty))
    _submit(sched, "true", name="run-a", no_tty=False)
    assert _rows(c)[0]["tty"] == str(fake_tty)
    procs = {}
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    _wait_exit(procs)
    sched._step(c, [0], procs, 2000, mem_fn=_free(0))
    text = fake_tty.read_text()
    assert "job 1" in text and "run-a" in text and "done" in text and "rc=0" in text


def test_no_tty_records_nothing(sched, tmp_path, monkeypatch):
    monkeypatch.setattr(sched, "cur_tty", lambda: str(tmp_path / "tty"))
    _submit(sched, "true", no_tty=True)
    assert _rows(sched.db())[0]["tty"] is None


def test_dead_tty_and_broken_hook_do_not_break_scheduling(sched, tmp_path):
    """A gone terminal / nonsense hook must not stop the daemon reaping jobs."""
    c = sched.db()
    c.execute("INSERT INTO jobs(cmd,status,gpus,pid,started,tty,notify) "
              "VALUES('x','running','0',999999,0,?,'this-command-does-not-exist')",
              (str(tmp_path / "gone-tty"),))
    c.commit()
    sched._step(c, [0], {}, 2000, mem_fn=_free(0))
    assert _rows(c)[0]["status"] == "done"


def test_notify_env_var_is_the_default_hook(sched, tmp_path, monkeypatch):
    monkeypatch.setenv("GSCHED_NOTIFY", "touch /dev/null")
    _submit(sched, "true", notify=None)
    assert _rows(sched.db())[0]["notify"] == "touch /dev/null"


def test_daemon_restart_keeps_the_hook(sched, tmp_path, monkeypatch):
    c = sched.db()
    c.execute("INSERT INTO daemon(id,pid,gpus,poll,mem,heartbeat,notify) "
              "VALUES(1,1,'0',7,2000,0,'my-hook')")
    c.commit()
    # cmd_daemon re-registers then loops forever; stop it right after registration
    monkeypatch.setattr(sched.time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        sched.cmd_daemon(types.SimpleNamespace(gpus="0", poll=7, mem=2000, notify=None,
                                               backfill=False, no_backfill=False))
    assert c.execute("SELECT notify FROM daemon WHERE id=1").fetchone()["notify"] == "my-hook"


def test_backfill_survives_a_restart_and_can_be_turned_off(sched, monkeypatch):
    c = sched.db()
    stop = lambda *_: (_ for _ in ()).throw(KeyboardInterrupt)   # noqa: E731
    monkeypatch.setattr(sched.time, "sleep", stop)
    ns = dict(gpus="0", poll=7, mem=2000, notify=None)
    for kw, want in ((dict(backfill=True, no_backfill=False), 1),      # set it
                     (dict(backfill=False, no_backfill=False), 1),     # restart remembers
                     (dict(backfill=False, no_backfill=True), 0)):     # explicit off
        with pytest.raises(KeyboardInterrupt):
            sched.cmd_daemon(types.SimpleNamespace(**ns, **kw))
        assert c.execute("SELECT backfill FROM daemon WHERE id=1").fetchone()[0] == want


# ---------- restart resilience ----------
def test_reaps_orphaned_running_row_after_restart(sched):
    # simulate a running row whose process is already dead and NOT in our procs
    # (as if the daemon restarted): _step should mark it done.
    c = sched.db()
    c.execute("INSERT INTO jobs(cmd,status,gpus,pid,started) VALUES('x','running','0',999999,0)")
    c.commit()
    sched._step(c, [0], {}, 2000, mem_fn=_free(0))
    assert _rows(c)[0]["status"] == "done"
