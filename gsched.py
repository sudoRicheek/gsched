#!/usr/bin/env python3
"""gsched - a tiny single-file GPU job scheduler (daemon + CLI).

One daemon per machine owns a list of GPUs. Every few seconds it checks which
of those GPUs are free (low used-memory AND not already running one of our jobs)
and launches the next queued job on each free GPU. That's the whole scheduling
logic - a poll loop over a FIFO queue.

Coordination is a single SQLite file (~/.gsched/gsched.db): the daemon and the
CLI both just read/write it, so there are no sockets or servers. Submitting a
job = inserting a 'queued' row; evicting = flipping a row's status; the daemon
picks the changes up on its next poll. rich renders the dashboard.

  # start the daemon once per box (nohup it), naming the allowed GPUs:
  python gsched.py daemon --gpus 0,1,2,3
  # then from anywhere on that box (same $HOME):
  python gsched.py submit "python train.py ..."  --name run-a
  python gsched.py submitf jobs.txt                       # queue every line
  python gsched.py status                                 # dashboard
  python gsched.py evict 7                                # cancel/kill job 7
  python gsched.py logs 7                                 # tail job 7's log
  python gsched.py rm --done                              # forget finished rows
"""
import argparse
import os
import signal
import sqlite3
import subprocess
import sys
import time

GDIR = os.path.join(os.path.expanduser("~"), ".gsched")
DB = os.path.join(GDIR, "gsched.db")
LOGDIR = os.path.join(GDIR, "logs")
MEM_FREE_MB = 2000     # a GPU with less used-memory than this counts as "free"
POLL_S = 7             # daemon poll interval

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  name     TEXT,
  cmd      TEXT NOT NULL,
  status   TEXT NOT NULL DEFAULT 'queued',   -- queued running done failed evicted
  gpu      INTEGER,
  pid      INTEGER,
  rc       INTEGER,
  log      TEXT,
  created  REAL,
  started  REAL,
  finished REAL
);
CREATE TABLE IF NOT EXISTS daemon(
  id INTEGER PRIMARY KEY CHECK (id = 1),
  pid INTEGER, gpus TEXT, poll REAL, mem INTEGER, heartbeat REAL
);
"""


# ---------- storage ----------
def db():
    os.makedirs(GDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=15000")
    c.executescript(SCHEMA)
    return c


# ---------- gpu / process helpers ----------
def gpu_mem():
    """{gpu_index: used_MiB} from nvidia-smi; {} if it is unavailable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
    except FileNotFoundError:
        return {}
    out = {}
    for line in r.stdout.strip().splitlines():
        idx, mem = line.split(",")
        out[int(idx)] = int(mem)
    return out


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


# ---------- scheduling core (one poll iteration; unit-tested) ----------
def _finish(c, jid, status, rc=None):
    c.execute("UPDATE jobs SET status=?, rc=?, finished=? WHERE id=?",
              (status, rc, time.time(), jid))


def _launch(c, row, gpu):
    """Start one queued row on `gpu`, mark it running, return its Popen."""
    log = os.path.join(LOGDIR, f"job{row['id']}.log")
    os.makedirs(LOGDIR, exist_ok=True)
    f = open(log, "a")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    p = subprocess.Popen(row["cmd"], shell=True, env=env,
                         stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    c.execute("UPDATE jobs SET status='running', gpu=?, pid=?, started=?, log=? WHERE id=?",
              (gpu, p.pid, time.time(), log, row["id"]))
    return p


def _step(c, gpus, procs, mem_thresh, mem_fn=gpu_mem):
    """One scheduling iteration: reap finished jobs, carry out evictions, and
    launch queued jobs onto free allowed GPUs. `procs` maps job_id->Popen for
    this session; `mem_fn` returns {gpu: used_MiB} (injectable for tests).
    Returns the number of jobs launched this step."""
    # 1. reap jobs we launched this session (exact return code)
    for jid, p in list(procs.items()):
        rc = p.poll()
        if rc is not None:
            _finish(c, jid, "done" if rc == 0 else "failed", rc)
            del procs[jid]
    # 2. reap running rows whose process is gone (daemon may have restarted)
    for row in c.execute("SELECT id,pid FROM jobs WHERE status='running'").fetchall():
        if row["id"] not in procs and not alive(row["pid"]):
            _finish(c, row["id"], "done")
    # 3. carry out CLI-requested evictions
    for row in c.execute("SELECT id,pid FROM jobs WHERE status='evicting'").fetchall():
        if alive(row["pid"]):
            try:
                os.killpg(os.getpgid(int(row["pid"])), signal.SIGTERM)
            except OSError:
                pass
        procs.pop(row["id"], None)  # else the next reap re-marks it 'failed'
        _finish(c, row["id"], "evicted")
    c.commit()
    # 4. dispatch: one queued job onto each free allowed GPU
    mem = mem_fn()
    busy = {r["gpu"] for r in c.execute("SELECT gpu FROM jobs WHERE status='running'")}
    launched = 0
    for g in gpus:
        if g in busy or mem.get(g, 10**9) >= mem_thresh:
            continue
        row = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
        if not row:
            break
        procs[row["id"]] = _launch(c, row, g)
        busy.add(g)
        launched += 1
        c.commit()
        print(f"launched job {row['id']} ({row['name'] or ''}) on GPU{g} "
              f"pid={procs[row['id']].pid}", flush=True)
    c.commit()
    return launched


# ---------- the daemon (loop over _step) ----------
def cmd_daemon(a):
    c = db()
    c.execute("INSERT OR REPLACE INTO daemon(id,pid,gpus,poll,mem,heartbeat) VALUES(1,?,?,?,?,?)",
              (os.getpid(), a.gpus, a.poll, a.mem, time.time()))
    c.commit()
    procs = {}  # job_id -> Popen
    print(f"gsched daemon pid={os.getpid()} gpus={a.gpus} poll={a.poll}s mem_free<{a.mem}MiB",
          flush=True)
    while True:
        # re-read config each loop so `gsched gpus ...` / `mem ...` apply live
        d = c.execute("SELECT gpus,mem,poll FROM daemon WHERE id=1").fetchone()
        gpus = [int(g) for g in d["gpus"].split(",")]
        _step(c, gpus, procs, d["mem"])
        c.execute("UPDATE daemon SET heartbeat=?, pid=? WHERE id=1", (time.time(), os.getpid()))
        c.commit()
        time.sleep(d["poll"])


def cmd_gpus(a):
    """Change the daemon's allowed GPU list live (no restart)."""
    c = db()
    if not c.execute("SELECT 1 FROM daemon WHERE id=1").fetchone():
        print("no daemon registered yet - start it first")
        return
    c.execute("UPDATE daemon SET gpus=? WHERE id=1", (a.gpus,))
    c.commit()
    print(f"allowed GPUs -> {a.gpus} (daemon applies it on the next poll)")


# ---------- CLI verbs ----------
def cmd_submit(a):
    c = db()
    cur = c.execute("INSERT INTO jobs(name,cmd,status,created) VALUES(?,?,'queued',?)",
                    (a.name, a.cmd, time.time()))
    c.commit()
    print(f"queued job {cur.lastrowid}")


def cmd_submitf(a):
    c = db()
    n = 0
    for line in open(a.file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = a.name
        # allow "name<TAB>command" lines for readable batches
        if "\t" in line:
            name, line = line.split("\t", 1)
        c.execute("INSERT INTO jobs(name,cmd,status,created) VALUES(?,?,'queued',?)",
                  (name, line.strip(), time.time()))
        n += 1
    c.commit()
    print(f"queued {n} jobs from {a.file}")


def cmd_evict(a):
    c = db()
    row = c.execute("SELECT status FROM jobs WHERE id=?", (a.id,)).fetchone()
    if not row:
        print(f"no job {a.id}")
        return
    if row["status"] == "queued":
        c.execute("UPDATE jobs SET status='evicted', finished=? WHERE id=?", (time.time(), a.id))
        print(f"cancelled queued job {a.id}")
    elif row["status"] == "running":
        c.execute("UPDATE jobs SET status='evicting' WHERE id=?", (a.id,))
        print(f"kill requested for running job {a.id} (daemon will SIGTERM its group)")
    else:
        print(f"job {a.id} is already {row['status']}")
    c.commit()


def cmd_rm(a):
    c = db()
    if a.all:
        c.execute("DELETE FROM jobs WHERE status IN ('done','failed','evicted')")
    elif a.done:
        c.execute("DELETE FROM jobs WHERE status IN ('done','failed','evicted')")
    elif a.id:
        c.execute("DELETE FROM jobs WHERE id=? AND status!='running'", (a.id,))
    c.commit()
    print("removed")


def cmd_logs(a):
    c = db()
    row = c.execute("SELECT log FROM jobs WHERE id=?", (a.id,)).fetchone()
    if not row or not row["log"] or not os.path.exists(row["log"]):
        print("no log yet")
        return
    subprocess.run(["tail", "-n", str(a.n), row["log"]])


def _dur(a, b):
    if not a:
        return "-"
    s = int((b or time.time()) - a)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def cmd_status(a):
    from rich.console import Console
    from rich.table import Table
    from rich import box
    c = db()
    con = Console()

    d = c.execute("SELECT * FROM daemon WHERE id=1").fetchone()
    allowed = [int(g) for g in d["gpus"].split(",")] if d else []
    if d:
        age = time.time() - d["heartbeat"]
        state = "[bold green]● alive[/]" if age < d["poll"] * 3 else f"[bold red]● stale {int(age)}s[/]"
        con.print(f"{state}  daemon pid [cyan]{d['pid']}[/]  gpus [cyan]{d['gpus']}[/]  "
                  f"poll {d['poll']:g}s  mem_free<{d['mem']}MiB")
    else:
        con.print("[bold red]no daemon registered[/] - start:  python gsched.py daemon --gpus 0,1,2,3")

    # GPU panel
    mem = gpu_mem()
    onpu = {r["gpu"]: r for r in c.execute("SELECT * FROM jobs WHERE status='running'")}
    gt = Table(box=box.SIMPLE_HEAVY, title="GPUs", title_style="bold", expand=False)
    gt.add_column("gpu", justify="right")
    gt.add_column("mem MiB", justify="right")
    gt.add_column("job")
    for g in sorted(set(list(mem.keys())) | set(allowed)):
        j = onpu.get(g)
        tag = "[dim]not managed[/]" if g not in allowed else (
            f"[green]#{j['id']} {j['name'] or ''}[/]" if j else "[yellow]idle[/]")
        m = mem.get(g, 0)
        mc = "green" if m < (d["mem"] if d else MEM_FREE_MB) else "red"
        gt.add_row(str(g), f"[{mc}]{m}[/]", tag)
    con.print(gt)

    def jtable(title, rows, cols, style):
        t = Table(box=box.SIMPLE_HEAVY, title=title, title_style=f"bold {style}", expand=False)
        for name, j in cols:
            t.add_column(name, justify=j)
        for r in rows:
            t.add_row(*[str(x) for x in r])
        if rows:
            con.print(t)

    run = c.execute("SELECT * FROM jobs WHERE status='running' ORDER BY gpu").fetchall()
    jtable("running", [(r["id"], r["name"] or "", r["gpu"], r["pid"],
                        _dur(r["started"], None), (r["cmd"][:60] + "…") if len(r["cmd"]) > 60 else r["cmd"])
                       for r in run],
           [("id", "right"), ("name", "left"), ("gpu", "right"), ("pid", "right"),
            ("elapsed", "right"), ("cmd", "left")], "green")

    q = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id").fetchall()
    jtable("queued", [(i + 1, r["id"], r["name"] or "",
                       (r["cmd"][:70] + "…") if len(r["cmd"]) > 70 else r["cmd"])
                      for i, r in enumerate(q)],
           [("#", "right"), ("id", "right"), ("name", "left"), ("cmd", "left")], "yellow")

    fin = c.execute("SELECT * FROM jobs WHERE status IN ('done','failed','evicted') "
                    "ORDER BY finished DESC LIMIT ?", (a.recent,)).fetchall()
    col = {"done": "green", "failed": "red", "evicted": "magenta"}
    jtable(f"finished (last {a.recent})",
           [(r["id"], r["name"] or "", f"[{col[r['status']]}]{r['status']}[/]",
             "-" if r["rc"] is None else r["rc"], _dur(r["started"], r["finished"]))
            for r in fin],
           [("id", "right"), ("name", "left"), ("status", "left"),
            ("rc", "right"), ("dur", "right")], "white")

    n = {s: c.execute("SELECT count(*) FROM jobs WHERE status=?", (s,)).fetchone()[0]
         for s in ("queued", "running", "done", "failed", "evicted")}
    con.print(f"[dim]queued {n['queued']}  running {n['running']}  done {n['done']}  "
              f"failed {n['failed']}  evicted {n['evicted']}[/]")


def main():
    p = argparse.ArgumentParser(description="tiny single-file GPU scheduler")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daemon", help="run the scheduler loop (nohup this)")
    d.add_argument("--gpus", required=True, help="comma list, e.g. 0,1,2,3")
    d.add_argument("--poll", type=float, default=POLL_S)
    d.add_argument("--mem", type=int, default=MEM_FREE_MB, help="free-GPU memory threshold MiB")
    d.set_defaults(fn=cmd_daemon)

    s = sub.add_parser("submit", help="queue one command")
    s.add_argument("cmd")
    s.add_argument("--name", default=None)
    s.set_defaults(fn=cmd_submit)

    sf = sub.add_parser("submitf", help="queue every line of a file (optional name<TAB>cmd)")
    sf.add_argument("file")
    sf.add_argument("--name", default=None)
    sf.set_defaults(fn=cmd_submitf)

    g = sub.add_parser("gpus", help="change the daemon's allowed GPU list live")
    g.add_argument("gpus", help="comma list, e.g. 0,1,2,3,4,5")
    g.set_defaults(fn=cmd_gpus)

    e = sub.add_parser("evict", help="cancel a queued job or kill a running one")
    e.add_argument("id", type=int)
    e.set_defaults(fn=cmd_evict)

    r = sub.add_parser("rm", help="forget finished rows")
    r.add_argument("id", type=int, nargs="?")
    r.add_argument("--done", action="store_true")
    r.add_argument("--all", action="store_true")
    r.set_defaults(fn=cmd_rm)

    lg = sub.add_parser("logs", help="tail a job's log")
    lg.add_argument("id", type=int)
    lg.add_argument("-n", type=int, default=40)
    lg.set_defaults(fn=cmd_logs)

    st = sub.add_parser("status", help="dashboard")
    st.add_argument("--recent", type=int, default=8)
    st.set_defaults(fn=cmd_status)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
