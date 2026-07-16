# gsched

A tiny GPU job scheduler. One small daemon per machine owns a list of GPUs,
polls every few seconds, and runs the next **queued** job on any of its GPUs
that is **free**. Submit jobs from anywhere on the box; they queue and run as
GPUs open up. That's the whole idea.

- **One file** (`gsched.py`), one dependency (`rich`).
- **Shared-box friendly:** only uses the GPUs you name, treats a GPU as free
  only when its memory is actually low, and never touches other processes.
- **No server:** all state is a single SQLite file (`~/.gsched/gsched.db`) that
  the daemon and the CLI share.

## Install

```bash
pip install git+https://github.com/sudoRicheek/gsched.git
# or, from a clone:
git clone git@github.com:sudoRicheek/gsched.git && cd gsched && pip install -e .
```

This gives you a `gsched` command. (No install needed either — you can always
just run `python gsched.py ...`.)

## Quickstart

**1. Start the daemon** (once per machine, in the background). Name the GPUs it
may use; export any env vars your jobs need first — the daemon passes its
environment to every job.

```bash
nohup gsched daemon --gpus 0,1,2,3 > ~/.gsched/daemon.log 2>&1 &
```

**2. Submit jobs.** Write each job **as if it has one GPU** — the daemon sets
`CUDA_VISIBLE_DEVICES` for you.

```bash
gsched submit "python train.py --lr 3e-4" --name run-a     # one job
gsched submitf jobs.txt                                     # a whole file of jobs
```

**3. Watch.**

```bash
gsched status
```

```
● alive  daemon pid 12345  gpus 0,1,2,3  poll 7s  mem_free<2000MiB
 GPUs                    running                          queued
 gpu  mem    job         id name   gpu pid   elapsed      # id name
   0  17800  #7 run-a     7 run-a   0  9001   0:12:33      1 11 sweep-lr-0
   1  17800  #8 run-b     8 run-b   1  9002   0:12:31      2 12 sweep-lr-1
   2      1  idle
   3      1  idle
```

## Commands

| command | what it does |
|---|---|
| `daemon --gpus 0,1,2,3 [--poll 7] [--mem 2000]` | run the scheduler loop (background it) |
| `submit "<cmd>" [--name N]` | queue one command |
| `submitf <file> [--name N]` | queue every line of a file (`name<TAB>cmd` supported; `#`/blank lines skipped) |
| `status [--recent K]` | dashboard: daemon health, per-GPU, running, queued, last K finished |
| `gpus 0,1,2,3,4,5` | change the allowed GPU list **live** (no restart) |
| `evict <id>` | cancel a queued job, or `SIGTERM` a running one (its GPU frees for the next) |
| `logs <id> [-n K]` | tail that job's stdout/stderr |
| `rm --done` / `rm <id>` | forget finished rows |

### Change which GPUs are used, live

The allowed list lives in the DB and the daemon re-reads it every poll, so you
never restart it:

```bash
gsched gpus 0,1,2,3,4,5      # add 4 and 5
gsched gpus 0,1             # go back to just 0 and 1 (running jobs keep going)
```

## Specifying commands (the one thing to know)

Storage is safe — commands are stored verbatim in SQLite. Jobs run via `sh -c`
(so inline env vars, redirects, `&&`, globs all work), which means there are two
shell layers:

- **At submit time** your shell parses `submit "..."`, so quote the whole
  command. Plain commands like `python train.py +a=b +c=d` need nothing special.
- **At run time** `sh -c` re-parses it; `$VAR` expands in the **daemon's**
  environment, not your submit shell's.
- For commands containing quotes, prefer **`submitf`** (each line is taken as-is,
  no submit-time quoting). Its only caveat: the optional name is split on the
  first **TAB**.

**Don't set `CUDA_VISIBLE_DEVICES` yourself** — the daemon owns it. Write jobs as
single-GPU (`cuda:0`).

## How it works

Each job is a row with a `status`: **queued → running → done / failed / evicted**.
Every `--poll` seconds the daemon:

1. **reaps** finished processes (records exit code → `done`/`failed`);
2. **evicts** any job you asked to kill;
3. **dispatches** the oldest `queued` job onto each allowed GPU that is free
   (memory `< --mem` MiB *and* not already running one of our jobs), FIFO.

Logs go to `~/.gsched/logs/job<id>.log`. The daemon can be restarted anytime — it
re-reads the DB, and jobs whose process died while it was down are marked done on
the next poll.

## Tests

```bash
pytest
```

`tests/test_gsched.py` drives the scheduling core (`_step`) with fake GPU memory
and trivial jobs (`true`/`false`/`sleep`) — the full reap/evict/dispatch logic is
verified deterministically, no real GPUs needed.

## License

MIT
